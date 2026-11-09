"""Download and verify Like_Bach model weights and tokenizers from Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "chemahc94/like-bach-models"
PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACT_MAP = {
    "v4/bach_model.pt": PROJECT_ROOT / "data" / "processed" / "v4" / "bach_model.pt",
    "v4/bach_model.safetensors": (
        PROJECT_ROOT / "data" / "processed" / "v4" / "bach_model.safetensors"
    ),
    "v4/bach_model.onnx": PROJECT_ROOT / "data" / "processed" / "v4" / "bach_model.onnx",
    "v4/tokenizer.pkl": PROJECT_ROOT / "data" / "processed" / "v4" / "tokenizer.pkl",
    "v5/fugue_model_v5.pt": (
        PROJECT_ROOT / "data" / "processed" / "v5" / "fugue_model_v5.pt"
    ),
    "v5/fugue_model_v5.safetensors": (
        PROJECT_ROOT / "data" / "processed" / "v5" / "fugue_model_v5.safetensors"
    ),
    "v5/fugue_model_v5.onnx": (
        PROJECT_ROOT / "data" / "processed" / "v5" / "fugue_model_v5.onnx"
    ),
    "v5/fugue_vocab_v5.pkl": (
        PROJECT_ROOT / "data" / "processed" / "v5" / "fugue_vocab_v5.pkl"
    ),
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(
    repo_id: str = DEFAULT_REPO_ID,
    formats: tuple[str, ...] = (".pt", ".safetensors", ".onnx", ".pkl"),
    force: bool = False,
) -> bool:
    manifest_local = PROJECT_ROOT / "data" / "processed" / "release_manifest.json"
    if not manifest_local.exists() or force:
        cached_manifest = hf_hub_download(repo_id=repo_id, filename="release_manifest.json")
        manifest_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_manifest, manifest_local)

    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
    expected_artifacts = manifest.get("artifacts", {})

    all_ok = True
    for remote_key, local_path in ARTIFACT_MAP.items():
        if not any(remote_key.endswith(ext) for ext in formats):
            continue

        expected_meta = expected_artifacts.get(remote_key, {})
        expected_sha = expected_meta.get("sha256")

        if local_path.exists() and not force and expected_sha:
            actual_sha = compute_sha256(local_path)
            if actual_sha == expected_sha:
                print(f"[OK] Cached & verified: {remote_key} ({actual_sha[:12]}...)")
                continue

        print(f"[DOWNLOAD] Fetching {remote_key} from {repo_id}...")
        cached_file = Path(hf_hub_download(repo_id=repo_id, filename=remote_key))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, local_path)

        if expected_sha:
            actual_sha = compute_sha256(local_path)
            if actual_sha != expected_sha:
                print(
                    f"[ERROR] SHA-256 mismatch for {remote_key}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
                all_ok = False
            else:
                print(f"[OK] Downloaded & verified: {remote_key} ({actual_sha[:12]}...)")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify Like_Bach model weights from Hugging Face Hub."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if local hash matches"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Download only .pt and .pkl files required for default runtime",
    )
    args = parser.parse_args()

    formats = (".pt", ".pkl") if args.minimal else (".pt", ".safetensors", ".onnx", ".pkl")
    ok = download_all(repo_id=args.repo_id, formats=formats, force=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
