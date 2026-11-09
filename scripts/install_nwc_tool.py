"""Installer script to register Like_Bach as NoteWorthy Composer 2 (NWC2) User Tools."""

from __future__ import annotations

import os
import subprocess
import sys
import winreg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_TOOL_SCRIPT = os.path.join(PROJECT_ROOT, "src", "nwc", "nwc_user_tool.py")


def get_nwc_ini_targets() -> list[str]:
    """Locate all active NWC2 user tool configuration file paths on this machine."""
    targets: list[str] = []

    local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser(r"~\AppData\Local"))
    primary_dir = os.path.join(local_appdata, "Noteworthy Software", "nwc2")
    targets.append(os.path.join(primary_dir, "nwc2UserTools.ini"))

    appdata = os.environ.get("APPDATA", os.path.expanduser(r"~\AppData\Roaming"))
    roaming_dir = os.path.join(appdata, "Noteworthy Software", "nwc2")
    if os.path.exists(os.path.dirname(roaming_dir)):
        targets.append(os.path.join(roaming_dir, "nwc2UserTools.ini"))

    # Also check if a custom Config folder is configured in HKCU registry
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Noteworthy Software\NoteWorthy Composer 2\Folders",
            0,
            winreg.KEY_READ,
        ) as key:
            cfg_dir, _ = winreg.QueryValueEx(key, "Config")
            if cfg_dir and os.path.isdir(cfg_dir):
                cand = os.path.join(cfg_dir, "nwc2UserTools.ini")
                if cand not in targets:
                    targets.append(cand)
    except OSError:
        pass

    return targets


def build_tool_definitions(python_exe: str, script_path: str) -> dict[str, str]:
    """
    Build the NWC2 User Tool definitions using NWC2's native INI format:
    ToolName='<flags>,<command_line>'
    Flags:
      5 = Input: File Text (bit 0) + Returns File Text (bit 2) -> Full multi-staff score expansion
      0 = Input: Clip Text -> Selected notation clip transformation
    """
    base_cmd = f'"{python_exe}" "{script_path}"'
    return {
        "Like Bach: 4-Voice Chorale (Full Score)": (
            f"'5,{base_cmd} --mode chorale --measures 8'"
        ),
        "Like Bach: 4-Voice Fugue (Full Score)": (
            f"'5,{base_cmd} --mode fugue --measures 8'"
        ),
        "Like Bach: Custom Generator (Prompt)": (
            f"'5,{base_cmd} "
            f"--mode <PROMPT:Generation Mode:=|Fugue|Chorale> "
            f"--measures <PROMPT:Target Measures:=#[4,64]> "
            f"--temp <PROMPT:Temperature:=|0.7|0.55|0.8|0.9>'"
        ),
        "Like Bach: Harmonize Selected Clip (Clip Mode)": (
            f"'0,{base_cmd} --mode chorale --clip-style chord'"
        ),
    }


def update_ini_file(ini_path: str, tool_defs: dict[str, str]) -> None:
    """Update or create nwc2UserTools.ini while preserving any existing user tools."""
    os.makedirs(os.path.dirname(ini_path), exist_ok=True)

    sections: dict[str, dict[str, str]] = {"Default": {}}
    current_sec = "Default"

    if os.path.exists(ini_path):
        try:
            with open(ini_path, "r", encoding="mbcs", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith(";"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        current_sec = line[1:-1].strip()
                        sections.setdefault(current_sec, {})
                    elif "=" in line:
                        k, v = line.split("=", 1)
                        sections.setdefault(current_sec, {})[k.strip()] = v.strip()
        except OSError:
            pass

    for name, val in tool_defs.items():
        sections["Default"][name] = val

    out_lines: list[str] = []
    for sec_name, kv_map in sections.items():
        out_lines.append(f"[{sec_name}]")
        for k, v in kv_map.items():
            out_lines.append(f"{k}={v}")
        out_lines.append("")

    with open(ini_path, "w", encoding="mbcs", errors="replace") as f:
        f.write("\r\n".join(out_lines))


def update_nwc_registry(default_tool_name: str) -> None:
    """Set the default selected tool group and tool name in NWC2 HKCU registry."""
    reg_path = r"Software\Noteworthy Software\NoteWorthy Composer 2\Options"
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            reg_path,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "LastToolGrp", 0, winreg.REG_SZ, "Default")
            winreg.SetValueEx(key, "LastToolRun", 0, winreg.REG_SZ, default_tool_name)
    except OSError as exc:
        print(f"Warning: Could not update NWC2 registry Options: {exc}")


def verify_user_tool_execution(python_exe: str, script_path: str) -> bool:
    """Run a quick smoke test passing sample NWC2 score text to the user tool script."""
    sample_nwctxt = "\n".join([
        "!NoteWorthyComposer(2.75)",
        "|AddStaff|Name:\"Soprano\"|Group:\"Standard\"",
        "|Clef|Type:Treble",
        "|Key|Signature:C|Tonic:C",
        "|TimeSig|Signature:4/4",
        "|Note|Dur:4th|Pos:-6",
        "|Note|Dur:4th|Pos:-5",
        "|Note|Dur:4th|Pos:-4",
        "|Note|Dur:4th|Pos:-3",
        "!NoteWorthyComposer-End",
    ])
    proc = subprocess.run(
        [python_exe, script_path, "--mode", "fugue", "--measures", "4"],
        input=sample_nwctxt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        print(f"Verification failed (exit {proc.returncode}): {proc.stderr}")
        return False
    out = proc.stdout
    return (
        out.startswith("!NoteWorthyComposer(2.75)")
        and '|AddStaff|Name:"Soprano"' in out
        and '|AddStaff|Name:"Alto"' in out
        and '|AddStaff|Name:"Tenor"' in out
        and '|AddStaff|Name:"Bass"' in out
        and "!NoteWorthyComposer-End" in out
    )


def main() -> int:
    python_exe = sys.executable
    if not os.path.exists(USER_TOOL_SCRIPT):
        print(f"Error: User tool script not found at {USER_TOOL_SCRIPT}")
        return 1

    tool_defs = build_tool_definitions(python_exe, USER_TOOL_SCRIPT)
    targets = get_nwc_ini_targets()

    for ini_path in targets:
        update_ini_file(ini_path, tool_defs)
        print(f"Registered Like_Bach NWC2 User Tools in: {ini_path}")

    update_nwc_registry("Like Bach: 4-Voice Fugue (Full Score)")
    print("Updated NWC2 Registry Options (LastToolGrp=Default, LastToolRun=Like Bach: 4-Voice Fugue (Full Score)).")

    ok = verify_user_tool_execution(python_exe, USER_TOOL_SCRIPT)
    if not ok:
        print("Error: Smoke verification of NWC2 user tool failed.")
        return 1

    print("Verification passed: 4-voice NWC2 score generation confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
