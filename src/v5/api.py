import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import os
import queue
import sys
import threading
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from music21 import key, meter, midi, note, stream, tempo
from pydantic import BaseModel

sys.path.append(os.getcwd())

from src.v5.evaluator import evaluate_composition_official
from src.v5.neural_engine import HybridFugueEngine, NeuralBachEngine


def notes_to_midi_base64(notes_list):
    s = stream.Score()
    s.insert(0, tempo.MetronomeMark(number=120))
    s.insert(0, meter.TimeSignature("4/4"))
    s.insert(0, key.Key("C"))

    parts = {1: stream.Part(), 2: stream.Part(), 3: stream.Part(), 4: stream.Part()}
    for v in parts.values():
        s.insert(0, v)

    for n in notes_list:
        m21_note = note.Note(int(n["pitch"]))
        m21_note.quarterLength = float(n["duration"])
        v_id = int(n.get("voice", 1))
        if v_id not in parts:
            v_id = max(1, min(4, v_id))
        parts[v_id].insert(float(n["offset"]), m21_note)

    try:
        mf = midi.translate.music21ObjectToMidiFile(s)
        buf = io.BytesIO()
        mf.openFileLike(buf)
        mf.write()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting notes to MIDI base64: {e}")
        return ""


engine = None
fugue_engine_v5 = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, fugue_engine_v5
    print("Loading Neural Bach Engine v4.5...")
    try:
        engine = NeuralBachEngine(
            model_path="data/processed/v4/bach_model.pt",
            tokenizer_path="data/processed/v4/tokenizer.pkl",
        )
        print("Neural Bach Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading AI Engine: {e}")

    print("Loading V5 Hybrid Fugue Engine...")
    try:
        fugue_engine_v5 = HybridFugueEngine()
        print("V5 Hybrid Fugue Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading V5 Fugue Engine: {e}")
    yield


app = FastAPI(title="Like Bach v5 API Engine", lifespan=lifespan)

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
cors_env = os.environ.get("LIKE_BACH_CORS_ORIGINS", "")
allowed_origins = (
    [o.strip() for o in cors_env.split(",") if o.strip()]
    if cors_env
    else default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Note(BaseModel):
    pitch: int
    duration: float
    offset: float
    voice: int = 1


class GenerationRequest(BaseModel):
    subject_notes: List[Note]
    target_measures: int = 16
    temperature: float = 0.8
    refine_iters: int = 3


class GenerationRequestFugue(BaseModel):
    subject_notes: List[Note]
    target_measures: int = 16
    temperature: float = 0.55
    refine_iters: int = 3
    voices: int = 2


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "fugue_engine_v5_ready": fugue_engine_v5 is not None,
    }


@app.post("/api/generate")
async def generate_bach(request: GenerationRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")

    try:
        notes_dict = [n.model_dump() for n in request.subject_notes]
        generated_notes = engine.generate_response(
            subject_notes=notes_dict,
            target_measures=request.target_measures if request.target_measures > 0 else 8,
            temperature=request.temperature,
            refine_iters=request.refine_iters,
        )

        midi_b64 = notes_to_midi_base64(generated_notes)
        official_eval = getattr(engine, "last_official_evaluation", None)
        if official_eval is None:
            official_eval = evaluate_composition_official(generated_notes)

        return {
            "success": True,
            "results": generated_notes,
            "midi_base64": midi_b64,
            "official_evaluation": official_eval,
        }
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream_fugue")
async def stream_fugue_bach(request: GenerationRequestFugue):
    global engine, fugue_engine_v5
    if engine is None and fugue_engine_v5 is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")

    notes_dict = [n.model_dump() for n in request.subject_notes]
    target_m = request.target_measures if request.target_measures > 0 else 8

    stream_q = queue.Queue()

    def generate_worker():
        try:
            target_engine = fugue_engine_v5 if fugue_engine_v5 else engine
            final_notes = target_engine.generate_fugue(
                subject_notes=notes_dict,
                target_measures=target_m,
                temperature=request.temperature,
                refine_iters=request.refine_iters,
                stream_queue=stream_q,
            )

            midi_b64 = notes_to_midi_base64(final_notes)
            official_eval = evaluate_composition_official(final_notes)

            stream_q.put(
                {
                    "type": "done",
                    "notes": final_notes,
                    "midi_base64": midi_b64,
                    "official_evaluation": official_eval,
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            stream_q.put({"type": "error", "message": str(e)})

    threading.Thread(target=generate_worker, daemon=True).start()

    async def event_generator():
        while True:
            try:
                msg = stream_q.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ["done", "error"]:
                    break
            except queue.Empty:
                await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
