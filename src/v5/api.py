import os
import sys
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import base64
import queue
import threading
import json
import asyncio
from music21 import stream, note, tempo, meter, key, midi

def notes_to_midi_base64(notes_list):
    s = stream.Score()
    s.insert(0, tempo.MetronomeMark(number=120))
    s.insert(0, meter.TimeSignature('4/4'))
    s.insert(0, key.Key('C'))
    
    parts = {1: stream.Part(), 2: stream.Part(), 3: stream.Part(), 4: stream.Part()}
    for v in parts.values():
        s.insert(0, v)
        
    for n in notes_list:
        m21_note = note.Note(n['pitch'])
        m21_note.quarterLength = n['duration']
        parts[n['voice']].insert(n['offset'], m21_note)
        
    try:
        # music21 객체를 MidiFile 객체로 변환하여 메모리 버퍼에 기록
        mf = midi.translate.music21ObjectToMidiFile(s)
        buf = io.BytesIO()
        mf.openFileLike(buf)
        mf.write()
        
        # seek(0) 포인터 초기화로 빈 바이트 스트리밍 방지
        buf.seek(0)
        
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return b64
    except Exception as e:
        print(f"Error converting notes to MIDI base64: {e}")
        return ""

# Project root 추가
sys.path.append(os.getcwd())

from src.v5.neural_engine import NeuralBachEngine, HybridFugueEngine

app = FastAPI(title="Like Bach v5 API Engine")

# CORS 설정 (Frontend 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발 단계에서는 모두 허용, 운영 시 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 엔진 초기화 (싱글톤 방식)
engine = None
fugue_engine_v5 = None

@app.on_event("startup")
async def startup_event():
    global engine, fugue_engine_v5
    print("Loading Neural Bach Engine v4.5...")
    try:
        engine = NeuralBachEngine(
            model_path='data/processed/v4/bach_model.pt',
            tokenizer_path='data/processed/v4/tokenizer.pkl'
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
    temperature: float = 0.55  # Reduced from 0.8 for more stable/less chaotic Fugue generation
    refine_iters: int = 3
    voices: int = 2  # default Phase 1 invention is 2-part

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "engine_ready": engine is not None}

@app.post("/api/generate")
async def generate_bach(request: GenerationRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")
    
    try:
        # Pydantic 모델을 dict 리스트로 변환
        notes_dict = [n.dict() for n in request.subject_notes]
        
        # AI 생성 수행 (target_measures 명시적 전달)
        generated_notes = engine.generate_response(
            subject_notes=notes_dict,
            target_measures=request.target_measures if request.target_measures > 0 else 8,
            temperature=request.temperature,
            refine_iters=request.refine_iters
        )
        
        return {
            "success": True,
            "results": generated_notes
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream_fugue")
async def stream_fugue_bach(request: GenerationRequestFugue):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")
    
    notes_dict = [n.dict() for n in request.subject_notes]
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
                stream_queue=stream_q
            )
            
            midi_b64 = notes_to_midi_base64(final_notes)
            
            stream_q.put({
                "type": "done",
                "notes": final_notes,
                "midi_base64": midi_b64
            })
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
