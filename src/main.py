from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Correctly import models for pickle deserialization
from src.models import BachTokenizer, BachTransformer

app = FastAPI(title="BPGE Backend - Neural Hybrid v3.0")
neural_engine = None

# Neural Engine 초기화 시도
try:
    from src.neural_engine import NeuralBachEngine
    model_path = os.path.join('data', 'processed', 'bach_model.pt')
    if os.path.exists(model_path):
        neural_engine = NeuralBachEngine()
        print(">>> Neural Bach Engine Integrated Successfully. (v3.0 ready)")
except Exception as e:
    print(f">>> Neural Engine loading deferred or failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteData(BaseModel):
    """단일 음표의 물리적 화성학 명세서입니다."""
    pitch: int
    duration: float
    offset: float

class SubjectMelody(BaseModel):
    """사용자가 입력한 초기 소프라노 선율 시퀀스 집합입니다."""
    notes: List[NoteData]

@app.get("/")
def read_root() -> Dict[str, str]:
    """백엔드 작곡 서버의 상태 체크 및 현재 구동 중인 하드웨어 엔진 모드를 반환합니다."""
    mode = "Neural Only" if neural_engine else "Offline"
    return {"status": "online", "engine": "BPGE v3.0", "mode": mode}

@app.post("/compose")
def compose_piece(melody: SubjectMelody) -> Dict[str, list]:
    """주제를 입력받아 전체 곡을 작곡합니다."""
    if not melody.notes:
        raise HTTPException(status_code=400, detail="No notes provided")
    
    if not neural_engine:
        raise HTTPException(status_code=503, detail="Neural Engine is offline")
        
    notes_dict = [n.dict() for n in melody.notes]
    
    try:
        print(">>> Attempting neural generation for high-quality harmony...")
        neural_resp = neural_engine.generate_response(notes_dict)
        return {
            "key": "C",
            "part1": notes_dict,
            "part2": neural_resp if neural_resp else [],
            "duration_total": max([n['offset'] + n['duration'] for n in notes_dict]) if notes_dict else 0,
            "mode": "Neural Hybrid v3.0"
        }
    except Exception as e:
        print(f">>> Neural generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
