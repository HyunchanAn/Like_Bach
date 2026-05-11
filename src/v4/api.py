import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Project root 추가
sys.path.append(os.getcwd())

from src.v4.neural_engine import NeuralBachEngine

app = FastAPI(title="Like Bach v4.5 API Engine")

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

@app.on_event("startup")
async def startup_event():
    global engine
    print("Loading Neural Bach Engine v4.5...")
    try:
        engine = NeuralBachEngine(
            model_path='data/processed/v4/bach_model.pt',
            tokenizer_path='data/processed/v4/tokenizer.pkl'
        )
        print("Neural Bach Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading AI Engine: {e}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
