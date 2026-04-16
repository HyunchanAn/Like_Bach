from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.v4.neural_engine import NeuralBachEngine

app = FastAPI(title="Like Bach v4.0 - Master Bach API")

# Initialize Neural Engine v4.0
neural_engine = None
try:
    # paths are relative to project root
    model_path = os.path.join('data', 'processed', 'v4', 'bach_model.pt')
    tokenizer_path = os.path.join('data', 'processed', 'v4', 'tokenizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        neural_engine = NeuralBachEngine(model_path=model_path, tokenizer_path=tokenizer_path)
        print(">>> Neural Bach Engine v4.0 (4-Voice) Loaded Successfully.")
    else:
        print(f">>> Warning: V4.0 Model or Tokenizer not found at {model_path}")
except Exception as e:
    print(f">>> Neural Engine loading failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteData(BaseModel):
    pitch: int
    duration: float
    offset: float
    voice: Optional[int] = 1 # 1: Soprano, 2: Alto, 3: Tenor, 4: Bass

class SubjectMelody(BaseModel):
    notes: List[NoteData]

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "engine": "Like Bach v4.0", 
        "neural_ready": neural_engine is not None
    }

@app.post("/compose")
def compose_v4(melody: SubjectMelody):
    """
    [V4.0] 주제(Soprano)를 입력받아 알토, 테너, 베이스를 생성합니다.
    """
    if not melody.notes:
        raise HTTPException(status_code=400, detail="No notes provided")
    
    if not neural_engine:
        raise HTTPException(status_code=503, detail="Neural Engine not loaded")

    notes_dict = [n.dict() for n in melody.notes]
    
    try:
        print(f">>> Generating 4-voice harmony for {len(notes_dict)} subject notes...")
        responses = neural_engine.generate_response(notes_dict)
        
        # 전체 4성부 데이터를 합쳐서 반환
        full_score = notes_dict + responses
        
        return {
            "status": "success",
            "notes": full_score,
            "count": len(full_score)
        }
    except Exception as e:
        print(f">>> Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
