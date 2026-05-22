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
from src.engine import FugueEngine

app = FastAPI(title="BPGE Backend - Neural Hybrid v3.0")
engine = FugueEngine()
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
    """단일 음표의 물리적 화성학 명세서입니다.
    
    Attributes:
        pitch: 음높이 (MIDI 번호 규격).
        duration: 지속 시간 (4분음표 1.0 기준).
        offset: 마디 시작점으로부터의 오프셋 간격.
    """
    pitch: int
    duration: float
    offset: float

class SubjectMelody(BaseModel):
    """사용자가 입력한 초기 소프라노 선율 시퀀스 집합입니다.
    
    Attributes:
        notes: 음표 인스턴스 데이터 목록.
    """
    notes: List[NoteData]

@app.get("/")
def read_root() -> Dict[str, str]:
    """백엔드 작곡 서버의 상태 체크 및 현재 구동 중인 하드웨어 엔진 모드를 반환합니다.
    
    Returns:
        Dict[str, str]: status(온라인 여부), engine(버전 정보), mode(신경망 활성 모드).
    """
    mode = "Neural + Rule-based" if neural_engine else "Rule-based only"
    return {"status": "online", "engine": "BPGE v3.0", "mode": mode}

@app.post("/compose")
def compose_piece(melody: SubjectMelody) -> Dict[str, list]:
    """주제를 입력받아 전체 곡을 작곡합니다.
    
    우선 Neural Engine 시도 후, 실패 시 Rule-based Engine으로 폴백합니다.
    
    Args:
        melody: 사용자가 입력한 소프라노 선율 데이터 세트.
        
    Returns:
        Dict[str, list]: 각 성부별 음표 리스트 및 조성 분석 정보 메타데이터.
        
    Raises:
        HTTPException: 입력값 누락 시 400 에러, 생성 실패 시 500 내부 에러 발생.
    """
    if not melody.notes:
        raise HTTPException(status_code=400, detail="No notes provided")
    
    notes_dict = [n.dict() for n in melody.notes]
    
    # 1. Neural Generation 시도 (v3.0 신기능)
    if neural_engine:
        try:
            print(">>> Attempting neural generation for high-quality harmony...")
            neural_resp = neural_engine.generate_response(notes_dict)
            if neural_resp:
                # 신경망 생성 결과를 규칙 기반 엔진의 전체 곡 구성 로직과 결합
                result = engine.assemble_hybrid_score(notes_dict, neural_resp)
                result["mode"] = "Neural Hybrid v3.0"
                return result 
        except Exception as e:
            print(f">>> Neural generation failed, falling back to rule-based: {e}")
 
    # 2. Rule-based Fallback (v2.6 기반)
    try:
        result = engine.compose_full_piece(notes_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
