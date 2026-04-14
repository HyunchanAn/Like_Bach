from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from src.engine import FugueEngine

app = FastAPI(title="BPGE Backend")
engine = FugueEngine()

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

class SubjectMelody(BaseModel):
    notes: List[NoteData]

@app.get("/")
def read_root():
    return {"status": "online", "engine": "BPGE"}

@app.post("/analyze_subject")
def analyze_subject(melody: SubjectMelody):
    try:
        data = [n.dict() for n in melody.notes]
        analysis = engine.analyze_subject(data)
        answer = engine.generate_tonal_answer(analysis['stream'], analysis['key'])
        
        answer_notes = []
        for n in answer.recurse().notes:
            answer_notes.append({"pitch": n.pitch.midi, "duration": n.duration.quarterLength, "offset": n.offset})
            
        return {"key": str(analysis['key']), "answer_notes": answer_notes, "is_tonal": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compose")
def compose_piece(melody: SubjectMelody):
    """
    주제를 입력받아 전체 2성부 곡을 작곡합니다.
    """
    if not melody.notes:
        raise HTTPException(status_code=400, detail="No notes provided")
    try:
        result = engine.compose_full_piece([n.dict() for n in melody.notes])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
