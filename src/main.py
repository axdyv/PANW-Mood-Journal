from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from .analyzer import analyze_text
from .storage import create_entry, get_last_entries

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EntryRequest(BaseModel):
    text: str

class EntryResponse(BaseModel):
    id: int
    timestamp: str
    text: str
    mood: str
    energy: str

@app.post("/entries", response_model=EntryResponse)
def add_entry(request: EntryRequest):
    tags = analyze_text(request.text)
    entry = create_entry(request.text, tags)
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat(),
        "text": entry.text,
        "mood": entry.tags.get("mood", "Unknown"),
        "energy": entry.tags.get("energy", "Unknown")
    }

@app.get("/entries", response_model=List[EntryResponse])
def get_entries(limit: int = 50):
    entries = get_last_entries(limit)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "text": e.text,
            "mood": e.tags.get("mood", "Unknown"),
            "energy": e.tags.get("energy", "Unknown")
        }
        for e in entries
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
