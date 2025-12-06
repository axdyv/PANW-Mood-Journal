from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .analyzer import analyze_text
from .storage import create_entry, get_last_entries

app = FastAPI(title="AI Mood Journal API")


class EntryCreateRequest(BaseModel):
    text: str


class EntryResponse(BaseModel):
    id: int
    timestamp: datetime
    text: str
    mood: str
    energy: str


@app.post("/entries", response_model=EntryResponse)
def create_journal_entry(payload: EntryCreateRequest) -> EntryResponse:
    tags = analyze_text(payload.text)
    entry = create_entry(payload.text, tags)

    return EntryResponse(
        id=entry.id,
        timestamp=entry.timestamp,
        text=entry.text,
        mood=entry.tags.get("mood", "Unknown"),
        energy=entry.tags.get("energy", "Unknown"),
    )


@app.get("/entries", response_model=List[EntryResponse])
def list_entries(limit: int = 20) -> List[EntryResponse]:
    entries = get_last_entries(limit)
    return [
        EntryResponse(
            id=e.id,
            timestamp=e.timestamp,
            text=e.text,
            mood=e.tags.get("mood", "Unknown"),
            energy=e.tags.get("energy", "Unknown"),
        )
        for e in entries
    ]
