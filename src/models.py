from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any


@dataclass
class JournalEntry:
    id: int
    timestamp: datetime
    text: str
    tags: Dict[str, str]  # e.g. {"mood": "Positive", "energy": "High Energy"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        data = asdict(self)
        # Store timestamp as ISO string
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "JournalEntry":
        """Reconstruct from a dict (loaded from JSON)."""
        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw)
        else:
            # Fallback: now, if timestamp missing or invalid
            timestamp = datetime.now()

        return JournalEntry(
            id=int(data.get("id", 0)),
            timestamp=timestamp,
            text=data.get("text", ""),
            tags=data.get("tags", {}) or {},
        )
