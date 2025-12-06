import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from models import JournalEntry

# Path: repo_root/data/journal_entries.json
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ENTRIES_FILE = DATA_DIR / "journal_entries.json"


def _load_raw() -> List[Dict]:
    """Load raw JSON list from disk. If file missing/empty, return []."""
    if not ENTRIES_FILE.exists():
        return []

    try:
        text = ENTRIES_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)
    except json.JSONDecodeError:
        # If file is corrupted, you might log this; for now, start fresh
        return []


def _save_raw(raw_entries: List[Dict]) -> None:
    """Save raw list of dicts to disk as pretty JSON."""
    ENTRIES_FILE.write_text(
        json.dumps(raw_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_entries() -> List[JournalEntry]:
    """Return all entries as JournalEntry objects."""
    raw = _load_raw()
    return [JournalEntry.from_dict(item) for item in raw]


def _next_id(entries: List[JournalEntry]) -> int:
    """Compute the next id as max(existing) + 1."""
    if not entries:
        return 1
    return max(e.id for e in entries) + 1


def create_entry(text: str, tags: Dict[str, str]) -> JournalEntry:
    """
    Create a new JournalEntry with id + timestamp, persist it,
    and return the created entry.
    """
    entries = load_entries()
    new_id = _next_id(entries)
    entry = JournalEntry(
        id=new_id,
        timestamp=datetime.now(),
        text=text,
        tags=tags,
    )

    # append and save
    raw = [e.to_dict() for e in entries]
    raw.append(entry.to_dict())
    _save_raw(raw)

    return entry


def get_last_entries(n: int = 3) -> List[JournalEntry]:
    """
    Return the last n entries ordered by timestamp descending.
    """
    entries = load_entries()
    # sort by timestamp (latest first)
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return entries[:n]
