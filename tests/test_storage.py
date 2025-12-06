from src.storage import create_entry, get_last_entries

def test_storage_smoke():
    entry = create_entry("test entry", {"mood": "Neutral", "energy": "Unknown"})
    last = get_last_entries(1)[0]
    assert last.text == "test entry"
