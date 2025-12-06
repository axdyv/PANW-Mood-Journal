import json
from pathlib import Path

from src.analyzer import analyze_text


def test_full_journal_samples():
    path = Path("sample_data/journal_samples.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    mismatches = []

    for row in data:
        result = analyze_text(row["text"])
        mood = result["mood"]
        energy = result["energy"]

        if mood != row["expected_mood"] or energy != row["expected_energy"]:
            mismatches.append({
                "id": row["id"],
                "expected": (row["expected_mood"], row["expected_energy"]),
                "predicted": (mood, energy),
                "text": row["text"],
            })

    total = len(data)
    num_mismatch = len(mismatches)
    accuracy = (total - num_mismatch) / total if total else 0.0

    print(f"\n[All Samples] Accuracy: {accuracy:.1%} "
          f"({total - num_mismatch}/{total} correct)")
    if mismatches:
        print("[All Samples] Mismatches:")
        for m in mismatches:
            print(f"  id={m['id']}")
            print(f"    expected : {m['expected']}")
            print(f"    predicted: {m['predicted']}")
            print(f"    text     : {m['text']}\n")

    # Allow a small number of disagreements (journal emotion is subjective),
    # but still enforce a quality bar.
    assert num_mismatch <= 4, f"Too many mismatches on combined samples: {num_mismatch}"
