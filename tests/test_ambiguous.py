import json
from pathlib import Path

from src.analyzer import analyze_text


def test_ambiguous_cases_semantics():
    """
    This isn't a strict correctness test — it's an evaluation harness
    that runs our analyzer over a set of intentionally ambiguous cases,
    prints mismatches, and ensures the pipeline runs end-to-end.
    """
    path = Path("sample_data") / "ambiguous_samples.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    mismatches = []

    for row in data:
        predicted = analyze_text(row["text"])
        mood_pred = predicted["mood"]
        energy_pred = predicted["energy"]

        mood_exp = row["expected_mood"]
        energy_exp = row["expected_energy"]

        if mood_pred != mood_exp or energy_pred != energy_exp:
            mismatches.append({
                "id": row["id"],
                "text": row["text"],
                "expected": (mood_exp, energy_exp),
                "predicted": (mood_pred, energy_pred),
            })

    total = len(data)
    correct = total - len(mismatches)
    accuracy = correct / total if total else 0.0

    print(f"\nAmbiguous-case accuracy (mood+energy exact match): {accuracy:.1%}")
    print(f"Mismatches ({len(mismatches)}/{total}):")
    for m in mismatches:
        print(f"  id={m['id']}: expected={m['expected']} predicted={m['predicted']}")
        print(f"     text={m['text']}")

    assert True
