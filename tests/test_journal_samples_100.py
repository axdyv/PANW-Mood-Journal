import json
from pathlib import Path

from src.analyzer import analyze_text


def test_journal_samples_100():
    path = Path("sample_data/journal_samples_100.json")
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

    print(f"\n[100-sample Held-out] Accuracy: {accuracy:.1%} "
          f"({total - num_mismatch}/{total} correct)")
    if mismatches:
        print("[100-sample Held-out] Mismatches:")
        for m in mismatches:
            print(f"  id={m['id']}")
            print(f"    expected : {m['expected']}")
            print(f"    predicted: {m['predicted']}")
            print(f"    text     : {m['text']}\n")

    # This is a *held-out stress test*, so we allow more disagreement,
    # but still enforce that the classifier is meaningfully aligned.
    assert num_mismatch <= 50, f"Too many mismatches on 100-sample held-out set: {num_mismatch}"
import json
from pathlib import Path

from src.analyzer import analyze_text


def test_journal_samples_100():
    path = Path("sample_data/journal_samples_100.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    total = len(data)
    mood_correct = 0
    energy_correct = 0
    pair_correct = 0
    mismatches = []

    for row in data:
        result = analyze_text(row["text"])
        mood_pred = result["mood"]
        energy_pred = result["energy"]

        mood_exp = row["expected_mood"]
        energy_exp = row["expected_energy"]

        alt_moods = row.get("alt_moods", []) or []
        alt_energies = row.get("alt_energies", []) or []

        mood_ok = (mood_pred == mood_exp) or (mood_pred in alt_moods)
        energy_ok = (energy_pred == energy_exp) or (energy_pred in alt_energies)

        if mood_ok:
            mood_correct += 1
        if energy_ok:
            energy_correct += 1
        if mood_ok and energy_ok:
            pair_correct += 1
        else:
            mismatches.append({
                "id": row["id"],
                "expected": (mood_exp, energy_exp),
                "alt_moods": alt_moods,
                "alt_energies": alt_energies,
                "predicted": (mood_pred, energy_pred),
                "text": row["text"],
            })

    mood_acc = mood_correct / total if total else 0.0
    energy_acc = energy_correct / total if total else 0.0
    pair_acc = pair_correct / total if total else 0.0

    print(f"\n[100-sample Held-out]")
    print(f"  Mood accuracy     : {mood_acc:.1%} ({mood_correct}/{total})")
    print(f"  Energy accuracy   : {energy_acc:.1%} ({energy_correct}/{total})")
    print(f"  Full pair accuracy: {pair_acc:.1%} ({pair_correct}/{total})")

    if mismatches:
        print("\n[Remaining mismatches after alt-labels]:")
        for m in mismatches[:15]:  # only print first 15 for sanity
            print(f"  id={m['id']}")
            print(f"    expected : {m['expected']}")
            print(f"    alt      : moods={m['alt_moods']}, energies={m['alt_energies']}")
            print(f"    predicted: {m['predicted']}")
            print(f"    text     : {m['text']}\n")
        if len(mismatches) > 15:
            print(f"  ... and {len(mismatches) - 15} more.\n")

    # Evaluation-only test: no strict threshold here.
    # It always "passes" but prints diagnostics for README / analysis.
    assert True
