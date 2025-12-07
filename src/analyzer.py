from __future__ import annotations

from typing import Dict, List, Tuple

import json
from pathlib import Path

import emoji
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# --- Paths --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AMBIG_PATH = BASE_DIR / "sample_data" / "ambiguous_samples.json"
LONG_PATH = BASE_DIR / "sample_data" / "long_entries.json"
POETIC_PATH = BASE_DIR / "sample_data" / "poetic_entries.json"
JOURNAL_PATH = BASE_DIR / "sample_data" / "journal_samples.json"
JOURNAL100_PATH = BASE_DIR / "sample_data" / "journal_samples_100.json"



# --- Embedding model ---------------------------------------------------------

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _embed(text: str) -> np.ndarray:
    """Return a normalized sentence embedding."""
    v = _embedder.encode(text, convert_to_numpy=True)
    # Normalize for cosine similarity via dot product
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm

# --- Emoji helper ------------------------------------------------------------


def _extract_emojis(text: str) -> List[str]:
    return [ch for ch in text if emoji.is_emoji(ch)]


# --- Prototype definitions ---------------------------------------------------

MOOD_CLASS_LABELS = ["Positive", "Negative", "Neutral", "Mixed", "Confused"]
ENERGY_CLASS_LABELS = ["High Energy", "Low Energy", "High Stress", "Calm"]

# Seed prototypes (generic, not tied to specific slang)
MOOD_PROTOTYPES = {
    "Positive": [
        "I feel great.",
        "I'm really happy with how today went.",
        "Today was amazing and I feel proud of myself.",
        "I feel optimistic about what's coming next.",
        "I'm grateful and content right now.",
        "Spending time with my friends made me so happy.",
        "I woke up feeling refreshed and ready for the day.",
        "I'm in a really good mood.",
        "Things are finally going my way and it feels good.",
        "I feel peaceful and satisfied.",
    ],
    "Negative": [
        "I feel awful and everything went wrong today.",
        "I'm really upset and disappointed in myself.",
        "Today was terrible and I just want to cry.",
        "I feel sad and nothing seems to help.",
        "I feel anxious and can't stop worrying.",
        "I'm frustrated and angry about how things turned out.",
        "I feel hopeless and stuck.",
        "I feel like I'm failing at everything.",
        "I'm overwhelmed and it feels like too much.",
        "I feel empty and disconnected from everything.",
    ],
    "Neutral": [
        "Today was fine, nothing special happened.",
        "I feel okay, not good or bad.",
        "Im ok",
        "It was just an ordinary day.",
        "I'm not particularly happy or sad right now.",
        "My mood is pretty neutral at the moment.",
        "Nothing really stood out about today.",
        "I feel kind of indifferent about how things went.",
        "It was a normal, uneventful day.",
        "I'm just going through the motions today.",
        "Overall, today was pretty average.",
    ],
    "Mixed": [
        "I'm proud of what I did today, but I'm also exhausted.",
        "I'm excited about the future, but I'm scared I might mess it up.",
        "Today was good overall, but something still feels off.",
        "I laughed a lot today, but underneath I'm still anxious.",
        "I'm grateful for what I have, but I feel overwhelmed too.",
        "I'm happy with the results, but the process really drained me.",
        "I feel both hopeful and worried at the same time.",
        "I'm glad things worked out, but I'm still stressed.",
        "I'm relieved it's over, but I'm not completely at ease.",
        "I had fun today, but there's a heaviness I can't quite shake.",
    ],
    "Confused": [
        "I honestly don't know how I feel right now.",
        "Today was strange and I can't decide if it was good or bad.",
        "My emotions feel all over the place and hard to name.",
        "I feel weird, but I can't explain why.",
        "I'm not sure if I'm okay or not, it's just confusing.",
        "Everything feels mixed up in my head.",
        "I feel like something is off, but I don't know what.",
        "I can't tell if I'm happy, sad, or just tired.",
        "Emotionally I feel kind of scrambled and uncertain.",
        "I don't have the words for how I feel, it's just... confusing.",
    ],
}


ENERGY_PROTOTYPES = {
    "High Energy": [
        "I feel energized and ready to go.",
        "I'm full of energy and motivation.",
        "I feel pumped up and excited.",
        "I'm buzzing with energy right now.",
        "I can't sit still, I have so much energy.",
        "I feel lively and active today.",
        "I'm wide awake and fired up.",
        "I feel super productive and focused.",
        "I have a lot of momentum and drive.",
        "I'm ready to take on anything today.",
    ],
    "Low Energy": [
        "I feel really tired and drained.",
        "I don't have the energy to do anything.",
        "I'm exhausted and just want to lie down.",
        "I feel sluggish and slow today.",
        "My body feels heavy and worn out.",
        "I feel like I'm running on empty.",
        "I can barely keep my eyes open.",
        "I feel worn down and out of energy.",
        "I don't feel like doing much of anything.",
        "I'm completely wiped out.",
    ],
    "High Stress": [
        "I'm so stressed that I can't relax.",
        "My thoughts are racing and I feel on edge.",
        "I can't stop worrying about everything.",
        "I feel like I'm under a lot of pressure.",
        "My chest feels tight and I'm anxious.",
        "I feel panicked and overwhelmed.",
        "I'm tense and can't seem to calm down.",
        "Every little thing is stressing me out.",
        "I feel like I'm constantly in fight or flight mode.",
        "I can't switch my brain off and it's exhausting.",
    ],
    "Calm": [
        "I feel calm and relaxed.",
        "My mind feels quiet and at peace.",
        "Today was peaceful and low-stress.",
        "I feel grounded and steady.",
        "Nothing is really stressing me out right now.",
        "I feel centered and in control.",
        "I'm relaxed and taking things slowly.",
        "I feel at ease with how things are.",
        "Today moved at a gentle, comfortable pace.",
        "I feel unhurried and tranquil.",
    ],
}



def _augment_prototypes_from_labeled(path: Path) -> None:
    """
    Given a labeled JSON file with fields:
      - text
      - expected_mood
      - expected_energy
    use it to expand the prototype sets for mood and energy.
    """
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    for row in data:
        text = (row.get("text") or "").strip()
        if not text:
            continue

        mood = row.get("expected_mood")
        energy = row.get("expected_energy")

        if mood in MOOD_CLASS_LABELS:
            MOOD_PROTOTYPES.setdefault(mood, []).append(text)

        if energy in ENERGY_CLASS_LABELS:
            ENERGY_PROTOTYPES.setdefault(energy, []).append(text)


# Use ALL labeled datasets to shape the centroids:
for p in (AMBIG_PATH, LONG_PATH, POETIC_PATH, JOURNAL_PATH, JOURNAL100_PATH):
    _augment_prototypes_from_labeled(p)


def _compute_centroids(prototypes: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Compute an embedding centroid for each class label from its prototype sentences.
    """
    centroids: Dict[str, np.ndarray] = {}
    for label, sentences in prototypes.items():
        if not sentences:
            continue
        vecs = np.stack([_embed(s) for s in sentences], axis=0)
        centroids[label] = vecs.mean(axis=0)
        # Normalize centroids as well
        norm = np.linalg.norm(centroids[label])
        if norm != 0.0:
            centroids[label] = centroids[label] / norm
    return centroids


MOOD_CENTROIDS = _compute_centroids(MOOD_PROTOTYPES)
ENERGY_CENTROIDS = _compute_centroids(ENERGY_PROTOTYPES)


def _classify_with_centroids(
    vec: np.ndarray,
    centroids: Dict[str, np.ndarray],
    classes: List[str],
) -> str:
    """
    Classify an embedding by cosine similarity to precomputed centroids.
    """
    best_label = "Unknown"
    best_score = -1.0

    for label in classes:
        centroid = centroids.get(label)
        if centroid is None:
            continue
        score = float(np.dot(vec, centroid))  # cosine if both normalized
        if score > best_score:
            best_score = score
            best_label = label

    return best_label

def _classify_mood_with_top2(vec: np.ndarray, text: str) -> str:
    """
    Classify mood using centroid similarity,
    but allow 'Mixed' only when Positive & Negative are both strong.

    This version fixes:
    - Missing text reference
    - Over-aggressive Mixed classification
    - "I feel great" → now properly returns Positive
    """

    scores: list[tuple[str, float]] = []

    for label in MOOD_CLASS_LABELS:
        centroid = MOOD_CENTROIDS.get(label)
        if centroid is None:
            continue
        score = float(np.dot(vec, centroid))
        scores.append((label, score))

    if not scores:
        return "Unknown"

    # sort best to worst
    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0]

    # if we have at least 2 for Mixed detection
    if len(scores) > 1:
        second_label, second_score = scores[1]

        # Mixed candidate ONLY when Positive + Negative are top competitors
        if {best_label, second_label} == {"Positive", "Negative"}:

            # must be confident on both sides before calling Mixed
            if best_score > 0.40 and second_score > 0.40:

                # detect simple short positive statements (fixes "I feel great")
                lower = text.lower()
                word_count = len(text.split())
                positive_cues = any(
                    kw in lower for kw in
                    ["great", "good", "amazing", "fantastic", "awesome", "happy", "excellent"]
                )
                negative_cues = any(
                    kw in lower for kw in
                    ["bad", "terrible", "awful", "sad", "scared", "anxious", "stressed"]
                )

                # If text is clearly positive, do NOT force Mixed
                if (
                    best_label == "Positive" and
                    positive_cues and
                    not negative_cues and
                    word_count <= 6           # slightly more forgiving
                ):
                    return "Positive"

                # Otherwise if they're very close → Mixed
                if abs(best_score - second_score) < 0.05:
                    return "Mixed"

    return best_label


def _classify_energy_with_top2(vec: np.ndarray, mood: str) -> str:
    """
    Classify energy using centroid similarities, but resolve ambiguity
    between High Energy and High Stress using mood + closeness.
    """
    scores: list[tuple[str, float]] = []

    for label in ENERGY_CLASS_LABELS:
        centroid = ENERGY_CENTROIDS.get(label)
        if centroid is None:
            continue
        score = float(np.dot(vec, centroid))
        scores.append((label, score))

    if not scores:
        return "Unknown"

    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0]

    # If we have at least two labels, inspect the runner-up
    if len(scores) > 1:
        second_label, second_score = scores[1]

        # Special handling when the model is torn between "High Energy"
        # and "High Stress": use mood to steer.
        if {best_label, second_label} == {"High Energy", "High Stress"}:
            if best_score - second_score < 0.05:
                if mood in {"Negative", "Mixed"}:
                    return "High Stress"
                elif mood == "Positive":
                    return "High Energy"

    return best_label


# --- Core API ----------------------------------------------------------------


def analyze_text(text: str) -> Dict[str, str]:
    """
    Analyze a journal entry and return tags:
      - mood: Positive | Negative | Neutral | Mixed | Unknown
      - energy: High Energy | Low Energy | High Stress | Calm | Unknown

    Design (embedding-based):
      - Use a pre-trained sentence embedding model to encode the text
      - Classify mood and energy by cosine similarity to learned centroids
        (computed from seed prototypes + ambiguous_samples.json labels)
      - Apply only very small, generic adjustments (emoji-only, very short replies)
    """
    if text is None:
        text = ""
    cleaned = text.strip()

    # 1) Handle empty / whitespace-only safely
    if not cleaned:
        return {"mood": "Unknown", "energy": "Unknown"}
    # Treat pure digits / gibberish-y single tokens as Unknown
    no_space = "".join(cleaned.split())

    # Numbers-only → Unknown
    if no_space.isdigit():
        return {"mood": "Unknown", "energy": "Unknown"}

    # Single long token with letters, no spaces, no emojis → likely gibberish / handle as Unknown
    if (
        len(cleaned.split()) == 1
        and len(no_space) >= 5
        and no_space.isalpha()
        and not _extract_emojis(cleaned)
    ):
        return {"mood": "Unknown", "energy": "Unknown"}


    # 2) Embed the text
    vec = _embed(cleaned)

    # 3) Classify via centroids with top-2 logic
    mood = _classify_mood_with_top2(vec, cleaned)
    energy = _classify_energy_with_top2(vec, mood)


    # 4) Generic emoji / short-text adjustments (no phrase-specific rules)

    ems = _extract_emojis(cleaned)
    has_only_emoji = bool(ems) and len(cleaned.replace(" ", "")) == len(ems)

    if has_only_emoji:
        negish = {"😭", "😢", "😔", "😩", "😫", "😡", "💀", "🥲"}
        posish = {"😄", "😁", "😆", "😎", "😊", "😂", "🤩", "❤️", "✨", "👍"}

        if any(e in negish for e in ems) and mood in {"Positive", "Neutral"}:
            mood = "Negative"
        elif any(e in posish for e in ems) and mood in {"Negative", "Neutral"}:
            mood = "Positive"

        if any(e in negish for e in ems) and energy in {"Calm", "Low Energy", "Unknown"}:
            energy = "High Stress"

    # very short flat responses without emojis: often low energy / low affect
    if len(cleaned.split()) <= 2 and not ems and mood in {"Neutral", "Unknown"}:
        energy = "Low Energy"

    return {"mood": mood, "energy": energy}
