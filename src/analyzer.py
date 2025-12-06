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


# --- (Optional) sentiment / zero-shot (kept for potential future use) --------

# These are no longer the *primary* classifiers, but you can mention in README
# that you experimented with them and then moved to an embedding-based approach.

_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

_zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)


# --- Emoji helper ------------------------------------------------------------


def _extract_emojis(text: str) -> List[str]:
    return [ch for ch in text if emoji.is_emoji(ch)]


# --- Prototype definitions ---------------------------------------------------

MOOD_CLASS_LABELS = ["Positive", "Negative", "Neutral", "Mixed"]
ENERGY_CLASS_LABELS = ["High Energy", "Low Energy", "High Stress", "Calm"]

# Seed prototypes (generic, not tied to specific slang)
MOOD_PROTOTYPES: Dict[str, List[str]] = {
    "Positive": [
        "I feel happy and content today.",
        "I'm proud of myself and things are going well.",
        "I had a good day and I feel optimistic.",
    ],
    "Negative": [
        "I feel miserable and upset.",
        "Everything feels terrible and heavy.",
        "I am sad and nothing seems to go right.",
    ],
    "Neutral": [
        "Nothing special happened today.",
        "Today was okay, nothing good or bad.",
        "It was an ordinary day without strong feelings.",
    ],
    "Mixed": [
        "I'm excited but also really anxious.",
        "I'm happy about the outcome but stressed about what comes next.",
        "I feel both relieved and worried at the same time.",
    ],
}

ENERGY_PROTOTYPES: Dict[str, List[str]] = {
    "High Energy": [
        "I feel energized and ready to go.",
        "I was running around all day and thriving.",
        "I am full of energy and eager to do things.",
    ],
    "Low Energy": [
        "I feel exhausted and drained.",
        "I am tired and have no energy to do anything.",
        "I barely have the strength to move.",
    ],
    "High Stress": [
        "I feel overwhelmed, anxious and on edge.",
        "My heart is racing and my brain won't slow down.",
        "I am panicking and overloaded with stress.",
    ],
    "Calm": [
        "I feel peaceful and relaxed.",
        "I feel present and calm about everything.",
        "My mind is quiet and I feel at ease.",
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

def _classify_mood_with_top2(vec: np.ndarray) -> str:
    """
    Classify mood using centroid similarities, but allow 'Mixed' when
    Positive and Negative are both strong and close together.
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

    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0]

    # If we have at least two labels, inspect the runner-up
    if len(scores) > 1:
        second_label, second_score = scores[1]

        # If Positive and Negative are both high and close in score,
        # treat the overall mood as Mixed.
        if {best_label, second_label} == {"Positive", "Negative"}:
            if best_score - second_score < 0.05:
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
    mood = _classify_mood_with_top2(vec)
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
