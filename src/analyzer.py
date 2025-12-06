from __future__ import annotations

import re
from typing import Dict, List

import emoji
from transformers import pipeline

# Load sentiment model once at import time
# (small, widely used model)
_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


# --- Helper sets / patterns -------------------------------------------------

HIGH_ENERGY_WORDS = {
    "pumped",
    "hyped",
    "excited",
    "crushing it",
    "killing it",
    "on fire",
    "so happy",
    "so proud",
    "hyped up",
}

LOW_ENERGY_WORDS = {
    "tired",
    "exhausted",
    "drained",
    "numb",
    "dead tired",
    "burned out",
    "burnt out",
    "no energy",
    "low energy",
}

STRESS_WORDS = {
    "overwhelmed",
    "anxious",
    "anxiety",
    "panicking",
    "panic",
    "stressed",
    "stressful",
    "heart racing",
    "tight chest",
    "crushing me",
}

POSITIVE_EMOJI = {"😄", "😁", "😆", "😎", "😊", "😂", "🤩", "❤️", "✨", "👍"}
NEGATIVE_EMOJI = {"😭", "😢", "😔", "😩", "😫", "😡", "💀", "🥲", "😖", "😞"}


def _extract_emojis(text: str) -> List[str]:
    return [ch for ch in text if emoji.is_emoji(ch)]


def _contains_any(text: str, phrases: List[str] | set[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in phrases)


# --- Core API ----------------------------------------------------------------


def analyze_text(text: str) -> Dict[str, str]:
    """
    Analyze a journal entry and return tags:
      - mood: Positive | Negative | Neutral | Mixed | Unknown
      - energy: High Energy | Low Energy | High Stress | Calm | Unknown
    """
    if text is None:
        text = ""
    cleaned = text.strip()

    # Handle empty / whitespace-only safely
    if not cleaned:
        return {"mood": "Unknown", "energy": "Unknown"}

    lower = cleaned.lower()
    mood = "Unknown"
    energy = "Unknown"

    # 1) Phrase-level overrides for obvious contextual cases
    if "crushing it" in lower or "killing it" in lower:
        mood = "Positive"
        energy = "High Energy"
    elif "crushing me" in lower or "crushing my soul" in lower:
        mood = "Negative"
        energy = "High Stress"
    elif "dead inside" in lower:
        mood = "Negative"
        energy = "Low Energy"

    # 2) Base sentiment from transformer model (if not already decided)
    if mood == "Unknown":
        result = _sentiment(cleaned)[0]
        label = result.get("label", "").upper()

        if "POSITIVE" in label:
            mood = "Positive"
        elif "NEGATIVE" in label:
            mood = "Negative"
        else:
            mood = "Neutral"

    # 3) Emoji-based adjustments / detection
    ems = _extract_emojis(cleaned)
    has_pos_emoji = any(e in POSITIVE_EMOJI for e in ems)
    has_neg_emoji = any(e in NEGATIVE_EMOJI for e in ems)

    if has_pos_emoji and has_neg_emoji:
        mood = "Mixed"
    elif has_pos_emoji and mood == "Negative":
        # Conflicting signals => Mixed
        mood = "Mixed"
    elif has_neg_emoji and mood == "Positive":
        mood = "Mixed"
    elif has_neg_emoji and mood == "Neutral":
        mood = "Negative"

    # 4) Energy / stress heuristics
    #    If not forced by special phrase earlier
    if energy == "Unknown":
        exclamations = cleaned.count("!")
        has_high_energy_word = _contains_any(cleaned, HIGH_ENERGY_WORDS)
        has_low_energy_word = _contains_any(cleaned, LOW_ENERGY_WORDS)
        has_stress_word = _contains_any(cleaned, STRESS_WORDS)

        if has_stress_word:
            energy = "High Stress"
        elif has_high_energy_word or exclamations >= 2 or has_pos_emoji:
            energy = "High Energy" if mood in {"Positive", "Mixed"} else "High Stress"
        elif has_low_energy_word or (not ems and len(cleaned.split()) <= 4):
            # short, flat entries often feel low energy ("ok", "tired", etc.)
            energy = "Low Energy"
        else:
            # fallback defaults based on mood
            if mood == "Positive":
                energy = "Calm"
            elif mood == "Negative":
                energy = "High Stress"
            elif mood == "Mixed":
                energy = "High Stress"
            else:
                energy = "Low Energy"

    return {"mood": mood, "energy": energy}
