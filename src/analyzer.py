from __future__ import annotations

from typing import Dict, List, Tuple

import emoji
from transformers import pipeline


# --- Pipelines ----------------------------------------------------------------

# Sentiment for overall polarity (we'll use score distribution)
_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# Zero-shot classifier for mood / energy / tone (semantic)
_zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

MOOD_LABELS = [
    "positive feelings",
    "negative feelings",
    "mixed feelings",
    "neutral feelings",
]

ENERGY_LABELS = [
    "high energy",
    "low energy",
    "high stress",
    "calm",
]

TONE_LABELS = [
    "sarcastic",
    "sincere",
]


# --- Helpers ------------------------------------------------------------------


def _extract_emojis(text: str) -> List[str]:
    return [ch for ch in text if emoji.is_emoji(ch)]


def _sentiment_distribution(text: str) -> Tuple[str, float, float]:
    """
    Return (baseline_mood, pos_score, neg_score) from sentiment model.
    Uses distribution to detect 'Mixed' when scores are close.
    """
    results = _sentiment(text, return_all_scores=True)[0]
    scores_by_label = {r["label"].upper(): r["score"] for r in results}
    pos = scores_by_label.get("POSITIVE", 0.0)
    neg = scores_by_label.get("NEGATIVE", 0.0)

    # Base label from max score
    if pos > neg:
        base = "Positive"
    elif neg > pos:
        base = "Negative"
    else:
        base = "Neutral"

    # If the model is conflicted and reasonably confident overall, call it Mixed
    if abs(pos - neg) < 0.15 and max(pos, neg) > 0.4:
        base = "Mixed"

    return base, pos, neg


def _mood_from_zero_shot(text: str) -> str:
    """
    Use zero-shot to classify overall mood:
      positive / negative / mixed / neutral
    """
    res = _zero_shot(text, candidate_labels=MOOD_LABELS, multi_label=False)
    label = res["labels"][0].lower()

    if "mixed" in label:
        return "Mixed"
    if "positive" in label:
        return "Positive"
    if "negative" in label:
        return "Negative"
    if "neutral" in label:
        return "Neutral"
    return "Unknown"


def _tone_from_zero_shot(text: str) -> Tuple[str, float]:
    """
    Use zero-shot to classify tone: sarcastic vs sincere.
    """
    res = _zero_shot(text, candidate_labels=TONE_LABELS, multi_label=False)
    return res["labels"][0].lower(), float(res["scores"][0])


def _normalize_energy_label(label: str) -> str:
    label = label.lower().strip()
    if "high energy" in label:
        return "High Energy"
    if "low energy" in label:
        return "Low Energy"
    if "high stress" in label or "stressed" in label:
        return "High Stress"
    if "calm" in label or "relaxed" in label:
        return "Calm"
    return "Unknown"


def _energy_from_zero_shot_scores(text: str, mood: str) -> str:
    """
    Use multi-label zero-shot classification to infer energy/stress,
    then interpret the label distribution.
    """
    res = _zero_shot(
        text,
        candidate_labels=ENERGY_LABELS,
        multi_label=True,
    )
    labels = res["labels"]
    scores = res["scores"]

    score_map = dict(zip(labels, scores))
    # Take top 2 labels to understand the "shape" of the distribution
    top2 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:2]
    top_labels = [lbl for lbl, _ in top2]

    # If high stress is in the top-2, we treat it as High Stress
    if "high stress" in top_labels:
        return "High Stress"
    # High energy wins if present
    if "high energy" in top_labels:
        # If mood is clearly negative/mixed, high arousal may feel stressful
        if mood in {"Negative", "Mixed"}:
            return "High Stress"
        return "High Energy"
    # Calm vs low energy
    if "calm" in top_labels:
        return "Calm"
    if "low energy" in top_labels:
        return "Low Energy"

    # Fallback: map the single top label if we must
    return _normalize_energy_label(labels[0])


# --- Core API -----------------------------------------------------------------


def analyze_text(text: str) -> Dict[str, str]:
    """
    Analyze a journal entry and return tags:
      - mood: Positive | Negative | Neutral | Mixed | Unknown
      - energy: High Energy | Low Energy | High Stress | Calm | Unknown

    Design:
      - Use sentiment distribution to get a baseline mood (and detect 'Mixed')
      - Use zero-shot to refine mood (positive/negative/mixed/neutral)
      - Use zero-shot to infer sarcastic vs sincere tone
      - Use zero-shot multi-label to infer energy/stress from energy labels
      - Apply only very small, generic adjustments (emoji-only, very short replies)
    """
    if text is None:
        text = ""
    cleaned = text.strip()

    # 1) Handle empty / whitespace-only safely
    if not cleaned:
        return {"mood": "Unknown", "energy": "Unknown"}

    # 2) Baseline mood from sentiment distribution
    mood_sent, pos_score, neg_score = _sentiment_distribution(cleaned)

    # 3) Semantic mood from zero-shot (positive / negative / mixed / neutral)
    mood_zs = _mood_from_zero_shot(cleaned)

    # Combine mood signals:
    mood = mood_sent

    # If zero-shot says Mixed, trust that
    if mood_zs == "Mixed":
        mood = "Mixed"
    # If sentiment was Neutral but zero-shot is clear, adopt zero-shot
    elif mood_sent == "Neutral" and mood_zs in {"Positive", "Negative"}:
        mood = mood_zs
    # If sentiment is conflicted (close pos/neg) and zero-shot is clear, use zero-shot
    elif abs(pos_score - neg_score) < 0.15 and mood_zs in {"Positive", "Negative"}:
        mood = mood_zs

    # 4) Tone: sarcastic vs sincere
    tone_label, tone_score = _tone_from_zero_shot(cleaned)

    # If the text is confidently sarcastic and mood looks positive,
    # it's probably actually mixed or negative in intent.
    if tone_label == "sarcastic" and tone_score > 0.7:
        if mood == "Positive":
            mood = "Mixed"

    # 5) Energy / stress from zero-shot distribution
    energy = _energy_from_zero_shot_scores(cleaned, mood)

    # 6) Generic emoji / short-text adjustments (no phrase-specific rules)

    ems = _extract_emojis(cleaned)
    has_only_emoji = bool(ems) and len(cleaned.replace(" ", "")) == len(ems)

    # If it's literally only emojis, let emojis tilt mood a bit
    if has_only_emoji:
        negish = {"😭", "😢", "😔", "😩", "😫", "😡", "💀", "🥲"}
        posish = {"😄", "😁", "😆", "😎", "😊", "😂", "🤩", "❤️", "✨", "👍"}

        if any(e in negish for e in ems) and mood in {"Positive", "Neutral"}:
            mood = "Negative"
        elif any(e in posish for e in ems) and mood in {"Negative", "Neutral"}:
            mood = "Positive"

        # If negative emojis dominate and energy looks too calm/low, bump to High Stress
        if any(e in negish for e in ems) and energy in {"Calm", "Low Energy", "Unknown"}:
            energy = "High Stress"

    # very short flat responses without emojis: often low energy
    if len(cleaned.split()) <= 2 and not ems and mood in {"Neutral", "Unknown"}:
        energy = "Low Energy"

    return {"mood": mood, "energy": energy}
