# **PANW Mood Journal Project (Draft)**

*A semantic NLP engine for emotional inference in unstructured text*

---

## 📌 Overview

I used ChatGPT and Google Gemini 3 to help me generate ambiguous entries, generate base code solutions, create comments/improve readibility of the code, and verify the feasibility of my ideas. 
The PANW Mood Journal Project is an NLP-driven system that analyzes free-form journal text and assigns two-dimensional emotional labels:

Unlike traditional keyword-based sentiment engines, this system is built to handle **unstructured, messy text**: slang, sarcasm, typos, emojis, long-form reflections, and even fragmented emotional narratives.
This has meaningful applications in **communication intelligence**, cyber-investigation, and emotional signal detection within large text streams.

> Instead of asking *“is this positive or negative?”*, we ask:
> **What emotional state does this reflect, and how activated is it?**

---

## 🧠 Technical Motivation

Common sentiment models struggle with:

Sarcasm/Tone-Inversion: "Love that I'm working late" → incorrectly positive
Mixed emotional phrases: "Good day, but my chest feels tight"
Slang/Emojis/Typos: "supa mid day ngl, brain frieddd"
Long-form journaling: Sentences that are closer to a paragraph in length.

To address this, we moved to a **semantic vector embedding + centroid inference approach**.

---

## 🔍 NLP Architecture

### 1. Sentence → Embedding

```
Journal Entry → Encoder → 768-dimensional semantic vector
```

### 2. Emotion Classification via Cosine Similarity

Instead of feeding text directly into a sentiment classifier:

1. Define **prototype sentences** representing each emotion class.
2. Convert prototypes → embeddings.
3. Compute **centroids** per class:

> centroid(label) = mean(embedding(samples[label]))

4. To classify new text:

```python
score[label] = cosine_similarity(entry_vec, centroid[label])
mood = best_scoring_label
```

### Why this design worked

The embeddings help to capture semantics instead of just relying on keywords, handling slang/stylistic variation. Centroids also help generalize instead of memorize, avoiding brittle hardcoded rule lists.
By utilizing cosine similarity, the program is also able to support nuance by introducing Mixed and Confused centroids. The design of this implementation also allows for expandable label space with the freedom
to include more emotional representation

---

## 📈 Model Evolution & Accuracy History

### Phase 1 — Baseline (Zero-Shot + Sentiment)

* Good for simple sentences
* Failed edge cases and sarcasm
* **19/25 mismatches (24% success)**

### Phase 2 — Embedding Centroids

* Introduced vector similarity classification
* Better performance on ambiguity and tone
* **23/25 correct (92%)**

### Phase 3 — Stress Test: 100-Entry Dataset

The dataset included:
1. Long-form text and journaling paragraphs
2. Slang + shorthand (“idk, kinda mid ngl”)
3. Poems + abstract metaphorical entries
4. Emoji-only and near-gibberish cases
5. Empty strings, malformed grammar

Initial score: **66%**
Interpretation: Many “incorrect” outputs were *actually valid interpretations*. However, upon further inspection, I came to an important conclusion: emotional classification is inherently and largely subjective.

**Evaluation updated → Alternate valid labels allowed.**
Accuracy improved, with reduction in false negatives.

### Phase 4 — Expanded Mood Taxonomy

Introduced **Confused** mood label.

> Meaningfully separated entries like
> “I don’t even know how I feel today.”

Result: Improved categorization and reduced ambiguity collisions.

---

## 🛠 Features

1. Mood and Activity classification
2. Slang/Emoji handling
3. Incorrect grammar/messy text handling
4. FastAPI backend for inference and storage
5. Frontend journal UI with logging
6. Emoji visualization + colored tags

Future Implementation
7. User focused centroids: helps give more accurate classifications
8. Graph/improved tracking over periods of time to visualize mental health across time
9. Multi-label mood distributions: improves overall emotion scoring
10. Explainable reasoning output: helps user understand why certain entries are classified as what they are.
11. Real-time emotion classification: improves user experience and immersion

---

## 🚀 Running Locally

### Backend (FastAPI)

```bash
uvicorn src.main:app --reload
```

### Frontend (Vite)

```bash
npm install
npm run dev
```

Navigate to → `http://localhost:5173`

---

