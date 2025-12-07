# **PANW Mood Journal Project**

*A semantic NLP engine for emotional inference in unstructured text*

---

## 📌 Overview

I used ChatGPT and Google Gemini 3 to help me generate ambiguous entries, generate base code solutions, create comments/improve readibility of the code, and verify the feasibility of my ideas. 
The PANW Mood Journal Project is an NLP-driven system that analyzes free-form journal text and assigns two-dimensional emotional labels:

Classifying how a person feels just based off of what they're saying is difficult, even for people. The goal of this mood journal is to provide some deeper insight into **how** a person feels when they write certain journal entries. Unlike traditional keyword-based sentiment engines, this system is built to handle **unstructured, messy text**: slang, sarcasm, typos, emojis, long-form reflections, and even fragmented emotional narratives.
This has meaningful applications in **communication intelligence**, cyber-investigation, and emotional signal detection within large text streams.

> Instead of asking *“is this positive or negative?”*, we ask:
> **What emotional state does this reflect, and how activated is it?**

---

## 🧠 Technical Motivation

Common sentiment models struggle with:

1. Sarcasm/Tone-Inversion: "Love that I'm working late" → incorrectly positive
2. Mixed emotional phrases: "Good day, but my chest feels tight"
3. Slang/Emojis/Typos: "supa mid day ngl, brain frieddd"
4. Long-form journaling: Sentences that are closer to a paragraph in length.

To address this, I moved to a **semantic vector embedding + centroid inference approach**.

---
## 🤓 Methodology
Each library used in the project is there to build the foundations for the NLP pipeline and allows for greater interpretability, extensibility, and semantic nuance.
- FastAPI: Lightweight, async-native backend framework with automatic validation + OpenAPI schema. Low overhead, perfect for real-time journaling input.
- HuggingFace transformers: Provides access to LLM-grade embedding models + zero-shot classification. Enables semantic mood inference from messy language with no manual retraining.
- sentence-transformers: Converts journal entries into dense vector embeddings, necessary for cosine-similarity based classification.
- numpy: Enables fast vector operations for cosine similarity, centroid updates, and valence/arousal scoring.
- Pydantic: Ensures journal entries and response objects are well-typed and validated across the backend.
- uvicorn: The async server powering FastAPI. Which is needed for real-time journaling UI responsiveness.
- emoji: Detects unicode emojis which is essential for emotional context extraction beyond plain text.
- pytest: A tool that helps provide more insight when testing the model. Enables reproducible evaluation across ambiguous language, slang, typos, poetry, and emotional edge cases.

Instead of relying on static classification, this project uses extensible vector spaces. The main advantage of this means embeddings are used to learn contextually instead of by keyword. Additionally, the use of centroids allows for a classification by similarity instead of rigid rules. By utilizing cosine similarity, the model is also adaptive, learning from the user over new entries.

The backend is ran as a package instead of a regular script to prevent internal imports from breaking the program. Having this backend methodology also improves the scalability of this application, allowing for developers to expand much easier with clean and consistent test execution and module discovery.

This project uses Vite + React for the frontend to deliver a fast, seamless UI with hot refreshes and lots of flexibility in development. The main goal of the frontend was to make the mood journal feel **alive** instead of just a place for users to write sentences. Additionally, Vite's dev proxy makes backend routing clean during development, allowing for much quicker backend to frontend connection. This combination keeps frontend development as easy as it can be, while also allowing the interface to adapt and grow around emotional data.

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
to include more emotional representation.

---

## 📈 Model Evolution & Accuracy History

### Phase 1 — Baseline (Zero-Shot + Sentiment)

* Good for simple sentences
* Failed edge cases and sarcasm
* **19/25 mismatches (24% success)**

Upon seeing this, I realized I had to pivot my approach to the problem. Zero-shot models and sentiment analysis were not going to be enough to classify the level of messiness and ambiguity I was aiming for. 

### Phase 2 — Embedding Centroids

* Introduced vector similarity classification
* Better performance on ambiguity and tone
* **23/25 correct (92%)**

The Embedding Centroids methodology greatly increased the accuracy of the tests by ensuring a threshold of 80% and higher on all tests. It even scored 92% on the ambiguity test, which was alarmingly high. My next thought was to create a much larger dataset to test the model on a greater scale. 

### Phase 3 — Stress Test: 100-Entry Dataset

The dataset included:
1. Long-form text and journaling paragraphs
2. Slang + shorthand (“idk, kinda mid ngl”)
3. Poems + abstract metaphorical entries
4. Emoji-only and near-gibberish cases
5. Empty strings, malformed grammar

Initial score: **66%**
This was alarming to see at first, especially regarding the fact that it was previously scoring ~80-90% on the tests with 25 entries of each. Why did essentially combining all of these entries produce such low accuracy? Interpretation: Many “incorrect” outputs were *actually valid interpretations*. Upon further inspection, an important conclusion was reached: emotional classification is inherently and largely subjective. Tackling this problem meant accounting for that subjectivity or nuance in classification by loosening the evaluation criteria a bit. 

**Evaluation updated → Alternate valid labels allowed.**
After adding alternate valid labels for mood and energy, accuracy improved, with a reduction in false negatives. 

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

Future Implementations
7. Explainable reasoning output: helps user understand why certain entries are classified as what they are.
8. Graph/improved tracking over periods of time to visualize mental health across time
9. Multi-label mood distributions: improves overall emotion scoring
10. User focused centroids: helps give more accurate classifications
11. Real-time emotion classification: improves user experience and immersion

These future implementations would help improve the model's accuracy and provide some more clarity on how each entry is classified. Additionally, adding more real-time support and improving the interactivity of the UI would help user immersion and create a better experience overall. 

---

## 🚀 Running Locally
Using a virtual environment:
```bash
pip install -r requirements.txt
```

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


