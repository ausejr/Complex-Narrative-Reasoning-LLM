# IF-QSR: Boosting Large Language Models’ Reasoning in Intricate Narratives

This repository contains the official implementation of our paper:
"**IF-QSR: Boosting Large Language Models’ Reasoning in Intricate Narratives**"

## 1. Introduction

Large Language Models (LLMs) have achieved remarkable progress in various NLP tasks, but still face severe challenges in handling complex narrative reasoning. Specifically, existing LLMs often exhibit "shortcut dependency," struggling with rigorous multi-step logical chain reasoning, and may be misled by local information, leading to reasoning deviations.

To address these challenges, we propose a novel framework named **IF-QSR (Information Fusion and Annotation - Query-based Structured Reasoning)**. This framework systematically solves the challenges LLMs encounter in complex narrative tasks, aiming to enhance their ability to integrate dispersed information, ensure logical coherence, and perform systematic, unbiased reasoning even with incomplete information.

## 2. Key Modules

The IF-QSR framework comprises three synergistic modules:

* **Information Fusion and Annotation (IF) Module:** Dedicated to solving the challenges of dispersed original information and cross-information integration. It focuses on pre-processing and enhancing raw narrative text into structured, model-comprehensible inputs.
* **Structured Reasoning (SR) Module:** Concentrates on improving logical coherence and systematic thinking, guiding the model to follow predefined reasoning steps and progressively build causal relationship chains.
* **Query-based Hypothesis Generation (Q) Module:** Designed to handle complex uncertainty scenarios. It decomposes complex problems into simpler subproblems and explores multiple possibilities through heuristic hypothesis generation, thereby reducing hallucinations and improving reasoning accuracy.

## 3. Setup and Installation

This project requires Python **3.12**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ausejr/Complex-Narrative-Reasoning-LLM.git
    cd Complex-Narrative-Reasoning-LLM
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # For Windows: .\venv\Scripts\activate
    # For macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 4. Dataset

To comprehensively evaluate the reasoning capabilities of the IF-QSR framework, we constructed a specialized dataset for complex narrative reasoning cases, encompassing simple, medium, and complex difficulty levels.

**Data Availability and Structure:**
Due to the substantial size of the full dataset, **a typical example case for each difficulty level (simple, medium, and complex)** from our dataset is directly included in this repository under the `dataset` directory. These examples serve to demonstrate the dataset format and structure used in our experiments.

The dataset files are typically in `.json` format, with each entry containing a case description, predefined_cues, and corresponding gold-standard answer. After downloading the full dataset, please place the files into the `dataset` directory within this project.

---
