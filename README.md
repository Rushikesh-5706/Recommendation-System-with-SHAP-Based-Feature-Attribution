# Explainable Recommendation System with SHAP-Based Feature Attribution

A Neural Collaborative Filtering recommendation model with SHAP-powered item-level explanations,
built for transparent and auditable predictions in compliance with modern AI accountability standards.

---

## Table of Contents

- [Architecture](#architecture)
- [Model Choice and Justification](#model-choice-and-justification)
- [SHAP Integration Design](#shap-integration-design)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Output Files Reference](#output-files-reference)
- [Explanation Metrics](#explanation-metrics)
- [Runtime Notes](#runtime-notes)

---

## Architecture

The system is built as a multi-stage offline pipeline with a conceptual service layer for real-time explanation generation.

```text
Raw Dataset (MovieLens 1M)
         |
         v
+-------------------------+
|  1. Data Preprocessing  |  -> train.csv, test.csv
|     & Splitting         |     user_map.json, item_map.json
+-------------------------+
         |
         v
+-------------------------+        +----------------------------------+
|  2. Model Training      |        |         Service Logic            |
|     NeuMF (GMF + MLP)   |------->| Input: User ID                   |
|                         |        | -> Generate Top-N Candidates     |
+-------------------------+        | -> SHAP KernelExplainer Wrapper  |
         |     Trained Model       | -> Explanation Template Logic    |
         v                         | Output: Human-Readable Text      |
+-----------------------------------+
|  3. Recommendation & Explanation  |
|     Service                       |
+-----------------------------------+
         |
         v
+-------------------------------+
|  4. Evaluation                |
|  4a. Model Performance        |  -> model_metrics.json
|  4b. Explanation Quality      |  -> explanation_metrics.json
|  4c. User Study               |  -> user_study_report.md
+-------------------------------+
         |
         v
    Final Output
```

The offline pipeline is fully containerized and runs sequentially via docker-compose.

---

## Model Choice and Justification

Three candidate approaches were evaluated for this system:

| Model | Non-Linear Patterns | Personalization | SHAP Compatibility | Complexity |
|---|---|---|---|---|
| Popularity Baseline | No | None | Not applicable | O(n log n) |
| SVD / Matrix Factorization | No | High | Moderate (linear latent space) | O(k * iterations) |
| NeuMF (GMF + MLP) | Yes | High | Via embedding aggregation | O(E * B * n) |

**NeuMF was selected** for these reasons:

The model combines two complementary learning approaches. The Generalized Matrix Factorization (GMF) branch models linear feature interactions through element-wise product of user and item embeddings, capturing relationships similar to classical MF. The Multi-Layer Perceptron (MLP) branch learns non-linear interactions through a deep network over the concatenated embeddings. Fusing both branches allows the model to exploit the strengths of each.

From a SHAP integration standpoint, NeuMF is well-suited because its item embedding tables can be used to derive a user representation from an arbitrary interaction history. Given any binary vector of item interactions, the corresponding item embeddings can be averaged to form a user proxy, which is then passed through the model's forward logic. This embedding aggregation approach makes the model fully compatible with SHAP's KernelExplainer without modifying the trained weights.

SVD would have offered simpler SHAP integration but lacks the capacity to model non-linear preference patterns. The popularity baseline serves only as a lower-bound reference — it has zero personalization capability.

---

## SHAP Integration Design

**Why KernelExplainer:** SHAP's KernelExplainer is model-agnostic, requiring only a callable prediction function. This is necessary here because the model takes user and item IDs as input, not feature vectors. A custom wrapper translates binary interaction vectors into model-compatible representations, enabling KernelExplainer to treat each catalog item as a feature.

**Prediction wrapper:** Given a batch of binary interaction vectors (shape: n_samples x n_items), the wrapper:
1. For each row, identifies items where the value is 1 (interacted items)
2. Retrieves those items' embeddings from both the GMF and MLP embedding tables
3. Averages them to form a hypothetical user representation
4. Passes through the GMF branch (element-wise product) and MLP branch (concatenation through layers)
5. Returns the sigmoid-activated output score for the fixed target item

**Background dataset:** 100 randomly sampled training users, each represented as a binary interaction vector. This represents the baseline state — the expected model output when no item-specific interaction history is assumed.

**nsamples parameter:** Set to 150 for explanation generation and 50 for batch evaluation. Lower values trade explanation accuracy for runtime. These values represent a practical tradeoff for a catalog of ~3,700 items.

---

## Project Structure

```
.
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Downloaded dataset (not committed)
│   └── processed/        # train.csv, test.csv, mapping JSONs
├── models/               # Saved model artifact (not committed)
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── baseline_metrics.json
│   ├── model_metrics.json
│   ├── explanation.json
│   ├── explanation_metrics.json
│   └── user_study_report.md
└── src/
    ├── prepare_data.py
    ├── train_model.py
    ├── generate_explanation.py
    └── evaluate_explanations.py
```

---

## Setup and Installation

### Prerequisites

- Docker and docker-compose installed
- Internet access for dataset download (~25 MB)
- At least 4 GB RAM available for the training job

### Quick Start (Docker)

```bash
git clone https://github.com/Rushikesh-5706/Recommendation-System-with-SHAP-Based-Feature-Attribution.git
cd Recommendation-System-with-SHAP-Based-Feature-Attribution
cp .env.example .env
docker-compose up --build
```

The pipeline runs sequentially: data preparation → model training → explanation generation → evaluation. All output files are written to `data/processed/`, `models/`, and `results/`.

### Manual Execution (without Docker)

```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/train_model.py
python src/generate_explanation.py --user_id 1 --item_id 2
python src/evaluate_explanations.py
```

---

## Running the Pipeline

**Data preparation only:**
```bash
python src/prepare_data.py
```

**Model training:**
```bash
python src/train_model.py
```

**Generate explanation for a specific user-item pair:**
```bash
python src/generate_explanation.py --user_id <raw_user_id> --item_id <raw_movie_id>
```

**Evaluate explanation quality:**
```bash
python src/evaluate_explanations.py
```

---

## Output Files Reference

| File | Location | Description |
|---|---|---|
| train.csv | data/processed/ | Training ratings with user_idx, item_idx, rating |
| test.csv | data/processed/ | Test ratings with same schema |
| user_map.json | data/processed/ | Raw user ID to integer index mapping |
| item_map.json | data/processed/ | Raw movie ID to integer index mapping |
| recommender.joblib | models/ | Trained NeuMF model artifact with metadata |
| baseline_metrics.json | results/ | Popularity baseline Precision@10 and NDCG@10 |
| model_metrics.json | results/ | NeuMF Precision@10 and NDCG@10 |
| explanation.json | results/ | SHAP explanation for a user-item pair |
| explanation_metrics.json | results/ | Faithfulness and consistency scores |
| user_study_report.md | results/ | Qualitative evaluation report |

---

## Explanation Metrics

**Faithfulness** measures whether the items identified as top contributors by SHAP actually carry more predictive weight than randomly selected items. For each test pair, the prediction drop from removing top-SHAP items is compared against removing random items. A score of 1.0 means SHAP-identified items always matter more than random ones.

```
faithfulness = (1/N) * sum_i [ 1 if delta_shap_i > delta_random_i else 0 ]
```

**Consistency** measures whether similar users receive similar explanations for the same recommended item. User similarity is measured by cosine similarity on binary interaction vectors. Explanation similarity is measured by Jaccard similarity on the top-3 contributing item sets.

```
consistency = (1/M) * sum_j [ |top3_u1 ∩ top3_u2| / |top3_u1 ∪ top3_u2| ]
```

---

## Runtime Notes

| Step | Approximate Runtime |
|---|---|
| Data preparation | 2-3 minutes (includes download) |
| Model training (CPU, 20 epochs) | 15-25 minutes |
| Explanation generation (single pair) | 3-6 minutes |
| Evaluation (100 faithfulness + 50 consistency pairs) | 20-40 minutes |

SHAP's KernelExplainer scales as O(nsamples * n_features). For a catalog of ~3,700 items, nsamples=150 is a practical setting. Using a GPU reduces training time significantly but does not affect SHAP computation speed, which is CPU-bound.
