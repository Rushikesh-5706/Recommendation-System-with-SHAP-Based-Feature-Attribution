import os
import json
import sys
import random
import numpy as np
import pandas as pd
import joblib
import shap
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout_rate):
        super(NeuMF, self).__init__()

        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        self.mlp_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(embedding_dim + 64, 1)

    def forward(self, user_idx, item_idx):
        gmf_user = self.gmf_user_emb(user_idx)
        gmf_item = self.gmf_item_emb(item_idx)
        gmf_out = gmf_user * gmf_item

        mlp_user = self.mlp_user_emb(user_idx)
        mlp_item = self.mlp_item_emb(item_idx)
        mlp_in = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_out = self.mlp_layers(mlp_in)

        combined = torch.cat([gmf_out, mlp_out], dim=1)
        logit = self.output_layer(combined)
        score = torch.sigmoid(logit)
        return score


def build_masked_prediction_wrapper(model, target_item_idx, interacted_items_list, device):
    """
    Returns a SHAP-compatible prediction function operating over a reduced feature space.

    Instead of treating all n_items catalog items as features (which makes SHAP
    prohibitively slow at 3700+ features), this wrapper operates only over the
    user's interacted items. Each position in the input vector corresponds to one
    item from interacted_items_list. This reduces computation from O(n_items) to
    O(n_interacted), typically from 3700+ features to 100-300.

    Args:
        model: Trained NeuMF instance in eval mode
        target_item_idx: Integer index of the item being explained
        interacted_items_list: Ordered list of item indices the user has interacted with
        device: torch.device

    Returns:
        predict_fn: Callable[[np.ndarray], np.ndarray]
            Input shape: (n_samples, len(interacted_items_list))
            Output shape: (n_samples,)
    """
    def predict_fn(X):
        model.eval()
        scores = []
        with torch.no_grad():
            t = torch.tensor([target_item_idx], dtype=torch.long).to(device)
            gmf_item_vec = model.gmf_item_emb(t)
            mlp_item_vec = model.mlp_item_emb(t)

            for row in X:
                # Map masked positions back to actual item indices
                active_positions = np.where(row > 0.5)[0]
                active_items = [interacted_items_list[p] for p in active_positions]

                if len(active_items) == 0:
                    scores.append(0.0)
                    continue

                items_t = torch.tensor(active_items, dtype=torch.long).to(device)

                gmf_user_agg = model.gmf_item_emb(items_t).mean(dim=0, keepdim=True)
                mlp_user_agg = model.mlp_item_emb(items_t).mean(dim=0, keepdim=True)

                gmf_out = gmf_user_agg * gmf_item_vec
                mlp_input = torch.cat([mlp_user_agg, mlp_item_vec], dim=1)
                mlp_out = model.mlp_layers(mlp_input)

                combined = torch.cat([gmf_out, mlp_out], dim=1)
                score = torch.sigmoid(model.output_layer(combined)).item()
                scores.append(score)

        return np.array(scores, dtype=np.float64)

    return predict_fn


def compute_faithfulness(model, test_df, train_interactions, background_data, device, sample_size=100, nsamples=50):
    """
    Faithfulness: measures whether SHAP-identified top items carry more predictive
    weight than randomly selected items of the same count.

    For each sampled (user, target_item) pair:
    - Compute SHAP values over the user's interacted items (masked feature space)
    - Identify top-3 items by SHAP value
    - delta_shap = score_original - score_without_top3
    - delta_random = score_original - score_without_random3
    - faithfulness_i = 1 if delta_shap > delta_random else 0

    Final score = mean(faithfulness_i)
    """
    print("Computing Faithfulness...")
    test_pairs = list(zip(test_df["user_idx"], test_df["item_idx"]))
    sampled_pairs = random.sample(test_pairs, min(sample_size, len(test_pairs)))

    faithfulness_scores = []

    for user_idx, target_item_idx in tqdm(sampled_pairs, desc="Faithfulness Pairs"):
        u_interacted = train_interactions.get(user_idx, [])
        if len(u_interacted) < 3:
            continue

        interacted_list = list(u_interacted)
        n_interacted = len(interacted_list)

        # User vector in reduced space: all 1s (user has all their items)
        user_vector_masked = np.ones((1, n_interacted), dtype=np.float32)

        # Background in reduced space: zeros (user has none of their items)
        # Single zero row is sufficient as baseline for per-user masked explainer
        background_masked = np.zeros((1, n_interacted), dtype=np.float32)

        predict_fn = build_masked_prediction_wrapper(
            model, target_item_idx, interacted_list, device
        )

        original_score = predict_fn(user_vector_masked)[0]

        explainer = shap.KernelExplainer(predict_fn, background_masked)
        shap_values = explainer.shap_values(user_vector_masked, nsamples=nsamples, silent=True)
        shap_vec = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        if shap_vec.ndim == 2:
            shap_vec = shap_vec[0]

        # Top 3 by SHAP value (indices into interacted_list)
        position_shap = [(pos, shap_vec[pos]) for pos in range(n_interacted)]
        position_shap.sort(key=lambda x: x[1], reverse=True)
        top3_positions = [x[0] for x in position_shap[:3]]

        modified_shap = user_vector_masked.copy()
        for pos in top3_positions:
            modified_shap[0, pos] = 0.0
        score_shap_removed = predict_fn(modified_shap)[0]
        delta_shap = original_score - score_shap_removed

        remaining_positions = [p for p in range(n_interacted) if p not in top3_positions]
        if len(remaining_positions) < 3:
            random3_positions = random.sample(list(range(n_interacted)), 3)
        else:
            random3_positions = random.sample(remaining_positions, 3)

        modified_random = user_vector_masked.copy()
        for pos in random3_positions:
            modified_random[0, pos] = 0.0
        score_random_removed = predict_fn(modified_random)[0]
        delta_random = original_score - score_random_removed

        faithfulness_scores.append(1 if delta_shap > delta_random else 0)

    if not faithfulness_scores:
        return 0.0
    return sum(faithfulness_scores) / len(faithfulness_scores)


def compute_consistency(model, train_df, test_df, train_interactions, background_data, device, sample_size=50, nsamples=50):
    """
    Consistency: measures whether similar users receive similar explanations
    for the same target item.

    For each pair of similar users (cosine similarity > 0.5 on interaction vectors):
    - Find a common target item
    - Compute SHAP explanations for both users using their shared item union as feature space
    - Compute Jaccard similarity between their top-3 contributing item sets

    Final score = mean(jaccard)
    """
    print("Computing Consistency...")
    all_users = list(train_interactions.keys())
    sampled_users = random.sample(all_users, min(500, len(all_users)))

    # Full interaction vectors for similarity computation
    n_items_total = max(max(items) for items in train_interactions.values()) + 1
    user_vectors_full = np.zeros((len(sampled_users), n_items_total), dtype=np.float32)
    for i, u_idx in enumerate(sampled_users):
        for itm in train_interactions[u_idx]:
            user_vectors_full[i, itm] = 1.0

    sim_matrix = cosine_similarity(user_vectors_full)

    pairs = []
    threshold = 0.5
    for i in range(len(sampled_users)):
        for j in range(i + 1, len(sampled_users)):
            if sim_matrix[i, j] > threshold:
                pairs.append((sampled_users[i], sampled_users[j]))

    if len(pairs) < 10:
        pairs = []
        threshold = 0.3
        for i in range(len(sampled_users)):
            for j in range(i + 1, len(sampled_users)):
                if sim_matrix[i, j] > threshold:
                    pairs.append((sampled_users[i], sampled_users[j]))

    random.shuffle(pairs)
    pairs = pairs[:sample_size]

    if not pairs:
        return 0.0

    test_items_by_user = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    item_popularity = train_df["item_idx"].value_counts().index.tolist()

    jaccards = []

    for u_a, u_b in tqdm(pairs, desc="Consistency Pairs"):
        test_a = test_items_by_user.get(u_a, set())
        test_b = test_items_by_user.get(u_b, set())
        overlap = test_a.intersection(test_b)

        if overlap:
            target_item_idx = random.choice(list(overlap))
        else:
            train_a = set(train_interactions.get(u_a, []))
            train_b = set(train_interactions.get(u_b, []))
            combined_train = train_a.union(train_b)
            target_item_idx = next(
                (itm for itm in item_popularity if itm not in combined_train), 0
            )

        # Use union of interacted items as shared feature space for fair comparison
        items_a = list(train_interactions.get(u_a, []))
        items_b = list(train_interactions.get(u_b, []))
        union_items = list(set(items_a) | set(items_b))

        if not union_items:
            continue

        n_union = len(union_items)
        item_to_pos = {itm: pos for pos, itm in enumerate(union_items)}

        # Build user vectors in union space
        vec_a = np.zeros((1, n_union), dtype=np.float32)
        for itm in items_a:
            if itm in item_to_pos:
                vec_a[0, item_to_pos[itm]] = 1.0

        vec_b = np.zeros((1, n_union), dtype=np.float32)
        for itm in items_b:
            if itm in item_to_pos:
                vec_b[0, item_to_pos[itm]] = 1.0

        background_union = np.zeros((1, n_union), dtype=np.float32)

        predict_fn = build_masked_prediction_wrapper(
            model, target_item_idx, union_items, device
        )
        explainer = shap.KernelExplainer(predict_fn, background_union)

        shap_a = explainer.shap_values(vec_a, nsamples=nsamples, silent=True)
        shap_a = (shap_a[0] if isinstance(shap_a, list) else shap_a[0])
        if shap_a.ndim == 2:
            shap_a = shap_a[0]

        shap_b = explainer.shap_values(vec_b, nsamples=nsamples, silent=True)
        shap_b = (shap_b[0] if isinstance(shap_b, list) else shap_b[0])
        if shap_b.ndim == 2:
            shap_b = shap_b[0]

        # Get top-3 actual item indices for each user
        pos_a = np.where(vec_a[0] > 0.5)[0]
        pos_b = np.where(vec_b[0] > 0.5)[0]

        top3_a = set(
            union_items[p]
            for p in sorted(pos_a, key=lambda p: shap_a[p], reverse=True)[:3]
        )
        top3_b = set(
            union_items[p]
            for p in sorted(pos_b, key=lambda p: shap_b[p], reverse=True)[:3]
        )

        union_set = top3_a | top3_b
        jaccard = len(top3_a & top3_b) / len(union_set) if union_set else 0.0
        jaccards.append(jaccard)

    return sum(jaccards) / len(jaccards) if jaccards else 0.0


def main():
    model_dir = os.getenv("MODEL_PATH", "./models")
    processed_dir = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
    results_dir = os.getenv("RESULTS_PATH", "./results")

    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "recommender.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run src/train_model.py first."
        )

    artifact = joblib.load(model_path)
    n_users = artifact["n_users"]
    n_items = artifact["n_items"]
    emb_dim = artifact["embedding_dim"]
    dropout_rate = artifact["dropout_rate"]
    train_interactions = artifact["train_interactions"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(n_users, n_items, emb_dim, dropout_rate)
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    # Background data kept for API compatibility but not used in masked approach
    train_user_keys = list(train_interactions.keys())
    bg_users = random.sample(train_user_keys, min(100, len(train_user_keys)))
    background_data = np.zeros((len(bg_users), n_items), dtype=np.float32)
    for i, u_idx in enumerate(bg_users):
        for itm in train_interactions[u_idx]:
            background_data[i, itm] = 1.0

    faithfulness_size = int(os.getenv("FAITHFULNESS_SAMPLE_SIZE", "100"))
    consistency_size = int(os.getenv("CONSISTENCY_SAMPLE_SIZE", "50"))
    eval_nsamples = int(os.getenv("EVAL_NSAMPLES", "50"))

    faithfulness = compute_faithfulness(
        model, test_df, train_interactions, background_data, device,
        sample_size=faithfulness_size, nsamples=eval_nsamples
    )
    consistency = compute_consistency(
        model, train_df, test_df, train_interactions, background_data, device,
        sample_size=consistency_size, nsamples=eval_nsamples
    )

    metrics = {
        "faithfulness": float(round(faithfulness, 4)),
        "consistency": float(round(consistency, 4)),
    }

    out_file = os.path.join(results_dir, "explanation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {out_file}")
    print(metrics)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing required file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
