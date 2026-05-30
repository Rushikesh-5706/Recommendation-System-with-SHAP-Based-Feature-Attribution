import os
import sys
import json
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PopularityBaseline:
    def __init__(self):
        self.item_popularity = {}
        self.popular_items = []

    def fit(self, train_df):
        counts = train_df["item_idx"].value_counts()
        self.item_popularity = counts.to_dict()
        self.popular_items = counts.index.tolist()

    def recommend(self, user_idx, n, exclude_items):
        exclude_set = set(exclude_items)
        recommendations = []
        for item in self.popular_items:
            if item not in exclude_set:
                recommendations.append(item)
                if len(recommendations) == n:
                    break
        return recommendations


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout_rate):
        super(NeuMF, self).__init__()

        # Define exact named attributes required
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
        # GMF branch
        gmf_user = self.gmf_user_emb(user_idx)
        gmf_item = self.gmf_item_emb(item_idx)
        gmf_out = gmf_user * gmf_item

        # MLP branch
        mlp_user = self.mlp_user_emb(user_idx)
        mlp_item = self.mlp_item_emb(item_idx)
        mlp_in = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_out = self.mlp_layers(mlp_in)

        # Output
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        logit = self.output_layer(combined)
        score = torch.sigmoid(logit)
        return score


class NCFDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, label = self.interactions[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def evaluate_baseline(baseline, test_df, train_interactions):
    print("Evaluating PopularityBaseline...")
    test_users = test_df["user_idx"].unique()
    hits = []
    ndcgs = []

    # Pre-group test items for fast lookup
    test_user_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    for u in tqdm(test_users, desc="Baseline Eval"):
        test_items = test_user_items.get(u, set())
        if not test_items:
            continue

        exclude_items = train_interactions.get(u, set())
        recs = baseline.recommend(u, 10, exclude_items)

        hit_count = len(set(recs) & test_items)
        hits.append(
            hit_count / 10.0
        )  # Precision@10: fraction of recommended items that are in test set

        dcg = 0.0
        for rank, rec_item in enumerate(recs):
            if rec_item in test_items:
                dcg += 1.0 / np.log2(rank + 1 + 1)

        # Ideal DCG for 10 hits (if min(len(test_items), 10) hits were possible)
        idcg = sum(1.0 / np.log2(i + 1 + 1) for i in range(min(len(test_items), 10)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        "precision_at_10": float(np.mean(hits)),
        "ndcg_at_10": float(np.mean(ndcgs)),
    }

    return metrics


def evaluate_neumf(model, test_df, train_interactions, all_items, device):
    print("Evaluating NeuMF...")
    model.eval()
    test_users = test_df["user_idx"].unique()

    # Pre-group test items
    test_user_items_list = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    hits = []
    ndcgs = []

    with torch.no_grad():
        for u in tqdm(test_users, desc="NeuMF Eval"):
            u_test_items = test_user_items_list.get(u, [])
            if not u_test_items:
                continue

            pos_item = u_test_items[0]

            # Sample 99 negatives
            exclude_set = train_interactions.get(u, set()).union({pos_item})
            available_negatives = list(all_items - exclude_set)

            if len(available_negatives) < 99:
                neg_samples = available_negatives
            else:
                neg_samples = random.sample(available_negatives, 99)

            eval_items = [pos_item] + neg_samples

            users_tensor = torch.tensor([u] * len(eval_items), dtype=torch.long).to(
                device
            )
            items_tensor = torch.tensor(eval_items, dtype=torch.long).to(device)

            scores = model(users_tensor, items_tensor).squeeze().cpu().numpy()

            # Rank items by score
            item_scores = list(zip(eval_items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)

            ranked_items = [x[0] for x in item_scores]

            # Evaluate Hit and NDCG @ 10
            if pos_item in ranked_items[:10]:
                hits.append(1.0)
                rank = ranked_items.index(pos_item) + 1
                ndcgs.append(np.log2(2) / np.log2(rank + 1))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

    metrics = {
        "precision_at_10": float(np.mean(hits)),
        "ndcg_at_10": float(np.mean(ndcgs)),
    }

    return metrics


def main():
    processed_dir = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
    model_dir = os.getenv("MODEL_PATH", "./models")
    results_dir = os.getenv("RESULTS_PATH", "./results")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Hyperparameters
    emb_dim = int(os.getenv("EMBEDDING_DIM", "64"))
    lr = float(os.getenv("LEARNING_RATE", "0.001"))
    batch_size = int(os.getenv("BATCH_SIZE", "1024"))
    epochs = int(os.getenv("EPOCHS", "20"))
    dropout_rate = float(os.getenv("DROPOUT_RATE", "0.2"))
    neg_samples = int(os.getenv("NEG_SAMPLES_PER_POS", "4"))

    # Load mappings
    print("Loading mappings and datasets...")
    with open(os.path.join(processed_dir, "user_map.json"), "r") as f:
        user_map = json.load(f)
    with open(os.path.join(processed_dir, "item_map.json"), "r") as f:
        item_map = json.load(f)
    with open(os.path.join(processed_dir, "id_to_item.json"), "r") as f:
        id_to_item = json.load(f)
    with open(os.path.join(processed_dir, "item_names.json"), "r") as f:
        item_names = json.load(f)

    n_users = len(user_map)
    n_items = len(item_map)
    all_items = set(range(n_items))

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    # Build train_interactions (user_idx -> list of item_idxs)
    train_interactions_dict = (
        train_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    )
    # Also keep a set version for fast exclusion checking
    train_interactions_set = {
        u: set(items) for u, items in train_interactions_dict.items()
    }

    # Train & Evaluate Popularity Baseline
    baseline = PopularityBaseline()
    baseline.fit(train_df)
    baseline_metrics = evaluate_baseline(baseline, test_df, train_interactions_set)
    with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"Baseline Metrics: {baseline_metrics}")

    # Prepare NeuMF Training Data
    print("Preparing NeuMF training data...")
    # Positive interactions: ratings >= 4
    pos_train_df = train_df[train_df["rating"] >= 4]

    dataset_tuples = []
    for row in tqdm(
        pos_train_df.itertuples(index=False),
        desc="Negative Sampling",
        total=len(pos_train_df),
    ):
        u = row.user_idx
        i = row.item_idx
        dataset_tuples.append((u, i, 1.0))

        u_interacted = train_interactions_set.get(u, set())
        available_negatives = list(all_items - u_interacted)

        if available_negatives:
            sampled = random.sample(
                available_negatives, min(neg_samples, len(available_negatives))
            )
            for neg_i in sampled:
                dataset_tuples.append((u, neg_i, 0.0))

    dataset = NCFDataset(dataset_tuples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize NeuMF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = NeuMF(n_users, n_items, emb_dim, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training Loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_users, batch_items, batch_labels in dataloader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            preds = model(batch_users, batch_items).squeeze()

            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Mean Loss: {avg_loss:.4f}")

    # Evaluate NeuMF
    neumf_metrics = evaluate_neumf(
        model, test_df, train_interactions_set, all_items, device
    )
    with open(os.path.join(results_dir, "model_metrics.json"), "w") as f:
        json.dump(neumf_metrics, f, indent=4)
    print(f"NeuMF Metrics: {neumf_metrics}")

    # Persist Model
    print("Saving model artifact...")
    artifact = {
        "model_state_dict": model.state_dict(),
        "n_users": n_users,
        "n_items": n_items,
        "embedding_dim": emb_dim,
        "item_names": item_names,
        "user_map": user_map,
        "item_map": item_map,
        "id_to_item": id_to_item,
        "train_interactions": train_interactions_dict,
        "dropout_rate": dropout_rate,
    }
    joblib.dump(artifact, os.path.join(model_dir, "recommender.joblib"))
    print("Model training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
