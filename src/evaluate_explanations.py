import os
import json
import random
import numpy as np
import pandas as pd
import joblib
import shap
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Reconstruct NeuMF locally
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
            nn.ReLU()
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

def build_prediction_wrapper(model, target_item_idx, n_items, device):
    def predict_fn(X):
        model.eval()
        scores = []
        with torch.no_grad():
            t = torch.tensor([target_item_idx], dtype=torch.long).to(device)
            gmf_item_vec = model.gmf_item_emb(t)
            mlp_item_vec = model.mlp_item_emb(t)

            for row in X:
                interacted_indices = np.where(row > 0.5)[0]
                if len(interacted_indices) == 0:
                    scores.append(0.0)
                    continue

                items_t = torch.tensor(interacted_indices, dtype=torch.long).to(device)

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

def compute_faithfulness(model, test_df, train_interactions, n_items, background_data, device):
    print("Computing Faithfulness...")
    test_pairs = list(zip(test_df['user_idx'], test_df['item_idx']))
    sample_size = min(100, len(test_pairs))
    sampled_pairs = random.sample(test_pairs, sample_size)
    
    faithfulness_scores = []
    
    for user_idx, target_item_idx in tqdm(sampled_pairs, desc="Faithfulness Pairs"):
        # Build user vector
        user_vector = np.zeros((1, n_items), dtype=np.float32)
        u_interacted = train_interactions.get(user_idx, [])
        for itm in u_interacted:
            user_vector[0, itm] = 1.0
            
        predict_fn = build_prediction_wrapper(model, target_item_idx, n_items, device)
        
        explainer = shap.KernelExplainer(predict_fn, background_data)
        shap_values = explainer.shap_values(user_vector, nsamples=50, silent=True)
        shap_vec = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        if shap_vec.ndim == 2:
            shap_vec = shap_vec[0]
            
        original_score = predict_fn(user_vector)[0]
        
        interacted = np.where(user_vector[0] > 0.5)[0].tolist()
        if len(interacted) < 3:
            continue
            
        # Get top 3 by SHAP
        interacted_shap = [(itm, shap_vec[itm]) for itm in interacted]
        interacted_shap.sort(key=lambda x: x[1], reverse=True)
        top3_shap = [x[0] for x in interacted_shap[:3]]
        
        modified_shap = user_vector.copy()
        for idx in top3_shap:
            modified_shap[0, idx] = 0.0
        score_shap_removed = predict_fn(modified_shap)[0]
        delta_shap = original_score - score_shap_removed
        
        # Random 3
        remaining_interacted = list(set(interacted) - set(top3_shap))
        if len(remaining_interacted) < 3:
            random3 = random.sample(interacted, 3)
        else:
            random3 = random.sample(remaining_interacted, 3)
            
        modified_random = user_vector.copy()
        for idx in random3:
            modified_random[0, idx] = 0.0
        score_random_removed = predict_fn(modified_random)[0]
        delta_random = original_score - score_random_removed
        
        if delta_shap > delta_random:
            faithfulness_scores.append(1)
        else:
            faithfulness_scores.append(0)
            
    if len(faithfulness_scores) == 0:
        return 0.0
    return sum(faithfulness_scores) / len(faithfulness_scores)

def compute_consistency(model, train_df, test_df, train_interactions, n_items, background_data, device):
    print("Computing Consistency...")
    all_users = list(train_interactions.keys())
    sampled_users = random.sample(all_users, min(500, len(all_users)))
    
    user_vectors = np.zeros((len(sampled_users), n_items), dtype=np.float32)
    for i, u_idx in enumerate(sampled_users):
        for itm in train_interactions[u_idx]:
            user_vectors[i, itm] = 1.0
            
    sim_matrix = cosine_similarity(user_vectors)
    
    pairs = []
    threshold = 0.5
    for i in range(len(sampled_users)):
        for j in range(i + 1, len(sampled_users)):
            if sim_matrix[i, j] > threshold:
                pairs.append((sampled_users[i], sampled_users[j], i, j))
                
    if len(pairs) < 10:
        pairs = []
        threshold = 0.3
        for i in range(len(sampled_users)):
            for j in range(i + 1, len(sampled_users)):
                if sim_matrix[i, j] > threshold:
                    pairs.append((sampled_users[i], sampled_users[j], i, j))
                    
    random.shuffle(pairs)
    pairs = pairs[:50]
    
    if len(pairs) == 0:
        return 0.0
        
    test_items_by_user = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    item_popularity = train_df['item_idx'].value_counts().index.tolist()
    
    jaccards = []
    
    for u_a, u_b, idx_a, idx_b in tqdm(pairs, desc="Consistency Pairs"):
        test_a = test_items_by_user.get(u_a, set())
        test_b = test_items_by_user.get(u_b, set())
        overlap = test_a.intersection(test_b)
        
        target_item_idx = None
        if overlap:
            target_item_idx = random.choice(list(overlap))
        else:
            train_a = set(train_interactions.get(u_a, []))
            train_b = set(train_interactions.get(u_b, []))
            combined_train = train_a.union(train_b)
            for itm in item_popularity:
                if itm not in combined_train:
                    target_item_idx = itm
                    break
                    
        if target_item_idx is None:
            target_item_idx = 0
            
        predict_fn = build_prediction_wrapper(model, target_item_idx, n_items, device)
        explainer = shap.KernelExplainer(predict_fn, background_data)
        
        vec_a = user_vectors[idx_a:idx_a+1]
        vec_b = user_vectors[idx_b:idx_b+1]
        
        shap_a = explainer.shap_values(vec_a, nsamples=50, silent=True)[0]
        shap_b = explainer.shap_values(vec_b, nsamples=50, silent=True)[0]
        if shap_a.ndim == 2: shap_a = shap_a[0]
        if shap_b.ndim == 2: shap_b = shap_b[0]
        
        interacted_a = np.where(vec_a[0] > 0.5)[0]
        interacted_b = np.where(vec_b[0] > 0.5)[0]
        
        top3_a = set([x[0] for x in sorted([(i, shap_a[i]) for i in interacted_a], key=lambda x: x[1], reverse=True)[:3]])
        top3_b = set([x[0] for x in sorted([(i, shap_b[i]) for i in interacted_b], key=lambda x: x[1], reverse=True)[:3]])
        
        union = top3_a.union(top3_b)
        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(top3_a.intersection(top3_b)) / len(union)
        jaccards.append(jaccard)
        
    if len(jaccards) == 0:
        return 0.0
    return sum(jaccards) / len(jaccards)

def main():
    model_dir = os.getenv('MODEL_PATH', './models')
    processed_dir = os.getenv('PROCESSED_DATA_PATH', './data/processed')
    results_dir = os.getenv('RESULTS_PATH', './results')
    
    artifact = joblib.load(os.path.join(model_dir, 'recommender.joblib'))
    n_users = artifact['n_users']
    n_items = artifact['n_items']
    emb_dim = artifact['embedding_dim']
    dropout_rate = artifact['dropout_rate']
    train_interactions = artifact['train_interactions']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(n_users, n_items, emb_dim, dropout_rate)
    model.load_state_dict(artifact['model_state_dict'])
    model.to(device)
    
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    
    train_user_keys = list(train_interactions.keys())
    bg_users = random.sample(train_user_keys, min(100, len(train_user_keys)))
    background_data = np.zeros((len(bg_users), n_items), dtype=np.float32)
    for i, u_idx in enumerate(bg_users):
        for itm in train_interactions[u_idx]:
            background_data[i, itm] = 1.0
            
    faithfulness = compute_faithfulness(model, test_df, train_interactions, n_items, background_data, device)
    consistency = compute_consistency(model, train_df, test_df, train_interactions, n_items, background_data, device)
    
    metrics = {
        "faithfulness": float(round(faithfulness, 4)),
        "consistency": float(round(consistency, 4))
    }
    
    out_file = os.path.join(results_dir, 'explanation_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {out_file}")
    print(metrics)

if __name__ == '__main__':
    main()
