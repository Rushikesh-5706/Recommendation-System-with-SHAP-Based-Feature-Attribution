import os
import sys
import json
import random
import argparse
import numpy as np
import joblib
import shap
import torch
import torch.nn as nn
from dotenv import load_dotenv

# Define NeuMF architecture locally so it doesn't need to be imported
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
    """
    Returns a SHAP-compatible prediction function for a fixed target item.
    
    The function receives a batch of binary interaction vectors (shape: n_samples x n_items)
    and returns predicted scores for the target item for each hypothetical user.
    
    User representation is derived by averaging the item embeddings of all items
    the hypothetical user has interacted with (binary vector = 1). This is the
    embedding aggregation approach described in the NCF explainability literature.
    
    Args:
        model: Trained NeuMF instance
        target_item_idx: Integer index of the item being explained
        n_items: Total number of items in catalog
        device: torch.device
    
    Returns:
        predict_fn: Callable[[np.ndarray], np.ndarray]
    """
    def predict_fn(X):
        # X shape: (n_samples, n_items), dtype float or binary
        model.eval()
        scores = []
        with torch.no_grad():
            t = torch.tensor([target_item_idx], dtype=torch.long).to(device)
            gmf_item_vec = model.gmf_item_emb(t)   # (1, emb_dim)
            mlp_item_vec = model.mlp_item_emb(t)   # (1, emb_dim)

            for row in X:
                interacted_indices = np.where(row > 0.5)[0]

                if len(interacted_indices) == 0:
                    scores.append(0.0)
                    continue

                items_t = torch.tensor(interacted_indices, dtype=torch.long).to(device)

                # Average item embeddings as user proxy
                gmf_user_agg = model.gmf_item_emb(items_t).mean(dim=0, keepdim=True)  # (1, emb_dim)
                mlp_user_agg = model.mlp_item_emb(items_t).mean(dim=0, keepdim=True)  # (1, emb_dim)

                # GMF branch
                gmf_out = gmf_user_agg * gmf_item_vec                          # (1, emb_dim)

                # MLP branch
                mlp_input = torch.cat([mlp_user_agg, mlp_item_vec], dim=1)    # (1, 2*emb_dim)
                mlp_out = model.mlp_layers(mlp_input)                          # (1, 64)

                # Output
                combined = torch.cat([gmf_out, mlp_out], dim=1)               # (1, emb_dim+64)
                score = torch.sigmoid(model.output_layer(combined)).item()
                scores.append(score)

        return np.array(scores, dtype=np.float64)

    return predict_fn

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explanation for a user-item pair.")
    parser.add_argument("--user_id", type=int, required=True, help="Raw user ID")
    parser.add_argument("--item_id", type=int, required=True, help="Raw movie ID")
    args = parser.parse_args()
    
    load_dotenv()
    model_dir = os.getenv('MODEL_PATH', './models')
    results_dir = os.getenv('RESULTS_PATH', './results')
    nsamples = int(os.getenv('SHAP_NSAMPLES', '150'))
    
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'recommender.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train_model.py first.")
        
    print("Loading model artifact...")
    artifact = joblib.load(model_path)
    
    user_map = artifact['user_map']
    item_map = artifact['item_map']
    id_to_item = artifact['id_to_item']
    item_names = artifact['item_names']
    train_interactions = artifact['train_interactions']
    n_users = artifact['n_users']
    n_items = artifact['n_items']
    emb_dim = artifact['embedding_dim']
    dropout_rate = artifact['dropout_rate']
    
    str_user_id = str(args.user_id)
    str_item_id = str(args.item_id)
    
    if str_user_id not in user_map:
        raise ValueError(f"User ID {args.user_id} not found in dataset")
    if str_item_id not in item_map:
        raise ValueError(f"Item ID {args.item_id} not found in dataset")
        
    user_idx = user_map[str_user_id]
    item_idx = item_map[str_item_id]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(n_users, n_items, emb_dim, dropout_rate)
    model.load_state_dict(artifact['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Building background dataset...")
    # Background dataset: 100 random users from training
    train_user_keys = list(train_interactions.keys())
    background_user_idxs = random.sample(train_user_keys, min(100, len(train_user_keys)))
    background = np.zeros((len(background_user_idxs), n_items), dtype=np.float32)
    for i, u_idx in enumerate(background_user_idxs):
        for itm in train_interactions[u_idx]:
            background[i, itm] = 1.0

    print("Building user vector...")
    # User's own interaction vector
    user_vector = np.zeros((1, n_items), dtype=np.float32)
    user_items = train_interactions.get(user_idx, [])
    for itm in user_items:
        user_vector[0, itm] = 1.0
        
    predict_fn = build_prediction_wrapper(model, item_idx, n_items, device)
    recommendation_score = float(predict_fn(user_vector)[0])
    
    print("Computing SHAP values...")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(user_vector, nsamples=nsamples, silent=True)
    shap_vec = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    
    if shap_vec.ndim == 2:
        shap_vec = shap_vec[0]
    
    print("Extracting top contributors...")
    item_shap_pairs = []
    for itm in user_items:
        item_shap_pairs.append((itm, shap_vec[itm]))
        
    item_shap_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Take up to top 5 with positive shap_value
    top_contributors_idx = [p for p in item_shap_pairs if p[1] > 0][:5]
    
    # If fewer than 1 positive contributor exists, take the top 1 regardless of sign
    if len(top_contributors_idx) == 0 and len(item_shap_pairs) > 0:
        top_contributors_idx = [item_shap_pairs[0]]
        
    top_contributors = []
    for itm, s_val in top_contributors_idx:
        raw_id = id_to_item[str(itm)]
        name = item_names[str(itm)]
        top_contributors.append({
            "item_id": int(raw_id),
            "item_name": name,
            "shap_value": float(s_val)
        })
        
    target_title = item_names[str(item_idx)]
    contributor_titles = [c['item_name'] for c in top_contributors]

    if len(contributor_titles) == 0:
        explanation = f"We recommend '{target_title}' based on your general viewing history."
    elif len(contributor_titles) == 1:
        explanation = f"We recommend '{target_title}' because you watched '{contributor_titles[0]}'."
    elif len(contributor_titles) == 2:
        explanation = f"We recommend '{target_title}' because you watched '{contributor_titles[0]}' and '{contributor_titles[1]}'."
    else:
        listed = "', '".join(contributor_titles[:-1])
        explanation = f"We recommend '{target_title}' because you watched '{listed}', and '{contributor_titles[-1]}'."

    output_data = {
        "user_id": args.user_id,
        "item_id": args.item_id,
        "recommendation_score": recommendation_score,
        "top_contributors": top_contributors,
        "human_readable_explanation": explanation
    }
    
    out_file = os.path.join(results_dir, 'explanation.json')
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Explanation saved to {out_file}")
    print(f"Explanation text: {explanation}")

if __name__ == '__main__':
    main()
