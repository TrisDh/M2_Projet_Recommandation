#!/usr/bin/env python3
"""
Script de test et d'évaluation pour le système de recommandation.
Charge les embeddings, évalue les modèles Content-Based et Matrix Factorization,
et affiche les statistiques descriptives.

Usage: python test.py
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# =============================================================================
# 1) Métriques d'évaluation
# =============================================================================

def precision_at_k(reco, relevant, k=10):
    """Precision@K : proportion d'items pertinents parmi les K recommandations."""
    return len(set(reco[:k]) & relevant) / k


def recall_at_k(reco, relevant, k=10):
    """Recall@K : proportion des items pertinents retrouvés dans les K recommandations."""
    return len(set(reco[:k]) & relevant) / len(relevant) if len(relevant) > 0 else 0


def ndcg_at_k(reco, relevant, k=10):
    """NDCG@K : mesure la qualité du classement (les bons items doivent apparaître en haut)."""
    dcg = 0.0
    for i, item in enumerate(reco[:k], 1):
        if item in relevant:
            dcg += 1 / np.log2(i + 1)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0


# =============================================================================
# 2) Modèle Matrix Factorization (pour chargement)
# =============================================================================

class MatrixFactorization(nn.Module):
    """Modèle de factorisation matricielle avec biais."""
    
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, user_idx, item_idx):
        u = self.user_factors(user_idx)
        v = self.item_factors(item_idx)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        dot = (u * v).sum(dim=1)
        return dot + bu + bi + self.global_bias


# =============================================================================
# 3) Fonctions de recommandation
# =============================================================================

def recommend_for_user_content_based(user_id, ratings_df, id2idx, idx2id, item_matrix, k=10):
    """Recommandation Content-Based basée sur les embeddings BERT."""
    # Items déjà vus par l'utilisateur
    user_items = ratings_df[ratings_df["user_id"].astype(str) == str(user_id)]["item_id"].astype(str).tolist()
    
    if len(user_items) == 0:
        return []
    
    # Indices des items
    idxs = [id2idx[i] for i in user_items if i in id2idx]
    if len(idxs) == 0:
        return []
    
    # Profil utilisateur = moyenne des embeddings
    user_profile = item_matrix[idxs].mean(axis=0).reshape(1, -1)
    
    # Similarité cosine avec tous les items
    scores = cosine_similarity(user_profile, item_matrix).flatten()
    
    # Enlever les items déjà vus
    for i in idxs:
        scores[i] = -1
    
    # Top-k recommandations
    top_idx = np.argsort(scores)[::-1][:k]
    return [idx2id[i] for i in top_idx]


def score_mf(user_id, item_ids, model, user2idx, item2idx, device):
    """Calcule les scores MF pour un utilisateur et une liste d'items."""
    if user_id not in user2idx:
        return None
    
    uidx = user2idx[user_id]
    valid_items = [i for i in item_ids if i in item2idx]
    
    if len(valid_items) == 0:
        return None
    
    iidx = [item2idx[i] for i in valid_items]
    
    model.eval()
    with torch.no_grad():
        u = torch.tensor([uidx] * len(iidx), device=device)
        i = torch.tensor(iidx, device=device)
        scores = model(u, i).cpu().numpy()
    
    return valid_items, scores


# =============================================================================
# 4) Évaluation
# =============================================================================

def evaluate_content_based(users, train_items, test_items, id2idx, idx2id, item_matrix, df_train, K=10):
    """Évalue le modèle Content-Based."""
    results = []
    
    for u in users:
        relevant = test_items[u]
        if len(relevant) == 0:
            continue
        
        reco = recommend_for_user_content_based(u, df_train, id2idx, idx2id, item_matrix, K)
        reco = [str(i) for i in reco]
        
        results.append({
            "precision@10": precision_at_k(reco, relevant, K),
            "recall@10": recall_at_k(reco, relevant, K),
            "ndcg@10": ndcg_at_k(reco, relevant, K)
        })
    
    return pd.DataFrame(results)


def evaluate_matrix_factorization(df_test, df_train, model, user2idx, item2idx, device, K=10):
    """Évalue le modèle Matrix Factorization avec negative sampling."""
    random.seed(42)
    mf_results = []
    all_items = list(item2idx.keys())
    
    for _, row in df_test.iterrows():
        user = row["user_id"]
        pos_item = row["item_id"]
        
        if user not in user2idx or pos_item not in item2idx:
            continue
        
        # Items vus en train
        seen = set(df_train[df_train["user_id"] == user]["item_id"])
        seen.add(pos_item)
        
        # Négatifs
        negatives = list(set(all_items) - seen)
        if len(negatives) < 20:
            continue
        
        neg_samples = random.sample(negatives, 20)
        candidates = [pos_item] + neg_samples
        
        out = score_mf(user, candidates, model, user2idx, item2idx, device)
        if out is None:
            continue
        
        valid_items, scores = out
        ranked = [x for _, x in sorted(zip(scores, valid_items), reverse=True)]
        
        mf_results.append({
            "precision@10": precision_at_k(ranked, {pos_item}, K),
            "recall@10": recall_at_k(ranked, {pos_item}, K),
            "ndcg@10": ndcg_at_k(ranked, {pos_item}, K)
        })
    
    return pd.DataFrame(mf_results)


# =============================================================================
# 5) Visualisation
# =============================================================================

def plot_comparison(summary):
    """Affiche les graphiques de comparaison des modèles."""
    for metric in summary.columns:
        plt.figure(figsize=(8, 5))
        summary[metric].plot(kind="bar", color=['steelblue', 'coral'])
        plt.title(f"Comparaison des modèles - {metric}")
        plt.ylabel(metric)
        plt.xlabel("Modèle")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"comparison_{metric.replace('@', '_at_')}.png", dpi=150)
        plt.show()
        print(f"✓ Graphique sauvegardé : data/comparison_{metric.replace('@', '_at_')}.png")


def print_statistics(df_content, df_mf):
    """Affiche les statistiques descriptives détaillées."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DESCRIPTIVES")
    print("=" * 60)
    
    print("\n--- Content-Based ---")
    print(f"Nombre d'évaluations : {len(df_content)}")
    print(df_content.describe())
    
    print("\n--- Matrix Factorization ---")
    print(f"Nombre d'évaluations : {len(df_mf)}")
    print(df_mf.describe())
    
    # Comparaison
    summary = pd.DataFrame({
        "Precision@10": [
            df_mf["precision@10"].mean(),
            df_content["precision@10"].mean()
        ],
        "Recall@10": [
            df_mf["recall@10"].mean(),
            df_content["recall@10"].mean()
        ],
        "NDCG@10": [
            df_mf["ndcg@10"].mean(),
            df_content["ndcg@10"].mean()
        ]
    }, index=["Matrix Factorization", "Content-Based"])
    
    print("\n" + "=" * 60)
    print("COMPARAISON FINALE")
    print("=" * 60)
    print(summary)
    
    return summary


# =============================================================================
# 6) Main
# =============================================================================

def main():
    print("=" * 60)
    print("ÉVALUATION DU SYSTÈME DE RECOMMANDATION")
    print("=" * 60)
    
    # Chemins des fichiers
    EMBEDDINGS_PATH = "data/embeddings.pkl"
    MF_MODEL_PATH = "data/mf_model.pkl"
    K = 10
    
    # 1) Chargement des embeddings BERT
    print("\n[1/4] Chargement des embeddings BERT...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"ERREUR: Fichier {EMBEDDINGS_PATH} non trouvé.")
        print("Veuillez d'abord exécuter train.py")
        return
    
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"✓ {len(embeddings)} embeddings chargés")
    
    # Conversion en matrice
    item_ids = list(embeddings.keys())
    vectors = np.stack([embeddings[i] for i in item_ids])
    
    # Normalisation pour similarité cosine
    item_matrix = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    
    # Mappings
    id2idx = {iid: i for i, iid in enumerate(item_ids)}
    idx2id = {i: iid for i, iid in enumerate(item_ids)}
    
    print(f"  Dimension des embeddings : {vectors.shape[1]}")
    print(f"  Nombre d'items : {len(item_ids)}")
    
    # 2) Chargement du modèle MF
    print("\n[2/4] Chargement du modèle Matrix Factorization...")
    if not os.path.exists(MF_MODEL_PATH):
        print(f"ERREUR: Fichier {MF_MODEL_PATH} non trouvé.")
        print("Veuillez d'abord exécuter train.py")
        return
    
    with open(MF_MODEL_PATH, 'rb') as f:
        mf_data = pickle.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device utilisé : {device}")
    
    model = MatrixFactorization(mf_data["n_users"], mf_data["n_items"]).to(device)
    model.load_state_dict(mf_data["model_state_dict"])
    model.eval()
    
    user2idx = mf_data["user2idx"]
    item2idx = mf_data["item2idx"]
    df_train = mf_data["df_train"]
    df_test = mf_data["df_test"]
    
    print(f"  Nombre d'utilisateurs : {mf_data['n_users']}")
    print(f"  Nombre d'items : {mf_data['n_items']}")
    
    # 3) Préparation des données de test
    print("\n[3/4] Préparation des données de test...")
    df_train["user_id"] = df_train["user_id"].astype(str)
    df_train["item_id"] = df_train["item_id"].astype(str)
    df_test["user_id"] = df_test["user_id"].astype(str)
    df_test["item_id"] = df_test["item_id"].astype(str)
    
    train_items = df_train.groupby("user_id")["item_id"].apply(set)
    test_items = df_test.groupby("user_id")["item_id"].apply(set)
    
    users = list(set(train_items.index) & set(test_items.index))
    print(f"  Nombre d'utilisateurs évalués : {len(users)}")
    
    # 4) Évaluation
    print("\n[4/4] Évaluation des modèles...")
    
    print("\n--- Évaluation Content-Based ---")
    df_content = evaluate_content_based(users, train_items, test_items, id2idx, idx2id, item_matrix, df_train, K)
    print(f"Content-Based - Moyennes :")
    print(df_content.mean())
    
    print("\n--- Évaluation Matrix Factorization (avec negative sampling) ---")
    df_mf_results = evaluate_matrix_factorization(df_test, df_train, model, user2idx, item2idx, device, K)
    print(f"Matrix Factorization - Moyennes :")
    print(df_mf_results.mean())
    
    # Affichage des statistiques
    summary = print_statistics(df_content, df_mf_results)
    
    # Visualisation
    print("\n" + "=" * 60)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("=" * 60)
    plot_comparison(summary)
    
    # Sauvegarde des résultats
    results_path = "data/evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            "content_based": df_content,
            "matrix_factorization": df_mf_results,
            "summary": summary
        }, f)
    print(f"\n✓ Résultats sauvegardés dans {results_path}")
    
    print("\n" + "=" * 60)
    print("ÉVALUATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    main()
