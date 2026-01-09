#!/usr/bin/env python3
"""
Script d'entraînement pour le système de recommandation.
Charge les données, nettoie, génère les embeddings BERT,
entraîne le modèle Matrix Factorization et sauvegarde les embeddings.

Usage: python train.py
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# 1) Chargement des données
# =============================================================================

def load_json_lines(file_path):
    """Charge un fichier JSON ligne par ligne."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Erreur dans {file_path} :", e)
    return pd.DataFrame(data)


# =============================================================================
# 2) Nettoyage des textes
# =============================================================================

def clean_text(text):
    """Nettoie un texte : minuscules, suppression de ponctuation."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def preprocess_reviews(df):
    """Prétraite les colonnes review_text et review_summary."""
    df = df.copy()
    df["clean_review"] = df["review_text"].apply(clean_text)
    df["clean_summary"] = df["review_summary"].apply(clean_text)
    df["text"] = (df["clean_summary"] + " " + df["clean_review"]).str.strip()
    return df[df["text"] != ""]


# =============================================================================
# 3) Filtrage utilisateurs / items
# =============================================================================

def filter_data(df, min_reviews_user=3, min_reviews_item=1):
    """Filtre les utilisateurs et items selon un nombre minimum d'interactions."""
    # Filtrage utilisateurs
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_reviews_user].index
    df = df[df["user_id"].isin(valid_users)]
    
    # Filtrage items
    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_reviews_item].index
    df = df[df["item_id"].isin(valid_items)]
    
    return df


# =============================================================================
# 4) Génération des embeddings BERT
# =============================================================================

def generate_bert_embeddings(df, embeddings_path="embeddings.pkl"):
    """Génère les embeddings BERT pour chaque item."""
    
    if os.path.exists(embeddings_path):
        print(f"Chargement des embeddings depuis {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ {len(embeddings)} embeddings chargés!")
        return embeddings
    
    print("Calcul des embeddings BERT...")
    
    # Charger le tokenizer et le modèle BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    
    # Créer un mapping item_id → texte unique (concaténation des avis)
    item_texts = df.groupby("item_id")["text"].apply(lambda x: " ".join(x)).reset_index()
    
    def get_embedding(text):
        """Encode un texte en vecteur BERT (CLS token)."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().numpy()
    
    # Application sur tous les items
    embeddings = {}
    for _, row in tqdm(item_texts.iterrows(), total=len(item_texts), desc="Génération embeddings"):
        item_id = row["item_id"]
        text = row["text"]
        embeddings[item_id] = get_embedding(text)
    
    # Sauvegarder
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ {len(embeddings)} embeddings calculés et sauvegardés dans {embeddings_path}!")
    
    return embeddings


# =============================================================================
# 5) Modèle Matrix Factorization
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
        
        # Initialisation
        nn.init.normal_(self.user_factors.weight, 0, 0.1)
        nn.init.normal_(self.item_factors.weight, 0, 0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_idx, item_idx):
        u = self.user_factors(user_idx)
        v = self.item_factors(item_idx)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        dot = (u * v).sum(dim=1)
        return dot + bu + bi + self.global_bias


def train_matrix_factorization(df, n_factors=50, n_epochs=15, lr=0.01, weight_decay=1e-5):
    """Entraîne le modèle Matrix Factorization."""
    
    df_mf = df.copy()
    df_mf["rating"] = pd.to_numeric(df_mf["rating"], errors="coerce")
    df_mf = df_mf.dropna(subset=["rating"])
    
    print(f"Taille des données pour MF : {df_mf.shape}")
    
    # Encodage des IDs
    unique_users = df_mf["user_id"].unique()
    unique_items = df_mf["item_id"].unique()
    
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {i: j for j, i in enumerate(unique_items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}
    
    df_mf["user_idx"] = df_mf["user_id"].map(user2idx)
    df_mf["item_idx"] = df_mf["item_id"].map(item2idx)
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    print(f"Nombre d'utilisateurs : {n_users}")
    print(f"Nombre d'items        : {n_items}")
    
    # Train / Test split
    np.random.seed(42)
    indices = np.arange(len(df_mf))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(df_mf))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    df_train = df_mf.iloc[train_idx].reset_index(drop=True)
    df_test = df_mf.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train interactions : {len(df_train)}")
    print(f"Test  interactions : {len(df_test)}")
    
    # Tenseurs PyTorch
    user_train = torch.tensor(df_train["user_idx"].values, dtype=torch.long)
    item_train = torch.tensor(df_train["item_idx"].values, dtype=torch.long)
    rating_train = torch.tensor(df_train["rating"].values, dtype=torch.float32)
    
    user_test = torch.tensor(df_test["user_idx"].values, dtype=torch.long)
    item_test = torch.tensor(df_test["item_idx"].values, dtype=torch.long)
    rating_test = torch.tensor(df_test["rating"].values, dtype=torch.float32)
    
    train_dataset = TensorDataset(user_train, item_train, rating_train)
    test_dataset = TensorDataset(user_test, item_test, rating_test)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    # Entraînement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    
    model = MatrixFactorization(n_users, n_items, n_factors).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Évaluation RMSE sur test
        model.eval()
        with torch.no_grad():
            preds_list, true_list = [], []
            for u, i, r in test_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                preds = model(u, i)
                preds_list.append(preds.cpu().numpy())
                true_list.append(r.cpu().numpy())
        
        preds_all = np.concatenate(preds_list)
        true_all = np.concatenate(true_list)
        rmse = np.sqrt(np.mean((preds_all - true_all) ** 2))
        
        print(f"Epoch {epoch:02d} | Train MSE: {np.mean(train_losses):.4f} | Test RMSE: {rmse:.4f}")
    
    print("✅ Entraînement Matrix Factorization terminé.")
    
    return model, user2idx, item2idx, idx2user, idx2item, df_train, df_test, device


# =============================================================================
# 6) Main
# =============================================================================

def main():
    print("=" * 60)
    print("ENTRAÎNEMENT DU SYSTÈME DE RECOMMANDATION")
    print("=" * 60)
    
    # Chemins des fichiers
    DATA_PATH = "data/renttherunway_final_data.json"
    EMBEDDINGS_PATH = "data/embeddings.pkl"
    MF_MODEL_PATH = "data/mf_model.pkl"
    
    # 1) Chargement des données
    print("\n[1/5] Chargement des données...")
    if not os.path.exists(DATA_PATH):
        print(f"ERREUR: Fichier {DATA_PATH} non trouvé.")
        print("Veuillez placer le fichier dans le dossier courant.")
        return
    
    df_rent = load_json_lines(DATA_PATH)
    print(f"Données chargées : {df_rent.shape}")
    
    # 2) Prétraitement
    print("\n[2/5] Prétraitement des textes...")
    df_clean = preprocess_reviews(df_rent)
    print(f"Après nettoyage : {df_clean.shape}")
    
    # 3) Filtrage
    print("\n[3/5] Filtrage utilisateurs/items...")
    df_filtered = filter_data(df_clean, min_reviews_user=3, min_reviews_item=1)
    print(f"Après filtrage : {df_filtered.shape}")
    
    # 4) Embeddings BERT
    print("\n[4/5] Génération des embeddings BERT...")
    embeddings = generate_bert_embeddings(df_filtered, EMBEDDINGS_PATH)
    
    # Conversion en matrice numpy et sauvegarde
    item_ids = list(embeddings.keys())
    vectors = np.stack([embeddings[i] for i in item_ids])
    np.savez("data/item_embeddings.npz", item_ids=item_ids, vectors=vectors)
    print(f"✓ Embeddings sauvegardés dans item_embeddings.npz")
    
    # 5) Entraînement MF
    print("\n[5/5] Entraînement Matrix Factorization...")
    model, user2idx, item2idx, idx2user, idx2item, df_train, df_test, device = train_matrix_factorization(df_filtered)
    
    # Sauvegarde du modèle MF et mappings
    mf_data = {
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "df_train": df_train,
        "df_test": df_test
    }
    
    with open(MF_MODEL_PATH, 'wb') as f:
        pickle.dump(mf_data, f)
    print(f"✓ Modèle MF sauvegardé dans {MF_MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("Fichiers générés :")
    print(f"  - {EMBEDDINGS_PATH} (embeddings BERT)")
    print(f"  - item_embeddings.npz (embeddings en format numpy)")
    print(f"  - {MF_MODEL_PATH} (modèle Matrix Factorization)")
    print("=" * 60)


if __name__ == "__main__":
    main()
