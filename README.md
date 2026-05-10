# Détection de communautés sur graphes hétérogènes temporels
## Étude comparative : approches classiques vs hybrides quantiques

---

## Description

Ce projet propose un pipeline complet d'analyse d'un réseau académique dynamique construit à partir du dataset **DBLP**. Il combine :

- Construction d'un **graphe hétérogène temporel** (auteurs ↔ publications)
- Génération d'embeddings avec un modèle **THGNN**
- **Clustering classique** avec HDBSCAN
- **Transformation quantique** des embeddings (PennyLane)
- **Clustering hybride quantique** avec HDBSCAN
- **Comparaison** des deux approches via une interface Streamlit

---

## 🚀 Application déployée

L'interface est disponible en ligne via **Streamlit Cloud** :

👉 **[Accéder à l'application]([https://share.streamlit.io](https://thgnn-clustering-classique-vs-quantique-wrewyr6sah2qn9adoy2xv5.streamlit.app/))**

> Aucune installation requise pour consulter les résultats et visualisations.

---

## Structure du projet

    thgnn-clustering-classique-vs-quantique/
    │
    ├── data/                        ← instructions téléchargement Kaggle
    ├── approche_classique/          ← THGNN + train + HDBSCAN
    ├── approche_quantique/
    │   ├── research/                ← exploration des 16 méthodes d'encodage
    │   ├── benchmark/               ← comparaison + decision tree
    │   └── hybride/                 ← 5 méthodes retenues
    │       ├── 2024/                ← angle, variational, amplitude
    │       └── 2025/                ← angle, variational, amplitude, phase
    ├── app/                         ← interface Streamlit
    ├── results/                     ← métriques et figures
    ├── requirements.txt
    └── README.md

---

## Dataset

Le dataset est publié sur Kaggle et contient un graphe hétérogène temporel construit à partir de DBLP.

**Téléchargement :**

```bash
pip install kaggle
kaggle datasets download ayamhiri/temporal-graph-dataset
```

Ou télécharge manuellement depuis :
👉 [https://www.kaggle.com/datasets/ayamhiri/temporal-graph-dataset/data](https://www.kaggle.com/datasets/ayamhiri/temporal-graph-dataset/data)

Structure du dataset :

```
data/
├── selected_papers_full.jsonl
├── edges/
│   ├── edges_author_author.jsonl
│   ├── edges_author_paper.jsonl
│   └── edges_paper_paper.jsonl
├── embeddings/
│   └── paper_embeddings.npy
├── nodes/
│   ├── nodes_authors.jsonl
│   └── nodes_papers.jsonl
└── snapshots/
    ├── snapshot_2015.pt
    ├── snapshot_2016.pt
    ├── snapshot_2017.pt
    ├── snapshot_2018.pt
    ├── snapshot_2019.pt
    ├── snapshot_2020.pt
    ├── snapshot_2021.pt
    ├── snapshot_2022.pt
    ├── snapshot_2023.pt
    ├── snapshot_2024.pt
    └── snapshot_2025.pt
```

---

## Pipeline

    DBLP (Kaggle)
        ↓
    Graphe hétérogène temporel (auteurs ↔ articles, snapshots annuels)
        ↓
    THGNN → Embeddings d'auteurs
        ↓
    ┌─────────────────────┬──────────────────────────┐
    │  Approche classique │   Approche quantique      │
    │  HDBSCAN            │   Encodage → HDBSCAN      │
    └─────────────────────┴──────────────────────────┘
        ↓
     Comparaison : Silhouette Score, Davies-Bouldin Index, Bruit (noise ratio)
                + Interprétation des communautés générées
        ↓
     Interface Streamlit

---

## Méthodes d'encodage quantique explorées

16 méthodes explorées, 5 retenues :

| Statut | Méthode |
|--------|---------|
| ✅ Retenue | Angle Encoding |
| ✅ Retenue | Variational Encoding |
| ✅ Retenue | Amplitude Encoding |
| ✅ Retenue | Phase Encoding |
| 🔬 Explorée | Block Encoding |
| 🔬 Explorée | Density/Hybrid Encoding |
| 🔬 Explorée | Directional Encoding |
| 🔬 Explorée | Entangler Enhanced |
| 🔬 Explorée | Feature Map Encoding |
| 🔬 Explorée | QSample Encoding |
| 🔬 Explorée | Chebyshev Encoding |
| 🔬 Explorée | Fourier Encoding |
| 🔬 Explorée | Projected Unitary Encoding |
| 🔬 Explorée | Scaled Encoding |

---

## Lancer l'interface en local

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Technologies

| Composant | Outil |
|-----------|-------|
| Graphes hétérogènes | PyTorch Geometric |
| Modèle THGNN | PyTorch |
| Calcul quantique | PennyLane |
| Clustering | HDBSCAN, scikit-learn |
| Visualisation | Streamlit, Plotly, UMAP |
| Dataset | DBLP via Kaggle |
| Déploiement | Streamlit Cloud |

---

## Auteurs

Projet de Conception et Développement (PCD) — Élèves ingénieures, ENSI

- Aya MHIRI
- Sarra BOURAOUI
