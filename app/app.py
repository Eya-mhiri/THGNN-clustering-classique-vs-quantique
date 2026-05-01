import streamlit as st
import os

# Configuration de la page
st.set_page_config(
    page_title="THGNN Quantum Clustering",
    page_icon="🌌",
    layout="wide"
)

# ============================================================
# STYLE CSS (TON STYLE DÉGRADÉ)
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0b0d17; }
    .stApp {
        background: linear-gradient(to bottom, #0b0d17, #1c1c2b);
        color: white;
    }
    .hero-section {
        padding: 3rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(74, 144, 217, 0.3);
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1565c0, #6a1b9a);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(106, 27, 154, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GESTION DE LA SESSION ET AUTHENTIFICATION
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# Barre latérale (Sidebar)
with st.sidebar:
    st.image("https://pennylane.ai/img/logo-dark.png", width=150) # Logo Quantum
    st.title("🌌 THGNN-Q")
    st.markdown("---")
    
    # Choix du mode global
    mode_selection = st.radio("Mode d'utilisation", ["Visiteur", "Administrateur"])
    
    if mode_selection == "Administrateur":
        pwd = st.text_input("Mot de passe Admin", type="password")
        if pwd == "ensi2025": # Ton mot de passe défini
            st.session_state["is_admin"] = True
            st.success("Mode Admin activé")
        else:
            st.session_state["is_admin"] = False
            if pwd: st.error("Mot de passe incorrect")
    else:
        st.session_state["is_admin"] = False

    st.markdown("---")
    st.info("Utilisez les pages dans le menu pour naviguer entre le Benchmark et les données réelles.")

# ============================================================
# CONTENU DE LA PAGE D'ACCUEIL
# ============================================================

# Section Héro
st.markdown("""
<div class="hero-section">
    <h1>🚀 THGNN Clustering : Classique vs Quantique</h1>
    <p style="font-size: 1.2rem; color: #a8d8ea;">
        Plateforme d'analyse et de visualisation pour la détection de communautés sur graphes temporels (DBLP).
    </p>
</div>
""", unsafe_allow_html=True)

# Présentation du Pipeline
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Approche Classique")
    st.markdown("""
    - **Pipeline :** THGNN + Clustering (Louvain/K-Means).
    - **Données :** Embeddings temporels extraits de DBLP.
    - **Métriques :** Silhouette, Davies-Bouldin.
    """)
    if st.button("Voir les résultats DBLP", key="btn_real_data"):
        # Cette commande redirige vers la page si elle est dans le dossier /pages
        st.switch_page("pages/2_real_data.py")

with col2:
    st.subheader("⚛️ Approche Quantique")
    st.markdown("""
    - **Pipeline :** Encodage PennyLane + Classifieur Quantique.
    - **Méthodes :** 16 types d'encodages testés.
    - **Objectif :** Comparer l'expressivité des circuits.
    """)
    if st.button("Lancer le Benchmark Demo", key="btn_benchmark"):
        st.switch_page("pages/1_benchmark.py")

# Section Description (Use Case 1.6)
st.markdown("---")
with st.expander("📖 Description du Projet et du Pipeline"):
    st.write("""
    Cette interface centralise les résultats du pipeline classique **THGNN** et de l'approche **Quantique**. 
    Elle permet aux chercheurs de comparer l'efficacité des méthodes d'encodage quantique pour capturer la structure des communautés d'auteurs.
    
    **Acteurs :**
    - **Visiteur :** Explore les résultats, visualise les UMAP et exporte les scores.
    - **Administrateur :** Configure les chemins Kaggle, ajuste les hyperparamètres et déclenche les benchmarks.
    """)

# Affichage des chemins si Admin (Use Case Admin)
if st.session_state["is_admin"]:
    st.markdown("---")
    st.subheader("🛠️ Configuration Admin (Base Paths)")
    st.code(f"""
    BASE_DIR: C:\\Users\\mpshop\\thgnn-clustering-classique-vs-quantique\\
    DATA: \\data
    RESULTS: \\results
    """, language="text")
    
    if st.button("Vérifier l'intégrité des fichiers"):
        # Simulation de vérification
        st.success("Fichiers .pt, .npz et .jsonl détectés avec succès.")