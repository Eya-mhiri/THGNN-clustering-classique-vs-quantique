import streamlit as st
import os
import json
import pandas as pd
if not st.session_state.get("authentication_status"):
    st.error("⛔ Accès refusé. Connectez-vous depuis la page principale.")
    st.stop()
# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Administration", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    /* Cache la topbar */
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Descend le contenu sous la topbar */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }
    /* ── PAGE ── */
    .stApp { background-color: #fafafa; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }
    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background-color: #e8f0fe;
        border-right: 1px solid #c5d5f5;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #1a1a2e !important;
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: white;
        border: 1px solid #c5d5f5;
        border-radius: 6px;
        font-size: 1rem;
        padding: 0.5rem 0.7rem;
        color: #1a1a2e;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #1a1a2e !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem !important;
        width: 100%;
        padding: 0.6rem 1rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #4a90d9 !important;
    }

    /* ── TITRES ── */
    .page-title {
        font-size: 1.9rem;
        font-weight: 750;
        color: #111111;
        margin-bottom: 0.3rem;
    }
    .page-sub {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111111;
        border-left: 4px solid #4a90d9;
        padding-left: 0.8rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* ── BOUTONS PRINCIPAUX ── */
    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #4a90d9;
        color: white;
    }

    /* ── BOUTONS EXPORT ── */
    .stDownloadButton > button {
        background: #1a1a2e !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important;
        width: 100% !important;
    }
    .stDownloadButton > button:hover {
        background: #4a90d9 !important;
        color: white !important;
    }

    /* ── METRIC CARD ── */
    .metric-card {
        background: #f0f4ff;
        border: 1.5px solid #c5d5f5;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #4a5568;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111111;
    }

    /* ── TEXTE GÉNÉRAL ── */
    p, .stMarkdown p, label, .stCaption {
        font-size: 1rem !important;
    }
    .stDataFrame {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# VÉRIFICATION ACCÈS
# ============================================================
import yaml
from yaml.loader import SafeLoader
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    credentials = config["credentials"]
    cookie_name = config["cookie"]["name"]
    cookie_key = config["cookie"]["key"]
    cookie_expiry = config["cookie"]["expiry_days"]
else:
    credentials = {
        "usernames": {
            username: {
                "email": data["email"],
                "name": data["name"],
                "password": data["password"]
            }
            for username, data in st.secrets["credentials"]["usernames"].items()
        }
    }
    cookie_name = st.secrets["cookie"]["name"]
    cookie_key = st.secrets["cookie"]["key"]
    cookie_expiry = st.secrets["cookie"]["expiry_days"]

BASE    = st.session_state.get("base_path", "")
RESULTS = os.path.join(BASE, "results") if BASE else ""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Administration")
    st.divider()
    st.success("Mode Admin actif")
    st.caption(f"Chemin : {BASE if BASE else 'Non configuré'}")

# ============================================================
# TITRE
# ============================================================
st.markdown('<div class="page-title">Administration — Configuration système</div>',
            unsafe_allow_html=True)
st.markdown('<div class="page-sub">Gestion des chemins, audit des fichiers et hyperparamètres</div>',
            unsafe_allow_html=True)
st.divider()

# ============================================================
# SECTION 1 — CONFIGURATION
# ============================================================
st.markdown('<div class="section-header">Configuration des chemins</div>',
            unsafe_allow_html=True)

with st.expander("Chemins et modes", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mode d'interface**")
        app_mode = st.radio("Mode par défaut", ["Demo", "Real Data"])
        st.session_state["app_mode"] = app_mode

    with col2:
        st.markdown("**Chemins**")
        base_path = st.text_input(
            "Chemin racine",
            value=BASE,
            placeholder="C:/Users/.../thgnn-clustering-..."
        )
        if base_path != BASE:
            st.session_state["base_path"] = base_path
            BASE    = base_path
            RESULTS = os.path.join(BASE, "results")

        st.caption(f"Data    : {os.path.join(BASE, 'data') if BASE else 'Non configuré'}")
        st.caption(f"Results : {RESULTS if RESULTS else 'Non configuré'}")

# ============================================================
# SECTION 2 — AUDIT DES FICHIERS
# ============================================================
st.markdown('<div class="section-header">Audit des fichiers</div>',
            unsafe_allow_html=True)

files_to_check = [
    ("results/classique/clustering_2024.png",           "UMAP classique 2024"),
    ("results/classique/clustering_2025.png",           "UMAP classique 2025"),
    ("results/classique/comparaison_2024_2025.png",     "Comparaison 2024 vs 2025"),
    ("results/classique/clustering_summary.json",       "Métriques classiques"),
    ("results/classique/cluster_labels_2024.json",      "Labels classiques 2024"),
    ("results/classique/cluster_labels_2025.json",      "Labels classiques 2025"),
    ("results/quantique/2024/angle/qfig_angle_2024.png",             "UMAP Angle 2024"),
    ("results/quantique/2024/angle/qmetrics_angle_2024.json",        "Métriques Angle 2024"),
    ("results/quantique/2024/angle/quantum_summary.json",            "Summary Angle 2024"),
    ("results/quantique/2024/amplitude/qfig_amplitude_2024.png",     "UMAP Amplitude 2024"),
    ("results/quantique/2024/amplitude/qmetrics_amplitude_2024.json","Métriques Amplitude 2024"),
    ("results/quantique/2024/variational/qfig_variational_2024.png", "UMAP Variational 2024"),
    ("results/quantique/2024/variational/qmetrics_variational_2024.json","Métriques Variational 2024"),
    ("results/quantique/2025/angle/qfig_angle_2025.png",             "UMAP Angle 2025"),
    ("results/quantique/2025/angle/qmetrics_angle_2025.json",        "Métriques Angle 2025"),
    ("results/quantique/2025/amplitude/qfig_amplitude_2025.png",     "UMAP Amplitude 2025"),
    ("results/quantique/2025/amplitude/qmetrics_amplitude_2025.json","Métriques Amplitude 2025"),
    ("results/quantique/2025/variational/qfig_variational_2025.png", "UMAP Variational 2025"),
    ("results/quantique/2025/variational/qmetrics_variational_2025.json","Métriques Variational 2025"),
    ("results/quantique/2025/phase/qfig_phase_2025.png",             "UMAP Phase 2025"),
    ("results/quantique/2025/phase/qmetrics_phase_2025.json",        "Métriques Phase 2025"),
]

if st.button("Lancer l'audit complet"):
    if not BASE:
        st.warning("Configurez d'abord le chemin racine.")
    else:
        ok, ko = 0, 0
        cols = st.columns(3)
        for i, (rel_path, label) in enumerate(files_to_check):
            full = os.path.join(BASE, rel_path)
            with cols[i % 3]:
                if os.path.exists(full):
                    st.success(f"✅ {label}")
                    ok += 1
                else:
                    st.error(f"❌ {label}")
                    ko += 1
        st.divider()
        if ko == 0:
            st.success(f"Audit terminé — {ok}/{ok+ko} fichiers présents. Prêt pour déploiement.")
        else:
            st.warning(f"Audit terminé — {ok}/{ok+ko} fichiers présents. {ko} fichier(s) manquant(s).")

# ============================================================
# SECTION 3 — UPLOAD
# ============================================================
st.markdown('<div class="section-header">Chargement de fichiers</div>',
            unsafe_allow_html=True)

with st.expander("Uploader un fichier résultat"):
    dest_folder = st.selectbox("Dossier de destination", [
        "results/classique",
        "results/quantique/2024/angle",
        "results/quantique/2024/amplitude",
        "results/quantique/2024/variational",
        "results/quantique/2025/angle",
        "results/quantique/2025/amplitude",
        "results/quantique/2025/variational",
        "results/quantique/2025/phase",
        "data/embeddings",
    ])
    uploaded = st.file_uploader(
        "Fichier", type=["png", "json", "csv", "pt", "npz"]
    )
    if uploaded and BASE:
        target_dir  = os.path.join(BASE, dest_folder)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, uploaded.name)
        with open(target_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"'{uploaded.name}' chargé dans {dest_folder}")

# ============================================================
# SECTION 4 — HYPERPARAMÈTRES
# ============================================================
st.markdown('<div class="section-header">Hyperparamètres d\'affichage</div>',
            unsafe_allow_html=True)

col_h1, col_h2 = st.columns(2)
with col_h1:
    n_authors = st.number_input(
        "Nombre d'auteurs à analyser",
        value=st.session_state.get("n_authors", 1000),
        step=100, min_value=100, max_value=10000
    )
    st.session_state["n_authors"] = n_authors

    pub_threshold = st.slider(
        "Seuil publications (analyse longitudinale)",
        1, 50, st.session_state.get("pub_threshold", 5)
    )
    st.session_state["pub_threshold"] = pub_threshold

with col_h2:
    st.markdown("**Méthodes disponibles**")
    st.caption("2024 : Angle, Amplitude, Variational")
    st.caption("2025 : Angle, Amplitude, Variational, Phase")
    st.markdown("**Dataset**")
    st.caption("DBLP — Kaggle")
    st.caption("Snapshots : 2015 → 2025")

if st.button("Sauvegarder la configuration"):
    st.success("Configuration sauvegardée en session.")

# ============================================================
# SECTION 5 — DÉPLOIEMENT
# ============================================================
st.markdown('<div class="section-header">Déploiement</div>',
            unsafe_allow_html=True)

with st.expander("Instructions de déploiement"):
    st.markdown("**Streamlit Cloud — secrets.toml**")
    st.code("ADMIN_PASSWORD = \"votre_mot_de_passe\"", language="toml")

    st.markdown("**Lancement local**")
    st.code("streamlit run app.py", language="bash")

    st.markdown("**Tunnel local (test)**")
    st.code("ssh -R 80:localhost:8501 serveo.net", language="bash")

st.divider()
st.markdown(
    "<div style='text-align:center; color:#718096; font-size:0.88rem;'>"
    "Projet de Conception et Développement (PCD) — ENSI — "
    "Aya MHIRI & Sarra BOURAOUI"
    "</div>",
    unsafe_allow_html=True
)
