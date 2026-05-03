import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

st.set_page_config(
    page_title="THGNN Quantum Clustering",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .stApp { background-color: #fafafa; }
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }

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
        margin-top: 0.3rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #4a90d9 !important;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 750;
        color: #111111;
        line-height: 1.25;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111111;
        border-left: 4px solid #4a90d9;
        padding-left: 0.8rem;
        margin: 1.8rem 0 1rem 0;
    }
    .badge {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.95rem;
        margin: 3px;
        font-weight: 500;
    }
    .nav-card {
        background: #f0f4ff;
        border: 1.5px solid #c5d5f5;
        border-radius: 12px;
        padding: 2rem 1.8rem;
        text-align: center;
        height: 100%;
    }
    .nav-card h3 {
        color: #111111;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .nav-card p {
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
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
    }
    .desc-text {
        font-size: 1.05rem;
        color: #2d3748;
        line-height: 1.8;
    }
    .footer-note {
        font-size: 0.95rem;
        color: #718096;
        text-align: center;
    }
    .auth-box {
        background: #f0f4ff;
        border: 1.5px solid #c5d5f5;
        border-radius: 12px;
        padding: 2rem;
        max-width: 400px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "base_path": "",
    "n_authors": 1000,
    "pub_threshold": 5
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# AUTHENTIFICATION
# ============================================================
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### THGNN — Quantum Clustering")
    st.markdown("Détection de communautés sur graphes hétérogènes temporels — DBLP")
    st.divider()

    st.markdown("**Mode d'interface**")
    mode = st.radio("", ["Visiteur", "Administrateur"], label_visibility="collapsed")

    if mode == "Administrateur":
        st.markdown("**Authentification**")
        authenticator.login(
            location="sidebar",
            fields={
                "Form name": "Connexion",
                "Username": "Nom d'utilisateur",
                "Password": "Mot de passe",
                "Login": "Se connecter"
            }
        )

        name = st.session_state.get("name")
        auth_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")

        if auth_status is False:
            st.error("Identifiants incorrects.")
            st.session_state["is_admin"] = False
        elif auth_status is None:
            st.session_state["is_admin"] = False
        elif auth_status:
            st.success(f"Connecté — {name}")
            st.session_state["is_admin"] = True
            authenticator.logout("Se déconnecter", location="sidebar")
    else:
        st.session_state["is_admin"] = False
        if "is_admin" not in st.session_state:
            st.session_state["is_admin"] = False

    if st.session_state.get("is_admin"):
        st.divider()
        st.markdown("**Configuration**")
        base_path = st.text_input(
            "Chemin racine du projet",
            value=st.session_state.get("base_path", ""),
            placeholder="C:/Users/.../thgnn-clustering-..."
        )
        st.session_state["base_path"] = base_path

        n_authors = st.slider(
            "Nombre d'auteurs à analyser",
            100, 5000, st.session_state["n_authors"], step=100
        )
        st.session_state["n_authors"] = n_authors

        pub_threshold = st.slider(
            "Seuil publications",
            1, 50, st.session_state["pub_threshold"]
        )
        st.session_state["pub_threshold"] = pub_threshold

    st.divider()
    st.markdown(
        "<div style='font-size:0.95rem; color:#3d5a99;'>"
        "Projet de Conception et Développement<br>"
        "<b>ENSI</b> — Aya MHIRI & Sarra BOURAOUI"
        "</div>",
        unsafe_allow_html=True
    )

# ============================================================
# HERO
# ============================================================
st.markdown(
    '<div class="hero-title">Détection de communautés sur graphes hétérogènes temporels</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-sub">Étude comparative : approches classiques et hybrides quantiques — Dataset DBLP (2015–2025)</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div>
<span class="badge">PyTorch Geometric</span>
<span class="badge">PennyLane</span>
<span class="badge">HDBSCAN</span>
<span class="badge">UMAP</span>
<span class="badge">THGNN</span>
<span class="badge">DBLP</span>
<span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# DESCRIPTION + PIPELINE
# ============================================================
col_desc, col_pipeline = st.columns([1, 1], gap="large")

with col_desc:
    st.markdown('<div class="section-header">À propos du projet</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="desc-text">
    Ce projet propose un pipeline complet d'analyse d'un <b>réseau académique dynamique</b>
    construit à partir du dataset <b>DBLP</b> (2015–2025).<br><br>
    Les auteurs scientifiques sont représentés comme des nœuds dans un <b>graphe hétérogène
    temporel</b> reliant publications et collaborations. Un modèle <b>THGNN</b> génère des
    embeddings vectoriels capturant l'évolution temporelle de chaque auteur.<br><br>
    Ces embeddings sont clustérisés via <b>HDBSCAN</b> selon deux approches :
    une approche <b>classique</b> directe, et une approche <b>hybride quantique</b>
    où les embeddings sont transformés par un circuit variationnel <b>PennyLane</b>
    avant le clustering.<br><br>
    La comparaison s'appuie sur les métriques <b>Silhouette</b>,
    <b>Davies-Bouldin</b>, <b>taux de bruit</b> et l'<b>interprétation
    sémantique</b> des communautés générées.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Technologies</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="desc-text">
        • PyTorch Geometric<br>
        • PennyLane<br>
        • HDBSCAN
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="desc-text">
        • UMAP<br>
        • Streamlit<br>
        • DBLP via Kaggle
        </div>
        """, unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-header">Pipeline du projet</div>', unsafe_allow_html=True)

    @st.cache_data
    def generate_pipeline_figure():
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        boxes = [
            (5.0, 9.2, "DBLP Dataset\n(Kaggle)", "#dbeafe", "#3b82f6"),
            (5.0, 7.6, "Graphe hétérogène temporel\nAuteurs ↔ Articles (2015–2025)", "#ede9fe", "#7c3aed"),
            (5.0, 6.0, "Modèle THGNN\nEmbeddings d'auteurs", "#d1fae5", "#059669"),
            (2.5, 4.2, "Clustering classique\nHDBSCAN", "#fef3c7", "#d97706"),
            (7.5, 4.2, "Encodage quantique\nPennyLane → HDBSCAN", "#fce7f3", "#db2777"),
            (5.0, 2.3, "Comparaison des métriques\nSilhouette · Davies-Bouldin · Bruit", "#fee2e2", "#dc2626"),
            (5.0, 0.7, "Interface Streamlit\nVisualisation & Export", "#e0f2fe", "#0284c7"),
        ]

        box_w, box_h = 3.6, 0.85

        for (x, y, label, fc, ec) in boxes:
            fancy = mpatches.FancyBboxPatch(
                (x - box_w/2, y - box_h/2),
                box_w, box_h,
                boxstyle="round,pad=0.08",
                facecolor=fc, edgecolor=ec,
                linewidth=1.8, zorder=3
            )
            ax.add_patch(fancy)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="#1a1a2e", zorder=4, linespacing=1.4)

        arrow_props = dict(arrowstyle="-|>", color="#64748b",
                           lw=1.5, mutation_scale=14)

        ax.annotate("", xy=(5.0, 7.6+box_h/2),
                    xytext=(5.0, 9.2-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(5.0, 6.0+box_h/2),
                    xytext=(5.0, 7.6-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(2.5, 4.2+box_h/2),
                    xytext=(4.2, 6.0-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(7.5, 4.2+box_h/2),
                    xytext=(5.8, 6.0-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(4.2, 2.3+box_h/2),
                    xytext=(2.5, 4.2-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(5.8, 2.3+box_h/2),
                    xytext=(7.5, 4.2-box_h/2), arrowprops=arrow_props, zorder=2)
        ax.annotate("", xy=(5.0, 0.7+box_h/2),
                    xytext=(5.0, 2.3-box_h/2), arrowprops=arrow_props, zorder=2)

        ax.text(1.0, 4.2, "Classique", fontsize=8, color="#d97706",
                fontstyle="italic", va="center")
        ax.text(9.0, 4.2, "Quantique", fontsize=8, color="#db2777",
                fontstyle="italic", va="center", ha="right")

        plt.tight_layout(pad=0.5)
        return fig

    st.pyplot(generate_pipeline_figure(), use_container_width=True)

st.divider()

# ============================================================
# BOUTONS DE NAVIGATION
# ============================================================
st.markdown('<div class="section-header">Explorer l\'interface</div>', unsafe_allow_html=True)

nav1, nav2 = st.columns(2, gap="large")

with nav1:
    st.markdown("""
    <div class="nav-card">
        <h3>Benchmark des encodages quantiques</h3>
        <p>Comparez les <b>15 méthodes d'encodage quantique</b> sur un dataset
        de démonstration. Visualisez les performances en accuracy,
        temps d'exécution et consommation mémoire.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Lancer le Benchmark Demo", key="btn_benchmark"):
        st.switch_page("pages/1_benchmark.py")

with nav2:
    st.markdown("""
    <div class="nav-card">
        <h3>Analyse des données réelles DBLP</h3>
        <p>Explorez les résultats de clustering sur le dataset DBLP complet.
        Comparez les approches <b>classique et quantique</b>,
        analysez les communautés et exportez les résultats.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Explorer les données DBLP", key="btn_real"):
        st.switch_page("pages/2_real_data.py")

st.divider()

# ============================================================
# SECTION ADMIN
# ============================================================
if st.session_state.get("is_admin"):
    st.markdown('<div class="section-header">Administration — Vérification des fichiers</div>',
                unsafe_allow_html=True)
    base = st.session_state.get("base_path", "")
    if base:
        files_to_check = [
            ("results/classique/clustering_2024.png",           "UMAP classique 2024"),
            ("results/classique/clustering_2025.png",           "UMAP classique 2025"),
            ("results/classique/clustering_summary.json",       "Métriques classiques"),
            ("results/classique/cluster_labels_2024.json",      "Labels classiques 2024"),
            ("results/classique/cluster_labels_2025.json",      "Labels classiques 2025"),
            ("results/quantique/2024/amplitude/qfig_amplitude_2024.png",     "UMAP Amplitude 2024"),
            ("results/quantique/2024/angle/qfig_angle_2024.png",             "UMAP Angle 2024"),
            ("results/quantique/2024/variational/qfig_variational_2024.png", "UMAP Variational 2024"),
            ("results/quantique/2025/amplitude/qfig_amplitude_2025.png",     "UMAP Amplitude 2025"),
            ("results/quantique/2025/angle/qfig_angle_2025.png",             "UMAP Angle 2025"),
            ("results/quantique/2025/phase/qfig_phase_2025.png",             "UMAP Phase 2025"),
            ("results/quantique/2025/variational/qfig_variational_2025.png", "UMAP Variational 2025"),
        ]
        cols = st.columns(3)
        for i, (f, label) in enumerate(files_to_check):
            full = os.path.join(base, f)
            with cols[i % 3]:
                if os.path.exists(full):
                    st.success(f"✅ {label}")
                else:
                    st.error(f"❌ {label}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Recharger la vérification"):
                st.rerun()
        with c2:
            uploaded = st.file_uploader(
                "Charger un fichier résultat",
                type=["png", "json", "csv", "pt", "npz"]
            )
            if uploaded:
                target = os.path.join(base, "results", uploaded.name)
                with open(target, "wb") as f_out:
                    f_out.write(uploaded.getbuffer())
                st.success(f"Fichier {uploaded.name} chargé.")
    else:
        st.warning("Configurez le chemin racine dans la sidebar.")

st.divider()
st.markdown(
    "<div class='footer-note'>"
    "Projet de Conception et Développement (PCD) — ENSI — "
    "Aya MHIRI & Sarra BOURAOUI"
    "</div>",
    unsafe_allow_html=True
)