import streamlit as st
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import time
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from datasets import load_dataset

st.set_page_config(page_title="Benchmark Quantique", layout="wide")

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

st.title("Benchmark des méthodes d'encodage quantique")
st.markdown("Comparaison des 15 méthodes d'encodage — Dataset : Pima Indians Diabetes")
st.divider()

# ============================================================
# DÉFINITIONS
# ============================================================
METHOD_DEFINITIONS = {
    "Raw Data (Baseline)": {
        "description": "Référence classique sans encodage quantique. Les données normalisées dans [0,1] par min-max scaling sont directement transmises au classifieur.",
        "formule": "X_out = X_norm",
        "qubits": "0 qubit — pipeline classique",
        "avantage": "Temps minimal, référence absolue.",
        "limite": "Aucun enrichissement des features.",
        "couleur": "#6c757d"
    },
    "Angle Encoding": {
        "description": "Chaque feature xᵢ est encodée en angle de rotation via RX et RY sur un qubit dédié.",
        "formule": "RX(xᵢπ) · RY(xᵢπ/2) |0⟩ → Re(|ψ⟩) ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 1 par feature",
        "avantage": "Intuitif, continu, préserve la géométrie.",
        "limite": "Pas d'intrication entre features.",
        "couleur": "#4a90d9"
    },
    "Phase Encoding": {
        "description": "Encode les features dans la phase complexe via Hadamard + RZ.",
        "formule": "|0⟩ → H → RZ(xᵢπ) → (|0⟩ + e^{ixᵢπ}|1⟩)/√2",
        "qubits": "8 qubits — 1 par feature",
        "avantage": "Exploitable via l'interférence quantique.",
        "limite": "Phase indétectable sans circuit d'interférence.",
        "couleur": "#7b2ff7"
    },
    "Amplitude Encoding": {
        "description": "Encode N features dans les amplitudes d'un état à log₂(N) qubits. Compression exponentielle.",
        "formule": "|ψ⟩ = Σᵢ xᵢ|i⟩ avec ‖x‖₂=1 → 3 qubits pour 8 features",
        "qubits": "3 qubits — compression log₂(N)",
        "avantage": "Compression exponentielle.",
        "limite": "Normalisation L2 obligatoire.",
        "couleur": "#28a745"
    },
    "Density / Hybrid Encoding": {
        "description": "Représente l'état via la matrice densité ρ = |ψ⟩⟨ψ|. Capture les corrélations croisées.",
        "formule": "ρ = x_norm·x_normᵀ ∈ ℝ⁸ˣ⁸ → flatten → ℝ⁶⁴",
        "qubits": "8 qubits (conceptuel)",
        "avantage": "Capture les cohérences entre features.",
        "limite": "Dimension N² croît quadratiquement.",
        "couleur": "#fd7e14"
    },
    "Feature Map Encoding": {
        "description": "Circuit IQP après PCA à 3 composantes. Analogue aux noyaux kernel SVM.",
        "formule": "H⊗³ → RZ(πxᵢ) → CZ → RY(πxᵢ)",
        "qubits": "3 qubits (après PCA)",
        "avantage": "Kernel quantique puissant.",
        "limite": "PCA entraîne une perte d'information.",
        "couleur": "#ffc107"
    },
    "Variational Encoding": {
        "description": "Data re-uploading avec paramètres entraînables θ. Approximateur universel quantique.",
        "formule": "[RY(xᵢπ)·RZ(θᵢ) + CNOT] × 2 couches → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 16 paramètres",
        "avantage": "Expressivité maximale.",
        "limite": "Paramètres non optimisés dans ce benchmark.",
        "couleur": "#dc3545"
    },
    "Block Encoding": {
        "description": "Encode la matrice diagonale dans un bloc unitaire via qubits ancilla.",
        "formule": "H⊗³(ancilla) → ctrl-RY(xᵢ) → ⟨Z_ancilla⟩",
        "qubits": "11 qubits — 3 ancilla + 8 data",
        "avantage": "Base des algorithmes HHL.",
        "limite": "Sortie scalaire — faible expressivité.",
        "couleur": "#6c757d"
    },
    "Directional Encoding": {
        "description": "Exploite les 3 degrés de liberté de la sphère de Bloch via RX·RY·RZ.",
        "formule": "RX(xᵢ)·RY(xᵢ)·RZ(xᵢ)|0⟩ + CNOT chaîne → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 3 rotations + CNOT",
        "avantage": "Exploite les 3 axes de Bloch.",
        "limite": "Redondance des 3 rotations.",
        "couleur": "#20c997"
    },
    "Entangler Enhanced": {
        "description": "Évolution temporelle : H⊗⁸ → RZ → CNOT. Distribue l'information via l'intrication.",
        "formule": "H⊗⁸ → RZ(xᵢ) → CNOT chaîne → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — intrication globale",
        "avantage": "Représentation collective des features.",
        "limite": "Information individuelle diluée.",
        "couleur": "#6f42c1"
    },
    "QSample Encoding": {
        "description": "Encode les features comme probabilités de mesure P(|1⟩)=xᵢ via arcsin.",
        "formule": "RY(2·arcsin(√xᵢ))|0⟩ → ⟨Zᵢ⟩ = 1-2xᵢ ∈ [-1,1]⁸",
        "qubits": "8 qubits — encodage probabiliste",
        "avantage": "Sortie dans [-1,1] interprétable.",
        "limite": "Transformation monotone décroissante.",
        "couleur": "#a29bfe"
    },
    "Chebyshev Encoding": {
        "description": "Polynômes de Chebyshev Tₙ(x)=cos(n·arccos(x)) comme base orthogonale.",
        "formule": "θᵢ=arccos(2xᵢ-1) → RY(θᵢ)·RY(2θᵢ)|0⟩",
        "qubits": "8 qubits — T₁ et T₂",
        "avantage": "Base optimale pour l'approximation.",
        "limite": "Limité aux degrés 1 et 2.",
        "couleur": "#55efc4"
    },
    "Fourier Encoding": {
        "description": "Deux harmoniques de Fourier par qubit. Capture des patterns périodiques.",
        "formule": "H → RZ(xᵢπ) → RZ(2xᵢπ) |0⟩ → probs ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 2 harmoniques",
        "avantage": "Capture les patterns périodiques.",
        "limite": "Limité à 2 harmoniques.",
        "couleur": "#fd79a8"
    },
    "Projected Unitary Encoding": {
        "description": "Double projection sur observables X et Z après AngleEmbedding.",
        "formule": "AngleEmb(xᵢπ) → [⟨Xᵢ⟩, ⟨Zᵢ⟩] pour i=1..8 → ℝ¹⁶",
        "qubits": "8 qubits — observables X et Z",
        "avantage": "Double projection complémentaire.",
        "limite": "⟨X⟩²+⟨Z⟩²=1 — features non indépendantes.",
        "couleur": "#74b9ff"
    },
    "Scaled Encoding": {
        "description": "Angle Encoding sur cycle complet 2π. Adapté aux données périodiques.",
        "formule": "RY(xᵢ·2π)|0⟩ → probs ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — cycle [0, 2π]",
        "avantage": "Exploite le cycle complet de Bloch.",
        "limite": "Ambiguïté aux bornes (xᵢ=0 et xᵢ=1 identiques).",
        "couleur": "#e84393"
    },
}

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def load_data():
    ds = load_dataset("Genius-Society/Pima")
    df = pd.DataFrame(ds['train'])
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[feature_cols].values.astype(float)
    y = df['Outcome'].values
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    return X_norm, y

X_all, y_all = load_data()
N_TOTAL = len(X_all)

dev8  = qml.device("default.qubit", wires=8)
dev3  = qml.device("default.qubit", wires=3)

# ============================================================
# ENCODAGES
# ============================================================
def run_encoding(X, method):
    if method == "Raw Data (Baseline)":
        return X
    elif method == "Angle Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.RX(x[i] * np.pi, wires=i)
                qml.RY(x[i] * np.pi / 2, wires=i)
            return qml.state()
        return np.array([np.real(circ(r)) for r in X])
    elif method == "Phase Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.Hadamard(wires=i)
                qml.RZ(x[i] * np.pi, wires=i)
            return qml.probs(wires=range(8))
        return np.array([circ(r) for r in X])
    elif method == "Amplitude Encoding":
        @qml.qnode(dev3)
        def circ(x):
            qml.AmplitudeEmbedding(x, wires=range(3), normalize=True, pad_with=0.0)
            return qml.state()
        return np.real(np.array([circ(r) for r in X]))
    elif method == "Density / Hybrid Encoding":
        def density(x):
            xn = x / (np.linalg.norm(x) + 1e-9)
            return np.outer(xn, xn).flatten()
        return np.array([density(r) for r in X])
    elif method == "Feature Map Encoding":
        Xr = PCA(n_components=3).fit_transform(X)
        @qml.qnode(dev3)
        def circ(x):
            for i in range(3): qml.Hadamard(wires=i)
            for i in range(3): qml.RZ(np.pi * float(x[i]), wires=i)
            for i in range(2): qml.CZ(wires=[i, i+1])
            for i in range(3): qml.RY(np.pi * float(x[i]), wires=i)
            return qml.state()
        return np.real(np.array([circ(r) for r in Xr]))
    elif method == "Variational Encoding":
        np.random.seed(42)
        theta = np.random.rand(16)
        @qml.qnode(dev8)
        def circ(x, th):
            for i in range(8):
                qml.RY(x[i] * np.pi, wires=i)
                qml.RZ(th[i], wires=i)
            for i in range(7): qml.CNOT(wires=[i, i+1])
            for i in range(8):
                qml.RY(x[i] * np.pi, wires=i)
                qml.RZ(th[i+8], wires=i)
            return qml.state()
        return np.abs(np.array([circ(r, theta) for r in X]))**2
    elif method == "Block Encoding":
        dev_b = qml.device("default.qubit", wires=11)
        @qml.qnode(dev_b)
        def circ(x):
            xn = x / (np.linalg.norm(x) + 1e-9)
            for i in range(3): qml.Hadamard(wires=i)
            for i in range(8): qml.ctrl(qml.RY, control=range(3))(xn[i], wires=3+i)
            return qml.expval(qml.PauliZ(wires=3))
        return np.array([[circ(r)] for r in X])
    elif method == "Directional Encoding":
        @qml.qnode(dev8)
        def circ(x):
            xn = x / (np.max(x) + 1e-9)
            for i in range(8):
                qml.RX(xn[i], wires=i)
                qml.RY(xn[i], wires=i)
                qml.RZ(xn[i], wires=i)
            for i in range(7): qml.CNOT(wires=[i, i+1])
            return qml.state()
        return np.abs(np.array([circ(r) for r in X]))**2
    elif method == "Entangler Enhanced":
        @qml.qnode(dev8)
        def circ(x):
            xn = x / (np.max(x) + 1e-9)
            for i in range(8): qml.Hadamard(wires=i)
            for i in range(8): qml.RZ(xn[i], wires=i)
            for i in range(7): qml.CNOT(wires=[i, i+1])
            return qml.state()
        return np.abs(np.array([circ(r) for r in X]))**2
    elif method == "QSample Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.RY(2*np.arcsin(np.sqrt(np.clip(x[i], 0, 1))), wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        return np.array([circ(r) for r in X])
    elif method == "Chebyshev Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                angle = np.arccos(np.clip(2*x[i]-1, -1, 1))
                qml.RY(angle, wires=i)
                qml.RY(2*angle, wires=i)
            return qml.probs(wires=range(8))
        return np.array([circ(r) for r in X])
    elif method == "Fourier Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.Hadamard(wires=i)
                qml.RZ(x[i] * np.pi, wires=i)
                qml.RZ(2 * x[i] * np.pi, wires=i)
            return qml.probs(wires=range(8))
        return np.array([circ(r) for r in X])
    elif method == "Projected Unitary Encoding":
        @qml.qnode(dev8)
        def circ(x):
            qml.AngleEmbedding(x * np.pi, wires=range(8))
            return ([qml.expval(qml.PauliX(i)) for i in range(8)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(8)])
        return np.array([circ(r) for r in X])
    elif method == "Scaled Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8): qml.RY(x[i] * 2 * np.pi, wires=i)
            return qml.probs(wires=range(8))
        return np.array([circ(r) for r in X])
    return X

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Configuration")
    all_methods = list(METHOD_DEFINITIONS.keys())
    selected = st.multiselect(
        "Méthodes à benchmarker",
        all_methods,
        default=["Raw Data (Baseline)", "Angle Encoding", "Amplitude Encoding"]
    )
    limit = st.slider("Nombre d'échantillons", 50, N_TOTAL, 200, step=50,
                      help=f"Dataset complet = {N_TOTAL} échantillons")
    st.divider()
    st.caption(f"Dataset : Pima Indians Diabetes")
    st.caption(f"Classifieur : Decision Tree (depth=5)")
    st.caption(f"Split : 80/20")

# ============================================================
# SECTION 1 — DÉFINITION INTERACTIVE
# ============================================================
st.markdown('<div class="section-header">Section 1 — Explorer une méthode d\'encodage</div>',
            unsafe_allow_html=True)

method_preview = st.selectbox("Sélectionnez une méthode :", all_methods)
info = METHOD_DEFINITIONS[method_preview]

st.markdown(f"""
<div class="method-card">
<h4>{method_preview}</h4>
<p>{info['description']}</p>
<hr style="border-color:#dee2e6; margin:0.8rem 0"/>
<p><b>Formule :</b> <code>{info['formule']}</code></p>
<p><b>Qubits :</b> <span class="tag">{info['qubits']}</span></p>
<p><b>Avantage :</b> {info['avantage']}</p>
<p><b>Limite :</b> {info['limite']}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# SECTION 2 — BENCHMARK
# ============================================================
st.markdown('<div class="section-header">Section 2 — Lancer le Benchmark</div>',
            unsafe_allow_html=True)

if not selected:
    st.warning("Sélectionnez au moins une méthode dans la sidebar.")
else:
    st.info(f"{len(selected)} méthode(s) sélectionnée(s) — {limit} échantillons")

    if st.button("Lancer le Benchmark", use_container_width=True):
        X_sub, y_sub = X_all[:limit], y_all[:limit]
        results = []
        prog = st.progress(0, text="Initialisation...")

        for idx, method in enumerate(selected):
            prog.progress(idx / len(selected), text=f"{method} ({idx+1}/{len(selected)})")
            tracemalloc.start()
            t0 = time.time()
            try:
                X_enc = run_encoding(X_sub, method)
                Xtr, Xte, ytr, yte = train_test_split(X_enc, y_sub, test_size=0.2, random_state=42)
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
                clf.fit(Xtr, ytr)
                ypred = clf.predict(Xte)
                acc = accuracy_score(yte, ypred)
                t_tot = round(time.time() - t0, 2)
                _, mem_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                results.append({
                    "Méthode": method,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Temps (s)": t_tot,
                    "Mémoire (MB)": round(mem_peak / 1024**2, 3),
                    "Qubits": info["qubits"],
                    "CM": confusion_matrix(yte, ypred),
                    "y_test": yte,
                    "y_pred": ypred,
                })
            except Exception as e:
                tracemalloc.stop()
                st.error(f"Erreur sur {method} : {e}")

        prog.progress(1.0, text="Benchmark terminé.")

        df_res = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("CM", "y_test", "y_pred")}
            for r in results
        ]).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        st.markdown("#### Résultats")
        st.dataframe(df_res, use_container_width=True)

        best = df_res.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Meilleure méthode", best["Méthode"])
        col2.metric("Accuracy", f"{best['Accuracy (%)']:.2f}%")
        col3.metric("Temps", f"{df_res.nsmallest(1,'Temps (s)')['Temps (s)'].values[0]}s")

        # Export
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Exporter CSV", df_res.to_csv(index=False),
                               "benchmark.csv", "text/csv")
        with c2:
            st.download_button("Exporter JSON",
                               df_res.to_json(orient="records", indent=2),
                               "benchmark.json", "application/json")

        # Graphiques
        st.markdown("#### Visualisations")
        colors = [METHOD_DEFINITIONS.get(m, {}).get("couleur", "#4a90d9") 
                  for m in df_res["Méthode"]]

        tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Temps", "Mémoire", "Matrices de confusion"])

        def make_fig(h=None):
            fig, ax = plt.subplots(figsize=(10, h or max(4, len(df_res)*0.55)))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8f9fa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#dee2e6')
            return fig, ax

        with tab1:
            fig, ax = make_fig()
            bars = ax.barh(df_res["Méthode"], df_res["Accuracy (%)"],
                           color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Accuracy (%)")
            ax.set_title("Accuracy par méthode d'encodage", fontweight='bold')
            ax.set_xlim(0, 115)
            for bar, val in zip(bars, df_res["Accuracy (%)"]):
                ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va='center', fontsize=9)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            fig, ax = make_fig()
            ax.barh(df_res["Méthode"], df_res["Temps (s)"],
                    color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Temps (s)")
            ax.set_title("Temps d'exécution", fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            fig, ax = make_fig()
            ax.barh(df_res["Méthode"], df_res["Mémoire (MB)"],
                    color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Mémoire pic (MB)")
            ax.set_title("Consommation mémoire", fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            cols_cm = st.columns(min(3, len(results)))
            for i, r in enumerate(results):
                with cols_cm[i % 3]:
                    fig, ax = plt.subplots(figsize=(3.5, 3))
                    ConfusionMatrixDisplay(r["CM"],
                        display_labels=["Non-diab.", "Diab."]).plot(
                        ax=ax, colorbar=False, cmap="Blues")
                    ax.set_title(f"{r['Méthode']}\n{r['Accuracy (%)']}%", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)