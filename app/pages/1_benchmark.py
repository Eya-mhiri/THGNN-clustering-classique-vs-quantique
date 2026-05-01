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

# ============================================================
# CONFIGURATION ET STYLE (TON STYLE)
# ============================================================
st.set_page_config(page_title="Quantum Benchmark", layout="wide")
st.markdown("""
<style>
.method-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
    border-left: 4px solid #4a90d9;
    border-radius: 10px;
    padding: 2rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    color: #e0e0e0;
    font-size: 0.95rem;
    line-height: 1.8;
}
.method-card h3 { color: #7eb8f7; margin-bottom: 0.8rem; font-size: 1.3rem; }
.section-title {
    background: linear-gradient(90deg, #1565c0 0%, #6a1b9a 100%);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: bold;
    margin: 1.5rem 0 1rem 0;
}
.stButton>button {
    background: linear-gradient(90deg, #1565c0, #6a1b9a);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 0.6rem 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🌌 Quantum Benchmark")
st.markdown("**Exploration et comparaison des méthodes d'encodage quantique** — Dataset : Pima Indians Diabetes")

# ============================================================
# DÉVICES QUANTIQUES (FIX : Ajout de dev3 pour Amplitude/Feature Map)
# ============================================================
dev8 = qml.device("default.qubit", wires=8)
dev3 = qml.device("default.qubit", wires=3) 
dev11 = qml.device("default.qubit", wires=11) # Pour Block Encoding (8+3)

# ============================================================
# TES DÉFINITIONS (GARDÉES TELLES QUELLES)
# ============================================================
METHOD_DEFINITIONS = {
    "Raw Data (Baseline)": {
        "description": "Référence classique sans encodage quantique. Les données normalisées dans [0,1] par min-max scaling sont directement transmises au classifieur. Sert de référence absolue pour quantifier l'apport de chaque encodage quantique.",
        "formule": "X_out = X_norm  (identité, pas de transformation quantique)",
        "qubits": "0 qubit — pipeline entièrement classique",
        "avantage": "Temps d'exécution minimal, référence comparative absolue.",
        "limite": "Aucun enrichissement de l'espace de features.",
        "couleur": "#b2bec3", "icone": "⚫"
    },
    "Angle Encoding": {
        "description": "Chaque feature classique xᵢ ∈ [0,1] est convertie en angle de rotation appliqué à un qubit dédié via RX et RY. L'état final est le produit tensoriel des états individuels. La partie réelle du vecteur d'état (dim 256) est extraite comme features.",
        "formule": "RX(xᵢ·π) · RY(xᵢ·π/2) |0⟩ → Re(|ψ⟩) ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 1 qubit par feature",
        "avantage": "Intuitif, continu, préserve la structure géométrique.",
        "limite": "Pas d'intrication — features traitées indépendamment.",
        "couleur": "#4a90d9", "icone": "🔵"
    },
    "Phase Encoding": {
        "description": "Encode les features dans la phase complexe des amplitudes. Une porte Hadamard crée une superposition uniforme, puis RZ encode la feature dans la phase : (|0⟩ + e^{ixᵢπ}|1⟩)/√2. L'information est accessible via des circuits d'interférence.",
        "formule": "|0⟩ →[H]→ →[RZ(xᵢπ)]→ (|0⟩ + e^{ixᵢπ}|1⟩)/√2",
        "qubits": "8 qubits — 1 qubit par feature",
        "avantage": "Encode dans la phase, exploitable via l'interférence quantique.",
        "limite": "Phase indétectable sans circuit d'interférence dédié.",
        "couleur": "#7b2ff7", "icone": "🟣"
    },
    "Amplitude Encoding": {
        "description": "Encode un vecteur de N features directement dans les amplitudes d'un état quantique à ⌈log₂N⌉ qubits. Pour 8 features, seulement 3 qubits suffisent. Le vecteur doit être de norme L2 unitaire. Compression exponentielle de l'information.",
        "formule": "|ψ⟩ = Σᵢ xᵢ|i⟩ avec ‖x‖₂=1 → 3 qubits pour 8 features",
        "qubits": "3 qubits — compression exponentielle (log₂(8)=3)",
        "avantage": "Compression exponentielle : N features → log₂(N) qubits.",
        "limite": "Normalisation L2 obligatoire ; préparation d'état coûteuse.",
        "couleur": "#00b894", "icone": "🟢"
    },
    "Density / Hybrid Encoding": {
        "description": "Représente l'état quantique par sa matrice densité ρ = |ψ⟩⟨ψ|. Cette représentation capture les cohérences entre états (termes hors-diagonaux ρᵢⱼ), invisibles dans la distribution de probabilité classique. La matrice est aplatie en vecteur de dim 64.",
        "formule": "ρ = |ψ⟩⟨ψ| = x_norm·x_normᵀ ∈ ℝ⁸ˣ⁸ → flatten → ℝ⁶⁴",
        "qubits": "8 qubits (conceptuel) — implémentation classique via matrice densité",
        "avantage": "Capture les corrélations croisées entre features.",
        "limite": "Dimension de sortie N² croît quadratiquement.",
        "couleur": "#e17055", "icone": "🟠"
    },
    "Feature Map Encoding": {
        "description": "Projette les données dans un espace de Hilbert via un circuit IQP (Instantaneous Quantum Polynomial), analogue aux noyaux kernel SVM. Une PCA à 3 composantes précède l'encodage. Le noyau quantique K(x,z)=|⟨φ(x)|φ(z)⟩|² est potentiellement intractable classiquement.",
        "formule": "H⊗³ → RZ(π·xᵢ) → CZ → RY(π·xᵢ) | K(x,z)=|⟨φ(x)|φ(z)⟩|²",
        "qubits": "3 qubits (après PCA à 3 composantes)",
        "avantage": "Kernel quantique puissant pour données non linéairement séparables.",
        "limite": "PCA entraîne une perte d'information ; sensible aux répétitions.",
        "couleur": "#fdcb6e", "icone": "🟡"
    },
    "Variational Encoding": {
        "description": "Data re-uploading : les features sont ré-injectées dans le circuit, intercalées avec des paramètres entraînables θ. Deux couches alternent RY(xᵢπ)·RZ(θᵢ) avec intrication CNOT. Approximateur universel quantique selon Pérez-Salinas et al. (2020).",
        "formule": "[RY(xᵢπ)·RZ(θᵢ) + CNOT] × 2 couches → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 16 paramètres entraînables (2 couches × 8 qubits)",
        "avantage": "Expressivité maximale via re-uploading ; approximateur universel.",
        "limite": "Paramètres non optimisés dans ce benchmark (initialisation aléatoire).",
        "couleur": "#ff7675", "icone": "🔴"
    },
    "Block Encoding": {
        "description": "Encode la matrice diagonale des features dans un bloc d'un opérateur unitaire plus grand, via des qubits ancilla. Inspiré des algorithmes HHL et QLSA. Des rotations contrôlées ctrl-RY appliquent la feature uniquement si tous les ancilla sont dans |1⟩.",
        "formule": "H⊗³(ancilla) → ctrl-RY(xᵢ) → ⟨Z_{ancilla}⟩ ∈ ℝ¹",
        "qubits": "11 qubits — 3 ancilla + 8 data qubits",
        "avantage": "Fondement des algorithmes quantiques d'algèbre linéaire.",
        "limite": "Sortie scalaire (dim 1) — très faible expressivité pour la classification.",
        "couleur": "#636e72", "icone": "🔲"
    },
    "Directional Encoding": {
        "description": "Exploite les trois degrés de liberté de chaque qubit sur la sphère de Bloch via RX·RY·RZ avec la même valeur xᵢ. La non-commutativité des rotations crée un encodage riche. Une couche CNOT en chaîne crée ensuite de l'intrication entre qubits voisins.",
        "formule": "RX(xᵢ)·RY(xᵢ)·RZ(xᵢ)|0⟩ + CNOT chaîne → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 3 rotations par qubit + CNOT en chaîne",
        "avantage": "Exploite les 3 axes de la sphère de Bloch ; intrication entre voisins.",
        "limite": "Redondance : 3 rotations avec la même valeur xᵢ.",
        "couleur": "#00cec9", "icone": "🔷"
    },
    "Entangler Enhanced": {
        "description": "Inspiré de l'évolution temporelle sous un Hamiltonien de champ transverse. Étape 1 : superposition globale H⊗⁸. Étape 2 : évolution de phase RZ(xᵢ). Étape 3 : intrication CNOT en chaîne. L'information de chaque feature est distribuée globalement via l'intrication.",
        "formule": "H⊗⁸ → RZ(xᵢ) → CNOT chaîne → |ψ|² ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — évolution temporelle + intrication globale",
        "avantage": "Représentation collective des features via l'intrication.",
        "limite": "Information individuelle diluée dans l'état global.",
        "couleur": "#6c5ce7", "icone": "🔮"
    },
    "QSample Encoding": {
        "description": "Encode les features comme probabilités de mesure : P(|1⟩) = xᵢ. La porte RY(2·arcsin(√xᵢ)) prépare chaque qubit tel que sa probabilité de mesure soit exactement xᵢ. La valeur d'espérance ⟨Zᵢ⟩ = 1-2xᵢ ∈ [-1,1] est extraite comme feature.",
        "formule": "RY(2·arcsin(√xᵢ))|0⟩ → ⟨Zᵢ⟩ = 1-2xᵢ ∈ [-1,1]⁸",
        "qubits": "8 qubits — encodage probabiliste via arcsin",
        "avantage": "Encodage probabiliste naturel ; sortie dans [-1,1] interprétable.",
        "limite": "Transformation monotone décroissante — peut inverser l'ordre des features.",
        "couleur": "#a29bfe", "icone": "🫧"
    },
    "Chebyshev Encoding": {
        "description": "Motivé par la théorie de l'approximation fonctionnelle. Les polynômes de Chebyshev Tₙ(x)=cos(n·arccos(x)) forment une base orthogonale optimale. θᵢ=arccos(2xᵢ-1), puis RY(θᵢ) encode T₁ et RY(2θᵢ) encode T₂, projetant implicitement dans un espace polynomial de degré 2.",
        "formule": "θᵢ=arccos(2xᵢ-1) → RY(θᵢ)·RY(2θᵢ)|0⟩ → probs ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — encodage via polynômes T₁ et T₂ de Chebyshev",
        "avantage": "Capture des relations quadratiques ; base optimale pour l'approximation.",
        "limite": "Limité aux degrés 1 et 2 dans cette implémentation.",
        "couleur": "#55efc4", "icone": "🟦"
    },
    "Fourier Encoding": {
        "description": "Encode deux harmoniques de Fourier par qubit : H → RZ(xᵢπ) → RZ(2xᵢπ). Fondé sur le cadre des quantum Fourier features (Schuld et al.) : les circuits quantiques paramétriques calculent naturellement des séries de Fourier tronquées, capturant des patterns périodiques.",
        "formule": "H → RZ(xᵢπ) → RZ(2xᵢπ) |0⟩ → probs ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — 2 harmoniques de Fourier par qubit",
        "avantage": "Capture des patterns périodiques ; fondé sur les quantum Fourier features.",
        "limite": "Limité à 2 harmoniques ; pas d'intrication entre qubits.",
        "couleur": "#fab1a0", "icone": "🎵"
    },
    "Projected Unitary Encoding": {
        "description": "Mesure multi-observable : après AngleEmbedding(xᵢπ), les valeurs d'espérance ⟨Xᵢ⟩=sin(xᵢπ) et ⟨Zᵢ⟩=cos(xᵢπ) sont extraites pour chaque qubit. Les 16 features résultantes représentent chaque feature dans deux bases complémentaires avec la contrainte ⟨X⟩²+⟨Z⟩²=1.",
        "formule": "AngleEmb(xᵢπ) → [⟨Xᵢ⟩, ⟨Zᵢ⟩] pour i=1..8 → ℝ¹⁶",
        "qubits": "8 qubits — double projection sur observables X et Z",
        "avantage": "Double projection complémentaire ; sortie compacte de dimension 16.",
        "limite": "⟨X⟩²+⟨Z⟩²=1 — les 16 features ne sont pas indépendantes.",
        "couleur": "#74b9ff", "icone": "📐"
    },
    "Scaled Encoding": {
        "description": "Variante de l'Angle Encoding utilisant l'angle 2π au lieu de π, exploitant le cycle complet de la sphère de Bloch. Adapté aux données périodiques. Attention : RY(0)|0⟩ = RY(2π)|0⟩ = |0⟩ crée une ambiguïté aux bornes (xᵢ=0 et xᵢ=1 donnent le même état).",
        "formule": "RY(xᵢ·2π)|0⟩ → cos(xᵢπ)|0⟩+sin(xᵢπ)|1⟩ → probs ∈ ℝ²⁵⁶",
        "qubits": "8 qubits — cycle complet [0, 2π] de la sphère de Bloch",
        "avantage": "Exploite le cycle complet ; adapté aux données périodiques.",
        "limite": "Ambiguïté aux bornes : xᵢ=0 et xᵢ=1 donnent le même état.",
        "couleur": "#e84393", "icone": "📏"
    },
}


# ============================================================
# CHARGEMENT DATA
# ============================================================
@st.cache_data
def load_data():
    ds = load_dataset("Genius-Society/Pima")
    df = pd.DataFrame(ds['train'])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y

X_all, y_all = load_data()

# ============================================================
# LOGIQUE D'ENCODAGE (TON CODE INTACT AVEC FIX DEVICES)
# ============================================================
def run_encoding(X, method):
    if method == "Raw Data (Baseline)":
        return X
    elif method == "Angle Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.RX(x[i] * np.pi, wires=i)
                qml.RY(x[i] * (np.pi/2), wires=i)
            return qml.state()
        return np.real(np.array([circ(r) for r in X]))
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
            psi = xn.reshape(-1, 1)
            rho = np.dot(psi, psi.conj().T)
            return rho.real.flatten()
        return np.array([density(r) for r in X])
    elif method == "Feature Map Encoding":
        Xr = PCA(n_components=3).fit_transform(X)
        @qml.qnode(dev3)
        def circ(x):
            for i in range(3):
                qml.Hadamard(wires=i)
            for i in range(3):
                qml.RZ(np.pi * float(x[i]), wires=i)
            for i in range(2):
                qml.CZ(wires=[i, i+1])
            for i in range(3):
                qml.RY(np.pi * float(x[i]), wires=i)
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
            for i in range(7):
                qml.CNOT(wires=[i, i+1])
            for i in range(8):
                qml.RY(x[i] * np.pi, wires=i)
                qml.RZ(th[i+8], wires=i)
            return qml.state()
        return np.abs(np.array([circ(r, theta) for r in X]))**2
    elif method == "Block Encoding":
        n_data, n_anc = 8, 3
        dev_b = qml.device("default.qubit", wires=n_data+n_anc)
        @qml.qnode(dev_b)
        def circ(x):
            xn = x / (np.linalg.norm(x) + 1e-9)
            for i in range(n_anc):
                qml.Hadamard(wires=i)
            for i in range(n_data):
                qml.ctrl(qml.RY, control=range(n_anc))(xn[i], wires=n_anc+i)
            return qml.expval(qml.PauliZ(wires=n_anc))
        return np.array([[circ(r)] for r in X])
    elif method == "Directional Encoding":
        @qml.qnode(dev8)
        def circ(x):
            xn = x / (np.max(x) + 1e-9)
            for i in range(8):
                qml.RX(xn[i], wires=i)
                qml.RY(xn[i], wires=i)
                qml.RZ(xn[i], wires=i)
            for i in range(7):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        return np.abs(np.array([circ(r) for r in X]))**2
    elif method == "Entangler Enhanced":
        @qml.qnode(dev8)
        def circ(x):
            xn = x / (np.max(x) + 1e-9)
            for i in range(8):
                qml.Hadamard(wires=i)
            for i in range(8):
                qml.RZ(xn[i], wires=i)
            for i in range(7):
                qml.CNOT(wires=[i, i+1])
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
            return (
                [qml.expval(qml.PauliX(i)) for i in range(8)] +
                [qml.expval(qml.PauliZ(i)) for i in range(8)]
            )
        return np.array([circ(r) for r in X])
    elif method == "Scaled Encoding":
        @qml.qnode(dev8)
        def circ(x):
            for i in range(8):
                qml.RY(x[i] * 2 * np.pi, wires=i)
            return qml.probs(wires=range(8))
        return np.array([circ(r) for r in X])
    return X


# ============================================================
# UI ET BENCHMARK
# ============================================================
st.sidebar.header("⚙️ Config")
all_methods = list(METHOD_DEFINITIONS.keys())
selected = st.sidebar.multiselect("Méthodes", all_methods, default=all_methods[:3])
limit = st.sidebar.slider("Samples", 10, len(X_all), 100)

if st.button("🚀 Run Benchmark"):
    X, y = X_all[:limit], y_all[:limit]
    results = []
    tracemalloc.start() # Démarrage du suivi mémoire

    for method in selected:
        t0 = time.time()
        tracemalloc.clear_traces()
        
        try:
            X_enc = run_encoding(X, method)
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
            
            clf = DecisionTreeClassifier(max_depth=5)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            
            _, peak = tracemalloc.get_traced_memory()
            
            results.append({
                "Méthode": method,
                "Accuracy": f"{acc:.2%}",
                "Temps (s)": round(time.time() - t0, 2),
                "Mémoire (MB)": round(peak / 10**6, 2)
            })
        except Exception as e:
            st.error(f"Erreur sur {method}: {e}")

    st.table(pd.DataFrame(results))
    tracemalloc.stop()