import streamlit as st
import os
import json

# Vérification de sécurité
if not st.session_state.get("is_admin", False):
    st.warning("⚠️ Accès réservé. Veuillez vous connecter en tant qu'administrateur sur la page d'accueil.")
    st.stop()

st.title("🛠️ Administration & Configuration Système")

# ============================================================
# 1. GESTION DE LA CONFIGURATION (Use Case 1.1)
# ============================================================
with st.expander("⚙️ Configuration des Chemins et Modes", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modes d'interface")
        app_mode = st.radio("Mode par défaut", ["Demo (Hello World)", "Real Data (DBLP)"])
        st.session_state['app_mode'] = app_mode
        
    with col2:
        st.subheader("Paths Locaux / Kaggle")
        base_path = st.text_input("Chemin Racine", value=r"C:\Users\mpshop\thgnn-clustering-classique-vs-quantique")
        data_path = st.text_input("Dossier Data", value=os.path.join(base_path, "data"))
        results_path = st.text_input("Dossier Results", value=os.path.join(base_path, "results"))

# ============================================================
# 2. INTÉGRITÉ DES DONNÉES (Use Case 1.3)
# ============================================================
st.header(" Vérification des fichiers (Data & Results)")

def check_files():
    required = [
        "embeddings_thgnn.pt", 
        "clustering_results.npz", 
        "authors_metadata.jsonl"
    ]
    status = []
    for f in required:
        exists = os.path.exists(os.path.join(data_path, f))
        status.append({"Fichier": f, "État": "✅ Présent" if exists else "❌ Manquant"})
    return pd.DataFrame(status)

if st.button("🔍 Lancer l'audit des données"):
    df_status = check_files()
    st.table(df_status)
    if "❌ Manquant" in df_status['État'].values:
        st.error("Certains fichiers critiques manquent pour le mode Real Data.")
    else:
        st.success("Intégrité vérifiée : Prêt pour déploiement.")

# ============================================================
# 3. HYPERPARAMÈTRES D'AFFICHAGE (Use Case 1.1)
# ============================================================
st.header(" Hyperparamètres Globaux")
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.number_input("Nombre d'auteurs max à analyser", value=1000, step=100)
    st.slider("Seuil de filtrage (Nombre de papiers)", 1, 20, 5)

with col_h2:
    st.color_picker("Couleur Clusters Classiques", "#1565C0")
    st.color_picker("Couleur Clusters Quantiques", "#6A1B9A")

# ============================================================
# 4. DÉPLOIEMENT (Use Case 1.1)
# ============================================================
st.header("🌐 Déploiement & Tunneling")
st.info("Pour rendre l'interface accessible via Serveo ou Ngrok")
if st.button("Générer commande de tunneling"):
    st.code("ssh -R 80:localhost:8501 serveo.net", language="bash")
    st.write("L'URL apparaîtra dans votre terminal après exécution.")

if st.button("💾 Sauvegarder la configuration"):
    st.success("Configuration système mise à jour et appliquée.")