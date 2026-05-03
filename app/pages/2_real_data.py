import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Analyse DBLP", page_icon="📊", layout="wide")

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
# HELPERS
# ============================================================
is_admin = st.session_state.get("is_admin", False)
BASE     = st.session_state.get("base_path", "")
RESULTS  = os.path.join(BASE, "results") if BASE else ""

METHODS_2024  = ["angle", "amplitude", "variational"]
METHODS_2025  = ["angle", "amplitude", "variational", "phase"]
METHOD_LABELS = {
    "angle":      "Angle Encoding",
    "amplitude":  "Amplitude Encoding",
    "variational":"Variational Encoding",
    "phase":      "Phase Encoding"
}

def rpath(*parts):
    return os.path.join(RESULTS, *parts) if RESULTS else ""

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_classic_metrics(summary, year):
    """Extrait les métriques classiques pour une année donnée."""
    if not summary:
        return {}
    for m in summary.get("metrics", []):
        if str(m.get("year")) == str(year):
            return m
    return {}

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Analyse DBLP")
    st.divider()
    st.markdown("**Paramètres**")

    selected_year = st.selectbox("Année d'étude", ["2024", "2025"])
    methods_available = METHODS_2024 if selected_year == "2024" else METHODS_2025
    q_method = st.selectbox(
        "Méthode quantique",
        methods_available,
        format_func=lambda x: METHOD_LABELS[x]
    )

    st.divider()
    if is_admin:
        st.success("Mode Admin actif")
        pub_threshold = st.session_state.get("pub_threshold", 5)
        st.caption(f"Seuil publications : {pub_threshold}")
    else:
        st.info("Connectez-vous en admin depuis la page d'accueil.")

if not BASE:
    st.warning("Chemin racine non configuré. Connectez-vous en administrateur et configurez le chemin.")
    st.stop()

# ============================================================
# TITRE
# ============================================================
st.markdown(f'<div class="page-title">Analyse de clustering DBLP — {selected_year}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">Clustering classique THGNN vs hybride quantique ({METHOD_LABELS[q_method]})</div>',
            unsafe_allow_html=True)
st.divider()

tab_classic, tab_quantum, tab_compare, tab_semantic, tab_longitudinal = st.tabs([
    "Clustering Classique",
    "Clustering Quantique",
    "Comparaison",
    "Analyse Sémantique",
    "Analyse Longitudinale"
])

# ============================================================
# TAB 1 — CLUSTERING CLASSIQUE
# ============================================================
with tab_classic:
    st.markdown('<div class="section-header">Projection UMAP — Approche classique THGNN</div>',
                unsafe_allow_html=True)

    summary     = load_json(rpath("classique", "clustering_summary.json"))
    year_data   = get_classic_metrics(summary, selected_year)
    labels_data = load_json(rpath("classique", f"cluster_labels_{selected_year}.json"))

    col_img, col_metrics = st.columns([2, 1], gap="large")

    with col_img:
        img_path = rpath("classique", f"clustering_{selected_year}.png")
        if os.path.exists(img_path):
            st.image(img_path,
                     caption=f"Communautés THGNN — {selected_year}",
                     use_container_width=True)
        else:
            st.error(f"Image non trouvée : clustering_{selected_year}.png")

    with col_metrics:
        st.markdown("**Métriques globales**")
        if year_data:
            metric_card("Silhouette Score",
                        f"{year_data.get('silhouette', 'N/A'):.4f}")
            metric_card("Davies-Bouldin",
                        f"{year_data.get('davies_bouldin', 'N/A'):.4f}")
            metric_card("Nombre de clusters",
                        year_data.get('n_clusters', 'N/A'))
            metric_card("Taux de bruit",
                        f"{year_data.get('noise_pct', 'N/A'):.1f}%")
            metric_card("Calinski-Harabasz",
                        f"{year_data.get('calinski_harabasz', 'N/A'):.0f}")
            metric_card("Total auteurs",
                        f"{year_data.get('total_authors', 'N/A'):,}"
                        if isinstance(year_data.get('total_authors'), int) else "N/A")
        else:
            st.warning("Métriques non disponibles pour cette année.")

    # Comparaison 2024 vs 2025
    st.markdown('<div class="section-header">Comparaison 2024 vs 2025</div>',
                unsafe_allow_html=True)
    comp_path = rpath("classique", "comparaison_2024_2025.png")
    if os.path.exists(comp_path):
        st.image(comp_path,
                 caption="Évolution temporelle des communautés",
                 use_container_width=True)
    else:
        st.info("Image de comparaison non disponible.")

    # Top-10 clusters
    if year_data and labels_data:
        st.markdown('<div class="section-header">Distribution des clusters — Top 10</div>',
                    unsafe_allow_html=True)

        top10_raw = year_data.get("top10_clusters", [])
        if top10_raw:
            ids   = [str(c["cluster_id"]) for c in top10_raw]
            sizes = [c["size"] for c in top10_raw]

            fig, ax = plt.subplots(figsize=(11, 4))
            fig.patch.set_facecolor("#fafafa")
            ax.set_facecolor("#f0f4ff")
            bars = ax.bar(
                [f"C{c}" for c in ids], sizes,
                color="#4a90d9", edgecolor="white", linewidth=0.8
            )
            ax.set_xlabel("Cluster", fontsize=11)
            ax.set_ylabel("Nombre d'auteurs", fontsize=11)
            ax.set_title(f"Top 10 clusters — Classique {selected_year}",
                         fontweight="bold", fontsize=13)
            for spine in ax.spines.values():
                spine.set_edgecolor("#dee2e6")
            for bar, size in zip(bars, sizes):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(sizes)*0.01,
                        f"{size:,}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================
# TAB 2 — CLUSTERING QUANTIQUE
# ============================================================
with tab_quantum:
    st.markdown(
        f'<div class="section-header">Projection UMAP — {METHOD_LABELS[q_method]} ({selected_year})</div>',
        unsafe_allow_html=True
    )

    q_folder     = rpath("quantique", selected_year, q_method)
    qmetrics     = load_json(os.path.join(q_folder, f"qmetrics_{q_method}_{selected_year}.json"))
    qfig_path    = os.path.join(q_folder, f"qfig_{q_method}_{selected_year}.png")
    qsummary     = load_json(os.path.join(q_folder, "quantum_summary.json"))

    col_q1, col_q2 = st.columns([2, 1], gap="large")

    with col_q1:
        if os.path.exists(qfig_path):
            st.image(qfig_path,
                     caption=f"Communautés quantiques — {METHOD_LABELS[q_method]} {selected_year}",
                     use_container_width=True)
        else:
            st.error(f"Image non trouvée : qfig_{q_method}_{selected_year}.png")

    with col_q2:
        st.markdown("**Métriques quantiques**")
        if qmetrics:
            metric_card("Silhouette Score",
                        f"{qmetrics.get('silhouette', 'N/A'):.4f}")
            metric_card("Davies-Bouldin",
                        f"{qmetrics.get('davies_bouldin', 'N/A'):.4f}")
            metric_card("Nombre de clusters",
                        qmetrics.get("n_clusters", "N/A"))
            metric_card("Taux de bruit",
                        f"{qmetrics.get('noise_pct', 'N/A'):.1f}%")
            metric_card("Total auteurs",
                        f"{qmetrics.get('total_authors', 'N/A'):,}"
                        if isinstance(qmetrics.get('total_authors'), int) else "N/A")

            delta = qmetrics.get("delta_vs_classic", {})
            if delta:
                st.markdown("**Delta vs Classique**")
                sil_d   = delta.get("silhouette", 0)
                db_d    = delta.get("davies_bouldin", 0)
                noise_d = delta.get("noise_pct", 0)
                st.metric("Silhouette",      f"{sil_d:+.4f}",
                          delta_color="normal"  if sil_d   > 0 else "inverse")
                st.metric("Davies-Bouldin",  f"{db_d:+.4f}",
                          delta_color="inverse" if db_d    > 0 else "normal")
                st.metric("Bruit (%)",       f"{noise_d:+.2f}%",
                          delta_color="inverse" if noise_d > 0 else "normal")
        else:
            st.warning("Métriques non disponibles.")

    # Config VQC
    if qsummary:
        with st.expander("Configuration VQC & HDBSCAN"):
            vqc = qsummary.get("vqc_config", {})
            hdb = qsummary.get("hdbscan_params", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**VQC**")
                for k, v in vqc.items():
                    st.caption(f"{k} : {v}")
            with c2:
                st.markdown("**HDBSCAN**")
                for k, v in hdb.items():
                    st.caption(f"{k} : {v}")

    # Top 10 clusters quantiques
    if qmetrics:
        top10_q = qmetrics.get("top10_clusters", [])
        if top10_q:
            st.markdown('<div class="section-header">Top 10 clusters quantiques</div>',
                        unsafe_allow_html=True)
            ids_q   = [f"C{c['cluster_id']}" for c in top10_q]
            sizes_q = [c["size"] for c in top10_q]

            fig, ax = plt.subplots(figsize=(11, 4))
            fig.patch.set_facecolor("#fafafa")
            ax.set_facecolor("#f0f4ff")
            bars = ax.bar(ids_q, sizes_q, color="#6f42c1",
                          edgecolor="white", linewidth=0.8)
            ax.set_xlabel("Cluster", fontsize=11)
            ax.set_ylabel("Nombre d'auteurs", fontsize=11)
            ax.set_title(
                f"Top 10 clusters — {METHOD_LABELS[q_method]} {selected_year}",
                fontweight="bold", fontsize=13
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("#dee2e6")
            for bar, size in zip(bars, sizes_q):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(sizes_q)*0.01,
                        f"{size:,}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================
# TAB 3 — COMPARAISON
# ============================================================
with tab_compare:
    st.markdown('<div class="section-header">Comparaison classique vs quantique</div>',
                unsafe_allow_html=True)

    comp_q_path = rpath("quantique", selected_year, q_method,
                         "quantum_vs_classic_comparison.png")
    if os.path.exists(comp_q_path):
        st.image(comp_q_path,
                 caption=f"Comparaison — {METHOD_LABELS[q_method]} vs Classique — {selected_year}",
                 use_container_width=True)
    else:
        st.info("Figure de comparaison non disponible pour cette méthode.")

    st.divider()

    # Tableau comparatif
    st.markdown('<div class="section-header">Tableau comparatif des métriques</div>',
                unsafe_allow_html=True)

    summary_c = load_json(rpath("classique", "clustering_summary.json"))
    c_data    = get_classic_metrics(summary_c, selected_year)
    qmetrics2 = load_json(rpath("quantique", selected_year, q_method,
                                 f"qmetrics_{q_method}_{selected_year}.json"))

    if c_data and qmetrics2:
        metrics_list = [
            ("Silhouette Score",    "silhouette",         True),
            ("Davies-Bouldin",      "davies_bouldin",     False),
            ("Taux de bruit (%)",   "noise_pct",          False),
            ("Nombre de clusters",  "n_clusters",         None),
            ("Calinski-Harabasz",   "calinski_harabasz",  True),
        ]
        rows = []
        for label, key, hib in metrics_list:
            c_val = c_data.get(key)
            q_val = qmetrics2.get(key)
            if c_val is not None and q_val is not None and hib is not None:
                winner = "Classique" if (hib and c_val > q_val) or \
                                        (not hib and c_val < q_val) else "Quantique"
            else:
                winner = "—"
            rows.append({
                "Métrique":   label,
                "Classique":  round(c_val, 4) if isinstance(c_val, float) else c_val,
                f"Quantique ({METHOD_LABELS[q_method]})":
                              round(q_val, 4) if isinstance(q_val, float) else q_val,
                "Meilleur":   winner
            })

        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Barplot comparatif
        st.markdown('<div class="section-header">Visualisation comparative</div>',
                    unsafe_allow_html=True)

        metrics_plot  = ["silhouette", "davies_bouldin", "noise_pct"]
        labels_plot   = ["Silhouette", "Davies-Bouldin", "Bruit (%)"]
        c_vals        = [c_data.get(m, 0)    for m in metrics_plot]
        q_vals        = [qmetrics2.get(m, 0) for m in metrics_plot]

        x     = np.arange(len(labels_plot))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#f0f4ff")
        ax.bar(x - width/2, c_vals, width, label="Classique",
               color="#4a90d9", edgecolor="white")
        ax.bar(x + width/2, q_vals, width,
               label=f"Quantique ({METHOD_LABELS[q_method]})",
               color="#6f42c1", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot, fontsize=11)
        ax.set_title("Comparaison des métriques de clustering",
                     fontweight="bold", fontsize=13)
        ax.legend(fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#dee2e6")
        plt.tight_layout()
        st.pyplot(fig)

        # Export
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Exporter CSV",
                df_comp.to_csv(index=False),
                f"comparaison_{selected_year}_{q_method}.csv",
                "text/csv"
            )
        with c2:
            st.download_button(
                "Exporter JSON",
                df_comp.to_json(orient="records", indent=2),
                f"comparaison_{selected_year}_{q_method}.json",
                "application/json"
            )
    else:
        st.warning("Données insuffisantes pour la comparaison.")

# ============================================================
# TAB 4 — ANALYSE SÉMANTIQUE
# ============================================================
with tab_semantic:
    st.markdown('<div class="section-header">Analyse sémantique des communautés</div>',
                unsafe_allow_html=True)

    sem_c, sem_q = st.tabs(["Classique", "Quantique"])

    with sem_c:
        labels_c = load_json(rpath("classique", f"cluster_labels_{selected_year}.json"))
        if labels_c:
            search_c = st.text_input("Rechercher un thème :", key="search_classic")
            filtered_c = {
                k: v for k, v in labels_c.items()
                if not search_c or search_c.lower() in v.get("label","").lower()
            }
            st.caption(f"{len(filtered_c)} cluster(s) affichés")

            for cid, info in sorted(filtered_c.items(),
                                     key=lambda x: x[1].get("size", 0),
                                     reverse=True):
                with st.expander(
                    f"Cluster {cid} — {info.get('label','N/A')} "
                    f"({info.get('size',0):,} auteurs)"
                ):
                    kw_col, title_col = st.columns(2)
                    with kw_col:
                        st.markdown("**Mots-clés**")
                        for kw, cnt in info.get("top_keywords", []):
                            st.caption(f"• {kw} ({cnt})")
                    with title_col:
                        st.markdown("**Publications représentatives**")
                        for title in info.get("top_titles", []):
                            st.caption(f"• {title}")
        else:
            st.warning("Fichier d'analyse sémantique classique non disponible.")

    with sem_q:
        q_folder   = rpath("quantique", selected_year, q_method)
        q_labels   = load_json(os.path.join(
            q_folder, f"cluster_labels_quantum_{q_method}_{selected_year}.json"
        ))
        q_csv      = load_csv(os.path.join(
            q_folder, f"cluster_summary_quantum_{q_method}_{selected_year}.csv"
        ))

        if q_labels:
            search_q = st.text_input("Rechercher un thème :", key="search_quantum")
            filtered_q = {
                k: v for k, v in q_labels.items()
                if not search_q or search_q.lower() in v.get("label","").lower()
            }
            st.caption(
                f"{len(filtered_q)} cluster(s) affichés — {METHOD_LABELS[q_method]}"
            )
            for cid, info in sorted(filtered_q.items(),
                                     key=lambda x: x[1].get("size", 0),
                                     reverse=True):
                with st.expander(
                    f"Cluster {cid} — {info.get('label','N/A')} "
                    f"({info.get('size',0):,} auteurs)"
                ):
                    kw_col, title_col = st.columns(2)
                    with kw_col:
                        st.markdown("**Mots-clés**")
                        for kw, cnt in info.get("top_keywords", []):
                            st.caption(f"• {kw} ({cnt})")
                    with title_col:
                        st.markdown("**Publications représentatives**")
                        for title in info.get("top_titles", []):
                            st.caption(f"• {title}")

        if q_csv is not None:
            st.markdown("**Résumé CSV**")
            st.dataframe(q_csv, use_container_width=True)
            st.download_button(
                "Exporter CSV sémantique",
                q_csv.to_csv(index=False),
                f"semantic_{q_method}_{selected_year}.csv",
                "text/csv"
            )

        if not q_labels and q_csv is None:
            st.warning("Fichiers d'analyse sémantique quantique non disponibles.")

# ============================================================
# TAB 5 — ANALYSE LONGITUDINALE
# ============================================================
with tab_longitudinal:
    st.markdown('<div class="section-header">Analyse longitudinale des auteurs</div>',
                unsafe_allow_html=True)

    qsummary_l  = load_json(rpath("quantique", selected_year, q_method,
                                   "quantum_summary.json"))
    summary_cl  = load_json(rpath("classique", "clustering_summary.json"))
    c_data_l    = get_classic_metrics(summary_cl, selected_year)

    if qsummary_l and c_data_l:
        year_key = f"{q_method}_{selected_year}"
        q_data   = qsummary_l.get("quantum", {}).get(year_key, {})

        if not q_data:
            st.warning(f"Clé '{year_key}' non trouvée dans quantum_summary.json")
        else:
            # Résumé global
            st.markdown("**Résumé global**")
            col1, col2, col3, col4 = st.columns(4)

            total = q_data.get("total_authors")
            col1.metric("Total auteurs",
                        f"{total:,}" if isinstance(total, int) else "N/A")
            col2.metric("Clusters quantiques", q_data.get("n_clusters", "N/A"))
            col3.metric("Clusters classiques", c_data_l.get("n_clusters", "N/A"))

            delta_sil = q_data.get("delta_vs_classic", {}).get("silhouette")
            col4.metric("Delta Silhouette",
                        f"{delta_sil:+.4f}" if delta_sil is not None else "N/A",
                        delta_color="normal" if delta_sil and delta_sil > 0
                                    else "inverse")

            st.divider()

            # Top 10 clusters
            top10 = q_data.get("top10_clusters", [])
            if top10:
                st.markdown(
                    '<div class="section-header">Top 10 clusters quantiques</div>',
                    unsafe_allow_html=True
                )
                df_top = pd.DataFrame(top10)
                df_top.columns = ["Cluster ID", "Taille"]

                fig, ax = plt.subplots(figsize=(11, 4))
                fig.patch.set_facecolor("#fafafa")
                ax.set_facecolor("#f0f4ff")
                colors_bar = ["#6f42c1" if i < 3 else "#a29bfe"
                              for i in range(len(df_top))]
                bars = ax.bar(
                    [f"C{c}" for c in df_top["Cluster ID"]],
                    df_top["Taille"],
                    color=colors_bar, edgecolor="white", linewidth=0.8
                )
                ax.set_xlabel("Cluster", fontsize=11)
                ax.set_ylabel("Nombre d'auteurs", fontsize=11)
                ax.set_title(
                    f"Distribution clusters — {METHOD_LABELS[q_method]} {selected_year}",
                    fontweight="bold", fontsize=13
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor("#dee2e6")
                for bar, size in zip(bars, df_top["Taille"]):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(df_top["Taille"])*0.01,
                            f"{size:,}", ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Exporter top clusters CSV",
                        df_top.to_csv(index=False),
                        f"top_clusters_{q_method}_{selected_year}.csv",
                        "text/csv"
                    )
                with c2:
                    st.download_button(
                        "Exporter résumé JSON",
                        json.dumps(q_data, indent=2, ensure_ascii=False),
                        f"summary_{q_method}_{selected_year}.json",
                        "application/json"
                    )

            # Delta métriques
            delta = q_data.get("delta_vs_classic", {})
            if delta:
                st.markdown(
                    '<div class="section-header">Delta quantique vs classique</div>',
                    unsafe_allow_html=True
                )
                delta_items = [
                    ("Silhouette",     delta.get("silhouette", 0),    True),
                    ("Davies-Bouldin", delta.get("davies_bouldin", 0), False),
                    ("Bruit (%)",      delta.get("noise_pct", 0),     False),
                ]
                labels_d = [d[0] for d in delta_items]
                vals_d   = [d[1] for d in delta_items]
                colors_d = []
                for val, (_, _, hib) in zip(vals_d, delta_items):
                    colors_d.append(
                        "#28a745" if (hib and val > 0) or (not hib and val < 0)
                        else "#dc3545"
                    )

                fig, ax = plt.subplots(figsize=(7, 3.5))
                fig.patch.set_facecolor("#fafafa")
                ax.set_facecolor("#f0f4ff")
                bars_d = ax.bar(labels_d, vals_d, color=colors_d,
                                edgecolor="white", linewidth=0.8)
                ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
                ax.set_title("Delta quantique vs classique",
                             fontweight="bold", fontsize=13)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#dee2e6")
                for bar, val in zip(bars_d, vals_d):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.001,
                            f"{val:+.4f}", ha="center", va="bottom", fontsize=10)
                legend_patches = [
                    mpatches.Patch(color="#28a745", label="Amélioration"),
                    mpatches.Patch(color="#dc3545", label="Dégradation")
                ]
                ax.legend(handles=legend_patches, fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("Données longitudinales non disponibles.")