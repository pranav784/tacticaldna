# Beyond Traditional Positions: A ML Framework for Tactical Role Identification
# Author  : Pranav Nair
# Thesis  : Master's Dissertation — Computer Science / ML
# Pipeline: PCA (90% var) → GMM (k=8) → RF Validator → Cosine Similarity

# Standard library
import warnings
warnings.filterwarnings("ignore")

# Third-party
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
#  GLOBAL THEME & PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TacticalDNA - Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  :root {
    --bg-base:   #070B14;
    --bg-card:   #0D1526;
    --bg-raised: #111D33;
    --accent:    #00E5A0;
    --accent2:   #4F8EF7;
    --accent3:   #F7A34F;
    --text-hi:   #EDF2FF;
    --text-mid:  #9BAACB;
    --text-lo:   #4A5A7A;
    --border:    rgba(79,142,247,0.15);
    --glow:      0 0 28px rgba(0,229,160,0.12);
  }

  .stApp { background: var(--bg-base) !important; }
  .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1420px; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0F1E 0%, #070B14 100%) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-hi) !important; }

  /* Radio button spacing in sidebar */
  [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 6px !important; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 12px !important;
    border-radius: 8px;
    transition: background 0.15s;
    white-space: normal !important;
    word-break: break-word;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(79,142,247,0.08);
  }

  /* Hide Streamlit's built-in sidebar collapse button (renders as keyboard_double_arrow_left) */
  [data-testid="stSidebarCollapseButton"],
  button[aria-label="Collapse sidebar"],
  button[aria-label="Close sidebar"],
  button[kind="header"],
  section[data-testid="stSidebar"] > div:first-child button {
    display: none !important;
  }

  html, body, .stApp, p, div, span, label {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-hi);
  }
  h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

  [data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: var(--glow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px rgba(0,229,160,0.18);
  }
  [data-testid="stMetricLabel"] {
    color: var(--text-mid) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.85rem !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-mid);
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    padding: 6px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), #00BFA8) !important;
    color: var(--bg-base) !important;
    font-weight: 800;
  }

  .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    color: var(--text-hi) !important;
  }

  [data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  hr { border-color: var(--border) !important; }
  .stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
  }

  .sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    padding: 0.3rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
    display: block;
  }

  .player-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-raised) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: var(--glow);
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, transform 0.2s;
  }
  .player-card:hover {
    border-color: rgba(0,229,160,0.3);
    transform: translateY(-1px);
  }
  .player-card h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-hi);
    margin: 0 0 0.4rem;
    font-size: 1.05rem;
  }
  .badge {
    display: inline-block;
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.4);
    color: var(--accent);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 5px;
  }
  .badge-cluster {
    background: rgba(79,142,247,0.1);
    border-color: rgba(79,142,247,0.4);
    color: var(--accent2);
  }
  .badge-pos {
    background: rgba(247,163,79,0.1);
    border-color: rgba(247,163,79,0.4);
    color: var(--accent3);
  }

  .rank-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), #00BFA8);
    color: var(--bg-base);
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.78rem;
    margin-right: 0.6rem;
    flex-shrink: 0;
  }
  .sim-bar-bg {
    background: rgba(79,142,247,0.12);
    border-radius: 100px;
    height: 7px;
    margin-top: 8px;
    overflow: hidden;
  }
  .sim-bar-fill {
    height: 7px;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-hi);
    margin: 0 0 0.2rem;
  }
  .section-sub {
    color: var(--text-mid);
    font-size: 0.88rem;
    margin: 0 0 1.2rem;
  }
  .kpi-accent { color: var(--accent); }
  .kpi-accent2 { color: var(--accent2); }
  .kpi-accent3 { color: var(--accent3); }
  /* Kill Streamlit's default page-change slide animation */
  [data-testid="stAppViewContainer"],
  [data-testid="stMainBlockContainer"],
  section.main > div { transition: none !important; animation: none !important; }
  .stApp > div { transition: none !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  CONSTANTS
# =============================================================================

META_COLS = [
    "Player", "Nation", "Position", "Primary_Pos",
    "Age", "Matches Played", "Starts", "Minutes", "90s Played",
]

# Only the 3 progressive volume metrics lack a per-90 equivalent in the CSV.
# All goal/assist/xG metrics already exist as per-90 columns — we use those directly.
PROG_VOLUME_METRICS = [
    "Progressive Carries", "Progressive Passes", "Progressive Receives",
]

# Non-redundant per-90 metrics from the CSV (no double-counting)
PER90_METRICS = [
    "Goals Per 90", "Assists Per 90",
    "Non-Penalty Goals Per 90",
    "xG Per 90", "xAG Per 90",
    "npxG Per 90",
]

ARCHETYPE_NAMES = {
    0: "The Engine Room",
    1: "The Clinical Finisher",
    2: "The Defensive Anchor",
    3: "The Creative Playmaker",
    4: "The Wide Threat",
    5: "The Ball-Playing Defender",
    6: "The Box-to-Box Dynamo",
    7: "The Pressing Machine",
}

RADAR_FEATURES = [
    "Goals_Per_90_p90", "Assists_Per_90_p90", "xG_Per_90_p90",
    "Progressive_Carries_p90", "Progressive_Passes_p90",
    "Progressive_Receives_p90",
]
RADAR_LABELS = [
    "Goals/90", "Assists/90", "xG/90",
    "Prog Carries/90", "Prog Passes/90", "Prog Receives/90",
]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9BAACB"),
    legend=dict(
        bgcolor="rgba(13,21,38,0.85)",
        bordercolor="rgba(79,142,247,0.25)",
        borderwidth=1,
        font=dict(color="#EDF2FF", size=11),
    ),
)


CLUSTER_COLORS = [
    "#00E5A0", "#4F8EF7", "#F7A34F", "#E05AFF",
    "#FF5A8A", "#5AFFEB", "#FFD95A", "#A0A0FF",
]


# =============================================================================
#  DATA ENGINEERING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_engineer(filepath: str):
    """Load CSV, clean data, filter players, and build per-90 feature matrix."""
    df = pd.read_csv(filepath, index_col=0)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(",", "", regex=False).astype(float)
            except (ValueError, AttributeError):
                pass

    df.fillna(0, inplace=True)

    df = df.drop_duplicates(subset=["Player"], keep="first").reset_index(drop=True)

    df = df[~df["Position"].str.startswith("GB")].copy()
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)
    df = df[df["Minutes"] >= 900].copy().reset_index(drop=True)

    df["Primary_Pos"] = df["Position"].str[:2]

    df_meta = df[META_COLS].copy()

    nineties = df["90s Played"].replace(0, np.nan)

    feature_data = {}

    # Use per-90 columns from CSV directly — no re-dividing (avoids double-counting)
    for col in PER90_METRICS:
        if col in df.columns:
            key = col.replace(" ", "_").replace("+", "plus") + "_p90"
            feature_data[key] = df[col]

    # Progressive metrics: these have no per-90 equivalent in the CSV
    for col in PROG_VOLUME_METRICS:
        if col in df.columns:
            key = col.replace(" ", "_") + "_p90"
            feature_data[key] = df[col] / nineties

    # Disciplinary per 90
    feature_data["Yellow_Cards_p90"] = df["Yellow Cards"] / nineties
    feature_data["Red_Cards_p90"]    = df["Red Cards"] / nineties

    X_norm = pd.DataFrame(feature_data, index=df.index).fillna(0)

    # Ratio features — derived signals that separate positional profiles
    # Ball-playing CBs pass a lot but generate near-zero xG: high ratio = likely defender
    xg_safe = X_norm["xG_Per_90_p90"].clip(lower=0.01)
    feature_data["Pass_to_xG_Ratio"]     = X_norm["Progressive_Passes_p90"] / xg_safe
    # Wingers/attackers carry the ball far more than they pass progressively
    pass_safe = X_norm["Progressive_Passes_p90"].clip(lower=0.01)
    feature_data["Carry_to_Pass_Ratio"]  = X_norm["Progressive_Carries_p90"] / pass_safe
    # Attackers receive the most progressive passes; defenders receive the fewest
    feature_data["Receive_to_Pass_Ratio"]= X_norm["Progressive_Receives_p90"] / pass_safe

    X_norm = pd.DataFrame(feature_data, index=df.index).fillna(0)
    feature_cols = list(X_norm.columns)

    return df_meta, X_norm, feature_cols


# =============================================================================
#  ML PIPELINE
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_pipeline(filepath: str) -> dict:
    """Run full pipeline: Scaler → PCA → GMM → GridSearchCV RF → Cosine Similarity."""
    df_meta, X_norm, feature_cols = load_and_engineer(filepath)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_norm)

    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_pca = X_pca.shape[1]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_pca)],
    )

    top_loadings = {}
    for pc in ["PC1", "PC2", "PC3"]:
        top3 = loadings_df[pc].abs().nlargest(3)
        top_loadings[pc] = [
            {"Feature": feat, "Loading": round(loadings_df[pc][feat], 4)}
            for feat in top3.index
        ]

    gmm = GaussianMixture(
        n_components=8,
        covariance_type="full",
        random_state=42,
        n_init=5,
    )
    cluster_ids = gmm.fit_predict(X_pca)
    df_meta = df_meta.copy()
    df_meta["Cluster_ID"] = cluster_ids
    df_meta["Archetype"] = df_meta["Cluster_ID"].map(ARCHETYPE_NAMES)

    X_sup = np.hstack([X_pca, cluster_ids.reshape(-1, 1)])
    y_sup = df_meta["Primary_Pos"]

    param_grid = {
        "n_estimators":    [200, 400],
        "max_depth":       [8, 12, None],
        "min_samples_split": [5, 10],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=cv,
        scoring="accuracy", n_jobs=-1, verbose=0, refit=True,
    )
    grid_search.fit(X_sup, y_sup)

    best_rf = grid_search.best_estimator_
    cv_accuracy = grid_search.best_score_
    best_params = grid_search.best_params_

    sup_feature_names = [f"PC{i+1}" for i in range(n_pca)] + ["Cluster_ID"]
    importances_df = pd.DataFrame({
        "Feature": sup_feature_names,
        "Importance": best_rf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    y_pred = best_rf.predict(X_sup)
    clf_report = classification_report(y_sup, y_pred, output_dict=True)

    cos_sim_matrix = cosine_similarity(X_pca)

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_pca)],
    )

    return {
        "df_meta":        df_meta,
        "X_norm":         X_norm,
        "X_pca":          pca_df,
        "X_scaled":       X_scaled,
        "feature_cols":   feature_cols,
        "pca":            pca,
        "gmm":            gmm,
        "n_pca":          n_pca,
        "loadings_df":    loadings_df,
        "top_loadings":   top_loadings,
        "cv_accuracy":    cv_accuracy,
        "best_params":    best_params,
        "importances_df": importances_df,
        "clf_report":     clf_report,
        "cos_sim_matrix": cos_sim_matrix,
    }


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def get_replacements(target: str, results: dict,
                     strict_pos: bool = False, top_n: int = 5) -> pd.DataFrame:
    """Return top-N most similar players to target using cosine similarity (self-excluded)."""
    df_meta  = results["df_meta"]
    cos_sim  = results["cos_sim_matrix"]
    idx = df_meta[df_meta["Player"] == target].index[0]

    sim_series = pd.Series(cos_sim[idx], index=df_meta.index).drop(idx)

    if strict_pos:
        pos = df_meta.loc[idx, "Primary_Pos"]
        valid = df_meta[df_meta["Primary_Pos"] == pos].index
        sim_series = sim_series[sim_series.index.isin(valid)]

    top_idx = sim_series.nlargest(top_n).index
    recs = df_meta.loc[top_idx].copy()
    recs["Similarity"]     = sim_series[top_idx].values
    recs["Similarity_Pct"] = (recs["Similarity"] * 100).round(2)
    return recs.reset_index(drop=True)


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """
    Convert a 6-digit hex colour string to an rgba() string accepted by Plotly.
    Plotly does not support 8-digit hex (#RRGGBBAA); rgba() must be used instead.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def radar_chart(player_labels, label_axes, values_per_player, colors):
    """Render an overlapping multi-player radar chart."""
    fig = go.Figure()
    for name, vals, col in zip(player_labels, values_per_player, colors):
        r = vals + [vals[0]]
        theta = label_axes + [label_axes[0]]
        # Convert hex -> rgba for fill; Plotly rejects 8-digit hex (#RRGGBBAA)
        fill_color = hex_to_rgba(col, alpha=0.15) if col.startswith("#") else col
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=col, width=2.5),
            name=name,
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(13,21,38,0.6)",
            angularaxis=dict(
                tickfont=dict(color="#9BAACB", size=11),
                linecolor="rgba(79,142,247,0.2)",
                gridcolor="rgba(79,142,247,0.12)",
            ),
            radialaxis=dict(
                visible=True, showticklabels=False,
                linecolor="rgba(79,142,247,0.2)",
                gridcolor="rgba(79,142,247,0.12)",
            ),
        ),
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=40),
        height=430,
        **PLOTLY_BASE,
    )
    return fig


# =============================================================================
#  SIDEBAR NAVIGATION
# =============================================================================

DATA_PATH = "PlayersFBREF.csv"

with st.sidebar:
    st.markdown(
        '<span class="sidebar-brand">TacticalDNA ⚽</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;"
        "color:#EDF2FF;line-height:1.5;margin:-0.3rem 0 0.4rem'>"
        "Beyond Traditional Positions</p>"
        "<p style='color:#9BAACB;font-size:0.75rem;margin:0 0 1rem;line-height:1.65'>"
        "A ML Framework for Tactical Role<br>Identification &amp; Player Replacement"
        "<br><span style='color:#4A5A7A'>Pranav Nair &nbsp;&middot;&nbsp; Master&#39;s Thesis</span></p>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🔬 Methodology & XAI", "🌐 Tactical Manifold", "🎯 Replacement Engine"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<p style='color:#4A5A7A;font-size:0.75rem;line-height:1.6'>"
        "Data: FBRef · Season 2024/25<br>"
        "Pipeline: PCA → GMM → RF → Cosine<br>"
        "Min. filter: 900 minutes played</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  BOOT - Load data & train pipeline
# =============================================================================

with st.spinner("Fitting ML pipeline (PCA · GMM · RandomForest) — one moment..."):
    results      = train_pipeline(DATA_PATH)

df_meta      = results["df_meta"]
X_norm       = results["X_norm"]
X_pca        = results["X_pca"]
importances  = results["importances_df"]
top_loadings = results["top_loadings"]
cv_acc       = results["cv_accuracy"]
n_pca        = results["n_pca"]


# =============================================================================
#  PAGE 1 — METHODOLOGY & EXPLAINABLE AI
# =============================================================================

if page == "🔬 Methodology & XAI":

    # Title
    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#EDF2FF;margin-bottom:0'>"
        "Methodology & <span style='color:#00E5A0'>Explainable AI</span></h1>"
        "<p style='color:#9BAACB;margin-top:0.3rem'>"
        "Mathematical audit trail — pipeline transparency for academic and commercial review.</p>",
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated Players", f"{len(df_meta):,}")
    c2.metric("PCA Dimensions", str(n_pca))
    c3.metric("CV Accuracy (RF)", f"{cv_acc*100:.2f}%")
    c4.metric("Tactical Archetypes", "8 (GMM)")

    st.divider()

    # Executive summary — plain styled block, no expander widget
    st.markdown(
        "<div style='background:var(--bg-card);border:1px solid var(--border);"
        "border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:0.5rem'>"
        "<p style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
        "color:#EDF2FF;margin:0 0 0.8rem'>Executive Summary — Pipeline Architecture</p>",
        unsafe_allow_html=True,
    )
    st.markdown(f"""
**Research Objective:** Transcend volume-based scouting heuristics by constructing an
objective, mathematically rigorous Tactical DNA framework grounded in unsupervised
manifold learning.

| Phase | Method | Mathematical Rationale |
|---|---|---|
| Feature Engineering | Per-90 Normalisation | Removes playing-time confound (Simpson's Paradox) |
| Scaling | StandardScaler (Z-score) | Unit-variance alignment across heterogeneous feature scales |
| Dim. Reduction | PCA — 90% variance retained ({n_pca} components) | Collapses multicollinear metrics; orthogonal basis |
| Clustering | GMM (full covariance, k=8) | Soft assignments; ellipsoidal components respect tactical overlaps |
| Validation | RandomForest + GridSearchCV (5-fold StratifiedCV) | Supervised proof that GMM clusters encode positional DNA |
| Scouting | Cosine Similarity on PCA space | Magnitude-invariant; ideal for high-dimensional latent comparisons |

**Why GMM over K-Means?** K-Means enforces hard Voronoi partitions. Football players
(e.g. deep-lying forwards vs box-to-box midfielders) share statistical DNA — GMM's
probabilistic soft assignments natively handle overlapping distributions.

> **Data Limitation Note:** The FBRef dataset contains only attacking/progressive metrics.
> Defensive statistics (tackles, interceptions, clearances, aerial duels) are absent.
> To compensate, three ratio features are derived — `Pass-to-xG`, `Carry-to-Pass`,
> and `Receive-to-Pass` — which expose ball-playing defenders clustering with midfielders.
> Despite this, some high-passing centre-backs may still appear in midfielder archetypes,
> which is a data constraint, not a model flaw.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Two-column: importances + PCA loadings
    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown(
            "<p class='section-header'>RF Feature <span class='kpi-accent2'>Importances</span></p>"
            "<p class='section-sub'>Impurity-based importance scores. "
            "High Cluster_ID importance proves GMM captured tactical reality.</p>",
            unsafe_allow_html=True,
        )

        top15 = importances.head(15).copy()
        bar_colors = [
            "#00E5A0" if "Cluster" in row["Feature"] else "#4F8EF7"
            for _, row in top15.iterrows()
        ]

        fig_imp = go.Figure(go.Bar(
            x=top15["Importance"],
            y=top15["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            xaxis=dict(
                title="Mean Decrease in Impurity",
                gridcolor="rgba(79,142,247,0.1)",
                zerolinecolor="rgba(79,142,247,0.15)",
                tickfont=dict(color="#9BAACB"),
                title_font=dict(color="#9BAACB", size=11),
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(color="#EDF2FF", size=11),
            ),
            height=480,
            bargap=0.32,
            margin=dict(l=10, r=20, t=10, b=40),
            **PLOTLY_BASE,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        bp = results["best_params"]
        st.info(
            f"**Best Hyperparameters (GridSearchCV)** — "
            f"n_estimators: `{bp['n_estimators']}` · "
            f"max_depth: `{bp['max_depth']}` · "
            f"min_samples_split: `{bp['min_samples_split']}`"
        )

    with right_col:
        st.markdown(
            "<p class='section-header'>PCA Eigenvector <span class='kpi-accent3'>Loadings</span></p>"
            "<p class='section-sub'>Top-3 original features driving PC1, PC2, PC3 — "
            "the interpretive keys to the latent space.</p>",
            unsafe_allow_html=True,
        )

        pc_colors = {"PC1": "#00E5A0", "PC2": "#4F8EF7", "PC3": "#F7A34F"}
        for pc, col in pc_colors.items():
            st.markdown(
                f"<p style='font-family:Syne;font-weight:700;font-size:0.82rem;"
                f"color:{col};letter-spacing:0.1em;margin:0.8rem 0 4px'>{pc}</p>",
                unsafe_allow_html=True,
            )
            rows = top_loadings[pc]
            ldf = pd.DataFrame(rows)
            ldf["|Loading|"] = ldf["Loading"].abs()
            ldf = ldf[["Feature", "Loading", "|Loading|"]]
            st.dataframe(
                ldf.style
                   .background_gradient(subset=["|Loading|"], cmap="Blues")
                   .format({"Loading": "{:+.4f}", "|Loading|": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # Scree plot
        st.markdown(
            "<p class='section-header' style='font-size:1.05rem;margin-top:1.2rem'>"
            "Explained Variance <span class='kpi-accent'>Scree Plot</span></p>",
            unsafe_allow_html=True,
        )
        ev  = results["pca"].explained_variance_ratio_
        cev = np.cumsum(ev)
        nc  = min(n_pca, 12)

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(nc)],
            y=ev[:nc] * 100,
            name="Individual",
            marker_color="#4F8EF7",
            opacity=0.7,
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(nc)],
            y=cev[:nc] * 100,
            name="Cumulative",
            line=dict(color="#00E5A0", width=2.5),
            mode="lines+markers",
            marker=dict(size=6),
        ))
        fig_scree.add_hline(
            y=90, line_dash="dash", line_color="#F7A34F",
            annotation_text="90% threshold",
            annotation_font_color="#F7A34F",
        )
        fig_scree.update_layout(
            xaxis=dict(tickfont=dict(color="#9BAACB"), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(
                title="Variance (%)",
                tickfont=dict(color="#9BAACB"),
                gridcolor="rgba(79,142,247,0.1)",
                title_font=dict(color="#9BAACB", size=11),
            ),
            height=270,
            margin=dict(l=10, r=10, t=10, b=30),
            **PLOTLY_BASE,
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    st.divider()

    # Classification report
    st.markdown(
        "<p class='section-header'>Classifier <span class='kpi-accent'>Performance Report</span></p>"
        "<p class='section-sub'>Per-class metrics on training data (diagnostic).</p>",
        unsafe_allow_html=True,
    )
    clf = results["clf_report"]
    report_rows = []
    for pos in ["AT", "DF", "MT"]:
        if pos in clf:
            d = clf[pos]
            report_rows.append({
                "Position":  pos,
                "Precision": round(d["precision"], 3),
                "Recall":    round(d["recall"], 3),
                "F1-Score":  round(d["f1-score"], 3),
                "Support":   int(d["support"]),
            })
    st.dataframe(
        pd.DataFrame(report_rows)
          .style.background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="YlGn")
          .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
#  PAGE 2 — THE LATENT TACTICAL MANIFOLD
# =============================================================================

elif page == "🌐 Tactical Manifold":

    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#EDF2FF;margin-bottom:0'>"
        "The Latent <span style='color:#00E5A0'>Tactical Manifold</span></h1>"
        "<p style='color:#9BAACB;margin-top:0.3rem'>"
        "GMM cluster topology projected onto PC1, PC2, PC3. "
        "Proximity encodes tactical similarity.</p>",
        unsafe_allow_html=True,
    )

    # 3D scatter
    plot_df = pd.concat([
        df_meta[["Player", "Primary_Pos", "Cluster_ID", "Archetype"]].reset_index(drop=True),
        X_pca[["PC1", "PC2", "PC3"]].reset_index(drop=True),
    ], axis=1)
    plot_df["Cluster_Label"] = (
        plot_df["Cluster_ID"].astype(str) + "  ·  " + plot_df["Archetype"]
    )

    fig_3d = px.scatter_3d(
        plot_df,
        x="PC1", y="PC2", z="PC3",
        color="Cluster_Label",
        hover_data={
            "Player": True, "Primary_Pos": True,
            "Archetype": True, "Cluster_ID": True,
            "Cluster_Label": False,
        },
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.78,
    )
    fig_3d.update_traces(marker=dict(size=3.5, line=dict(width=0)))
    fig_3d.update_layout(
        height=640,
        scene=dict(
            bgcolor="rgba(7,11,20,1)",
            xaxis=dict(
                gridcolor="rgba(79,142,247,0.1)",
                backgroundcolor="rgba(7,11,20,0)",
                color="#4A5A7A",
            ),
            yaxis=dict(
                gridcolor="rgba(79,142,247,0.1)",
                backgroundcolor="rgba(7,11,20,0)",
                color="#4A5A7A",
            ),
            zaxis=dict(
                gridcolor="rgba(79,142,247,0.1)",
                backgroundcolor="rgba(7,11,20,0)",
                color="#4A5A7A",
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        **PLOTLY_BASE,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # Cluster deep-dive
    st.markdown(
        "<p class='section-header'>Cluster <span class='kpi-accent2'>Deep Dive</span></p>",
        unsafe_allow_html=True,
    )

    cluster_options = {f"Cluster {k} — {v}": k for k, v in ARCHETYPE_NAMES.items()}
    sel_label   = st.selectbox("Select Tactical Archetype", list(cluster_options.keys()))
    sel_cluster = cluster_options[sel_label]

    mask         = df_meta["Cluster_ID"] == sel_cluster
    c_meta       = df_meta[mask].copy()
    c_Xnorm      = X_norm[mask].copy()
    c_Xpca       = X_pca[mask].copy()

    r_left, r_right = st.columns([1, 1], gap="large")

    with r_left:
        st.markdown(
            f"<p class='section-header' style='font-size:1.05rem'>"
            f"Tactical Signature — <span class='kpi-accent'>Cluster {sel_cluster}</span></p>",
            unsafe_allow_html=True,
        )

        valid_r  = [f for f in RADAR_FEATURES if f in X_norm.columns]
        v_labels = [RADAR_LABELS[i] for i, f in enumerate(RADAR_FEATURES)
                    if f in X_norm.columns]
        maxv = X_norm[valid_r].max().values
        maxv[maxv == 0] = 1

        centroid = (c_Xnorm[valid_r].mean().values / maxv).tolist()
        league   = (X_norm[valid_r].mean().values / maxv).tolist()

        fig_radar = radar_chart(
            [f"Cluster {sel_cluster}", "League Avg"],
            v_labels,
            [centroid, league],
            ["#00E5A0", "#4F8EF7"],
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        pos_counts = c_meta["Primary_Pos"].value_counts()
        pos_str = " · ".join(
            [f"<span style='color:#4F8EF7'>{p}</span> ({n})"
             for p, n in pos_counts.items()]
        )
        st.markdown(
            f"<div style='background:rgba(13,21,38,0.9);border:1px solid var(--border);"
            f"border-radius:10px;padding:1rem;'>"
            f"<p style='margin:0;color:#9BAACB;font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:0.1em'>Cluster Composition</p>"
            f"<p style='margin:6px 0 0;font-family:Syne;font-size:0.95rem;color:#EDF2FF'>"
            f"<span style='color:#00E5A0'>{len(c_meta)}</span> players &nbsp;·&nbsp; {pos_str}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with r_right:
        st.markdown(
            "<p class='section-header' style='font-size:1.05rem'>"
            "Top 10 Players — <span class='kpi-accent3'>Nearest Centroid</span></p>",
            unsafe_allow_html=True,
        )
        centroid_pca = c_Xpca.mean().values
        dists = np.linalg.norm(c_Xpca.values - centroid_pca, axis=1)
        c_meta = c_meta.reset_index(drop=True)
        c_meta["Dist"] = dists
        top10 = c_meta.nsmallest(10, "Dist")[
            ["Player", "Nation", "Primary_Pos", "Age", "Dist"]
        ].rename(columns={"Dist": "Dist to Centroid"})

        st.dataframe(
            top10.style
                 .format({"Dist to Centroid": "{:.3f}", "Age": "{:.0f}"})
                 .background_gradient(subset=["Dist to Centroid"], cmap="Blues_r"),
            use_container_width=True,
            hide_index=True,
            height=390,
        )

    st.divider()

    # Position distribution bar
    st.markdown(
        "<p class='section-header'>Position Distribution "
        "<span class='kpi-accent3'>Across Archetypes</span></p>",
        unsafe_allow_html=True,
    )
    dist_df = (
        df_meta.groupby(["Cluster_ID", "Primary_Pos"])
        .size().reset_index(name="Count")
    )
    dist_df["Archetype"] = dist_df["Cluster_ID"].map(ARCHETYPE_NAMES)

    fig_dist = px.bar(
        dist_df, x="Archetype", y="Count", color="Primary_Pos",
        barmode="stack",
        color_discrete_map={"AT": "#00E5A0", "DF": "#4F8EF7", "MT": "#F7A34F"},
    )
    fig_dist.update_layout(
        xaxis=dict(
            tickfont=dict(color="#9BAACB"),
            title_font=dict(color="#9BAACB"),
            gridcolor="rgba(0,0,0,0)",
            tickangle=-30,
        ),
        yaxis=dict(
            title="Players",
            tickfont=dict(color="#9BAACB"),
            gridcolor="rgba(79,142,247,0.1)",
            title_font=dict(color="#9BAACB"),
        ),
        height=380,
        bargap=0.22,
        margin=dict(l=10, r=10, t=50, b=90),
        **PLOTLY_BASE,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
#  PAGE 3 — REPLACEMENT ENGINE
# =============================================================================

elif page == "🎯 Replacement Engine":

    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#EDF2FF;margin-bottom:0'>"
        "Data-Driven <span style='color:#00E5A0'>Replacement Engine</span></h1>"
        "<p style='color:#9BAACB;margin-top:0.3rem'>"
        "Cosine similarity scouting in PCA latent space — identify tactical equivalents "
        "by statistical DNA, not market value or reputation.</p>",
        unsafe_allow_html=True,
    )

    # Controls row
    ctrl1, ctrl2 = st.columns([2.5, 1])
    with ctrl1:
        player_list = sorted(df_meta["Player"].tolist())
        default = "Mohamed Salah" if "Mohamed Salah" in player_list else player_list[0]
        target = st.selectbox(
            "Search target player",
            player_list,
            index=player_list.index(default),
            placeholder="Type a player name...",
        )
    with ctrl2:
        strict = st.toggle("Strict Positional Filter", value=False)
        st.caption("ON: only same Primary_Pos returned")

    st.divider()

    # Target profile
    t_row = df_meta[df_meta["Player"] == target].iloc[0]
    t_idx = df_meta[df_meta["Player"] == target].index[0]

    recs = get_replacements(target, results, strict, top_n=5)

    prof_col, recs_col = st.columns([1.1, 0.9], gap="large")

    with prof_col:
        st.markdown(
            "<p class='section-header'>Target Player "
            "<span class='kpi-accent2'>Profile</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="player-card">
              <h3>{t_row['Player']}</h3>
              <span class="badge">{t_row['Primary_Pos']}</span>
              <span class="badge badge-cluster">{t_row['Archetype']}</span>
              <br/><br/>
              <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
                <tr>
                  <td style="color:#9BAACB;padding:4px 16px 4px 0">Nation</td>
                  <td style="color:#EDF2FF">{t_row['Nation']}</td>
                  <td style="color:#9BAACB;padding:4px 16px">Age</td>
                  <td style="color:#EDF2FF">{int(t_row['Age'])}</td>
                </tr>
                <tr>
                  <td style="color:#9BAACB;padding:4px 16px 4px 0">Minutes</td>
                  <td style="color:#EDF2FF">{int(t_row['Minutes']):,}</td>
                  <td style="color:#9BAACB;padding:4px 16px">Matches</td>
                  <td style="color:#EDF2FF">{int(t_row['Matches Played'])}</td>
                </tr>
                <tr>
                  <td style="color:#9BAACB;padding:4px 16px 4px 0">Position</td>
                  <td style="color:#EDF2FF">{t_row['Position']}</td>
                  <td style="color:#9BAACB;padding:4px 16px">Cluster</td>
                  <td style="color:#00E5A0">{int(t_row['Cluster_ID'])}</td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Radar feature list computed here — used later outside columns
        valid_r = [f for f in RADAR_FEATURES if f in X_norm.columns]
        maxv = X_norm[valid_r].max().values
        maxv[maxv == 0] = 1

    with recs_col:
        st.markdown(
            "<p class='section-header'>Top 5 "
            "<span class='kpi-accent'>Tactical Matches</span></p>",
            unsafe_allow_html=True,
        )
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            pct = row["Similarity_Pct"]
            st.markdown(
                f"""
                <div class="player-card" style="padding:0.85rem 1.1rem;margin-bottom:0.5rem">
                  <div style="display:flex;align-items:center;margin-bottom:0.4rem">
                    <span class="rank-num">{rank}</span>
                    <div>
                      <div style="font-family:Syne;font-weight:700;color:#EDF2FF;
                                  font-size:0.92rem;line-height:1.2">{row['Player']}</div>
                      <div style="margin-top:3px">
                        <span class="badge" style="font-size:0.63rem">{row['Primary_Pos']}</span>
                        <span class="badge badge-cluster" style="font-size:0.63rem">
                          {row['Archetype']}</span>
                      </div>
                    </div>
                  </div>
                  <div style="font-size:0.75rem;color:#9BAACB;margin-bottom:4px">
                    Tactical Similarity:
                    <strong style="color:#00E5A0">{pct:.1f}%</strong>
                  </div>
                  <div class="sim-bar-bg">
                    <div class="sim-bar-fill" style="width:{min(pct,100)}%"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Key stats row — full width so all 4 metrics breathe ────────────────
    key_stats = [
        ("Goals_Per_90_p90",         "Goals / 90"),
        ("xG_Per_90_p90",            "xG / 90"),
        ("Progressive_Carries_p90",  "Prog Carries / 90"),
        ("Progressive_Passes_p90",   "Prog Passes / 90"),
        ("Assists_Per_90_p90",       "Assists / 90"),
        ("npxG_Per_90_p90",          "npxG / 90"),
    ]
    available_stats = [(k, l) for k, l in key_stats if k in X_norm.columns]
    stat_cols = st.columns(len(available_stats))
    for col_st, (key, lbl) in zip(stat_cols, available_stats):
        col_st.metric(lbl, f"{X_norm.loc[t_idx, key]:.2f}")

    st.divider()

    # Comparative radar - target vs #1 replacement
    best_name = recs.iloc[0]["Player"]
    b_idx     = df_meta[df_meta["Player"] == best_name].index[0]

    st.markdown(
        f"<p class='section-header'>Comparative Radar — "
        f"<span class='kpi-accent'>{target}</span> vs "
        f"<span class='kpi-accent2'>{best_name}</span></p>"
        f"<p class='section-sub'>Per-90 tactical dimensions normalised to league maximum.</p>",
        unsafe_allow_html=True,
    )

    if valid_r:
        t_norm = (X_norm.loc[t_idx, valid_r].values / maxv).tolist()
        b_norm = (X_norm.loc[b_idx, valid_r].values / maxv).tolist()
        v_lbl  = [RADAR_LABELS[i] for i, f in enumerate(RADAR_FEATURES) if f in X_norm.columns]

        fig_cmp = radar_chart(
            [target, best_name],
            v_lbl,
            [t_norm, b_norm],
            ["#00E5A0", "#4F8EF7"],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    # Full scouting table
    st.markdown(
        "<p class='section-header'>Full "
        "<span class='kpi-accent3'>Scouting Report</span></p>",
        unsafe_allow_html=True,
    )

    stat_keys = [
        ("Goals_Per_90_p90",        "Goals/90"),
        ("xG_Per_90_p90",           "xG/90"),
        ("Progressive_Carries_p90", "Prog Carries/90"),
        ("Progressive_Passes_p90",  "Prog Passes/90"),
        ("Assists_Per_90_p90",      "Assists/90"),
    ]
    table_rows = []
    for _, row in recs.iterrows():
        ridx = df_meta[df_meta["Player"] == row["Player"]].index[0]
        r = {"Player": row["Player"], "Pos": row["Primary_Pos"],
             "Archetype": row["Archetype"], "Similarity %": row["Similarity_Pct"]}
        for key, lbl in stat_keys:
            if key in X_norm.columns:
                r[lbl] = round(X_norm.loc[ridx, key], 3)
        table_rows.append(r)

    tbl_df = pd.DataFrame(table_rows)
    st.dataframe(
        tbl_df.style
              .background_gradient(subset=["Similarity %"], cmap="Greens")
              .format({"Similarity %": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Similarity distribution histogram
    st.markdown(
        "<p class='section-header'>Similarity Score "
        "<span class='kpi-accent2'>Distribution</span></p>"
        f"<p class='section-sub'>Cosine similarity of all {len(df_meta):,} validated "
        f"players relative to {target}.</p>",
        unsafe_allow_html=True,
    )

    sim_all = pd.Series(
        results["cos_sim_matrix"][t_idx], index=df_meta.index
    ).drop(t_idx)

    threshold = recs["Similarity"].min()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim_all.values,
        nbinsx=60,
        marker=dict(color="#4F8EF7", opacity=0.65, line=dict(width=0)),
        name="All Players",
        hovertemplate="Sim: %{x:.3f} · Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(
        x=threshold, line_dash="dash", line_color="#00E5A0",
        annotation_text=f"Top-5 threshold ({threshold:.3f})",
        annotation_font_color="#00E5A0", annotation_font_size=12,
    )
    fig_hist.update_layout(
        xaxis=dict(
            title="Cosine Similarity Score",
            tickfont=dict(color="#9BAACB"),
            gridcolor="rgba(79,142,247,0.1)",
            title_font=dict(color="#9BAACB"),
        ),
        yaxis=dict(
            title="Player Count",
            tickfont=dict(color="#9BAACB"),
            gridcolor="rgba(79,142,247,0.1)",
            title_font=dict(color="#9BAACB"),
        ),
        height=320,
        bargap=0.05,
        margin=dict(l=10, r=10, t=20, b=40),
        **PLOTLY_BASE,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
