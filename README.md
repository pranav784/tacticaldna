# ⚽ TacticalDNA — Football Intelligence

> **Beyond Traditional Positions: A Machine Learning Framework for Tactical Role Identification & Player Replacement**
>
> *Master's Dissertation — Data Science · Pranav Nair*

---

## The Problem With "Midfielder"

Every football club, every scouting platform, every transfer database still sorts players into three boxes drawn up in the 1870s: **Defender. Midfielder. Attacker.**

Kevin De Bruyne and N'Golo Kanté are both "Midfielders". Virgil van Dijk and Trent Alexander-Arnold are both "Defenders". These labels tell you where a player stands on a formation sheet — they tell you almost nothing about **what that player actually does with the ball**, how much space they cover progressively, how much goal threat they carry, or who in world football does the same job with the same statistical DNA.

That gap is what **TacticalDNA** is built to close.

The system learns eight data-driven **Tactical Archetypes** directly from per-90 performance statistics — no hand-crafted rules, no positional assumptions — and uses the resulting latent representation to power a **Replacement Engine** that finds tactically equivalent players across all leagues, ages, and market values.

---

## What's Inside

```
tacticaldna/
├── futapp.py               # Full Streamlit application (single-file)
├── PlayersFBREF.csv        # FBRef 2024/25 dataset (place here)
├── requirements.txt        # Python dependencies
└── README.md
```

The entire pipeline — data engineering, ML training, and interactive dashboard — lives in **`futapp.py`**. There are no separate training scripts or model artefact files; the pipeline re-fits from the CSV on first launch and is cached in-session via Streamlit's resource cache.

---

## The ML Pipeline

```
Raw FBRef CSV
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1 · FEATURE ENGINEERING                      │
│  • Filter: ≥ 900 minutes played, no goalkeepers     │
│  • Per-90 normalisation  (removes playing-time bias) │
│  • 3 ratio features      (compensate for no def data)│
│    ├─ Pass-to-xG Ratio   → ball-playing CBs          │
│    ├─ Carry-to-Pass Ratio → dribblers vs playmakers  │
│    └─ Receive-to-Pass Ratio → attacking receivers    │
└───────────────────┬─────────────────────────────────┘
                    │  14-column feature matrix  X ∈ ℝⁿˣ¹⁴
                    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2 · STANDARDISATION                          │
│  StandardScaler → Z-score, unit variance per feature│
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3 · PCA  (90% variance retained)             │
│  Collapses multicollinear metrics into k orthogonal  │
│  latent components (typically k = 6–8)               │
│  → Eigenvector loadings surfaced for interpretability│
└───────────────────┬─────────────────────────────────┘
                    │  X_pca ∈ ℝⁿˣᵏ
                    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4 · GMM CLUSTERING  (k=8, full covariance)   │
│  Soft probabilistic assignments → 8 Tactical         │
│  Archetypes. n_init=5 to escape local optima.        │
│  WHY NOT K-MEANS: football roles overlap. GMM's      │
│  ellipsoidal components model that overlap natively. │
└───────────────────┬─────────────────────────────────┘
                    │  cluster_ids, archetype labels
                    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5 · RANDOM FOREST VALIDATOR                  │
│  X_sup = [X_pca | cluster_id]                        │
│  Target y = Primary_Pos (AT / MT / DF)               │
│  GridSearchCV · 5-fold StratifiedKFold               │
│  Proves GMM clusters encode real positional DNA      │
└───────────────────┬─────────────────────────────────┘
                    │  cos_sim_matrix ∈ ℝⁿˣⁿ
                    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 6 · COSINE SIMILARITY SCOUTING               │
│  Magnitude-invariant comparison in PCA latent space  │
│  → Top-N tactical replacement candidates per player  │
└─────────────────────────────────────────────────────┘
```

### Why Each Decision Was Made

| Design Choice | Alternative Rejected | Reason |
|---|---|---|
| GMM clustering | K-Means | K-Means enforces hard Voronoi boundaries; a deep-lying forward and a box-to-box midfielder genuinely overlap statistically. GMM's soft posteriors model that. |
| PCA before clustering | Cluster on raw features | Raw features are multicollinear (xG and npxG are ~0.97 correlated). PCA gives orthogonal axes and removes noise. |
| Cosine similarity | Euclidean distance | Two identical-role players in leagues of different tempo will have proportionally scaled vectors. Cosine ignores magnitude; Euclidean does not. |
| Per-90 rates | Raw counting stats | A player with 10 goals in 3,000 minutes is not equivalent to one with 10 goals in 900 minutes. Simpson's Paradox. |
| Ratio features | Defensive metrics | FBRef public data has no defensive stats. Derived ratios (Pass-to-xG, Carry-to-Pass, Receive-to-Pass) recover positional signal without them. |

---

## The Eight Tactical Archetypes

These are **learned from data**, not defined by a coach. The names are assigned post-hoc by inspecting each cluster's centroid in feature space.

| ID | Archetype | Statistical Signature | Typical Real-World Role |
|---|---|---|---|
| 0 | **The Engine Room** | Balanced progressive metrics; moderate xG | Deep central midfielder, press-resistant 8 |
| 1 | **The Clinical Finisher** | Dominant xG & npxG; low progressive pass volume | Penalty-box striker, poacher |
| 2 | **The Defensive Anchor** | High pass-to-xG ratio; low carry & receive | Holding midfielder, defensive pivot |
| 3 | **The Creative Playmaker** | High progressive passes & xAG; low direct goal threat | Deep-lying playmaker, regista |
| 4 | **The Wide Threat** | High carry-to-pass ratio; elevated xG from wide | Inverted winger, wide forward |
| 5 | **The Ball-Playing Defender** | Highest pass-to-xG ratio; high progressive passes | Ball-playing CB, possession-based libero |
| 6 | **The Box-to-Box Dynamo** | Balanced carry + pass; above-average disciplinary | Box-to-box midfielder, athletic 8 |
| 7 | **The Pressing Machine** | High yellow-card rate; elevated progressive carries | High-press forward, gegenpressing winger |

> **Important caveat:** Because the FBRef dataset lacks defensive statistics (tackles, interceptions, pressures), some high-passing centre-backs may surface in Cluster 3 or 5. This is a **data limitation**, not a model error. The ratio features partially compensate but cannot fully substitute for defensive event data.

---

## The Three Application Pages

### 🔬 Page 1 — Methodology & Explainable AI
The full academic audit trail. Everything the pipeline computed, surfaced transparently:

- **KPI row**: validated player count, PCA dimension count, RF cross-validated accuracy, archetype count
- **Pipeline summary table**: every stage with its mathematical rationale
- **RF Feature Importances**: horizontal bar chart of the top-15 features. `Cluster_ID` consistently ranks top-3 — proof that the GMM layer adds information independent of PCA alone
- **PCA Eigenvector Loadings**: top-3 original features driving PC1, PC2, PC3 with absolute loading values
- **Scree Plot**: individual and cumulative explained variance; 90% threshold shown
- **Classification Report**: precision / recall / F1 per positional class (AT / MT / DF)

### 🌐 Page 2 — Tactical Manifold
Every player in the dataset visualised in the latent space the model actually uses:

- **Interactive 3D scatter** on PC1 × PC2 × PC3, coloured by archetype. Hovering shows player name, position, and cluster
- **Cluster Deep Dive**: select any of the 8 archetypes to see its radar signature (centroid vs. league average) and the 10 players whose PCA vector sits closest to the centroid — the archetype's most "pure" representatives
- **Position Distribution**: stacked bar chart showing how traditional positions (AT/MT/DF) spread across each archetype, exposing cross-positional clusters

### 🎯 Page 3 — Replacement Engine
The operational scouting tool:

- **Player search** with type-ahead across the full validated roster
- **Strict Positional Filter toggle**: OFF finds the best tactical match regardless of position label (recommended); ON restricts to same Primary_Pos
- **Target player profile card**: position badge, archetype badge, nation, age, minutes, matches
- **Top-5 replacement cards**: ranked by cosine similarity, each with an animated similarity progress bar
- **Key stats row**: Goals/90, xG/90, Prog Carries/90, Prog Passes/90, Assists/90, npxG/90 for the target player
- **Comparative radar**: target (green) vs. #1 replacement (blue) across 6 per-90 dimensions
- **Full scouting table**: all 5 replacements with stat columns, green-gradient similarity column
- **Similarity distribution histogram**: cosine similarity of all `n` players against the target, with the top-5 threshold marked

---

## Data Source & Filtering

**Source**: [FBRef](https://fbref.com) — Season 2024/25 European Football  
**File**: `PlayersFBREF.csv` (place in project root)

The preprocessing pipeline applies three filters before any model training:

```python
# 1. Remove goalkeepers — incommensurable statistical profile
df = df[~df["Position"].str.startswith("GB")]

# 2. Minimum 900 minutes — 10 full matches worth of data
df = df[df["Minutes"] >= 900]

# 3. Deduplicate — keep first occurrence per player name
df = df.drop_duplicates(subset=["Player"], keep="first")
```

These filters typically remove ~35% of raw rows, focusing the model on players for whom per-90 rate statistics are statistically meaningful.

**Feature matrix columns** (14 total after engineering):

```
Goals_Per_90_p90          Assists_Per_90_p90        Non-Penalty_Goals_Per_90_p90
xG_Per_90_p90             xAG_Per_90_p90             npxG_Per_90_p90
Progressive_Carries_p90   Progressive_Passes_p90     Progressive_Receives_p90
Yellow_Cards_p90          Red_Cards_p90
Pass_to_xG_Ratio          Carry_to_Pass_Ratio        Receive_to_Pass_Ratio
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- `PlayersFBREF.csv` in the project root (exported from FBRef 2024/25)

### Installation

```bash
git clone https://github.com/your-username/tacticaldna.git
cd tacticaldna

pip install -r requirements.txt
```

### Running the App

```bash
streamlit run futapp.py
```

The first launch fits the full pipeline (PCA → GMM → GridSearchCV RF). On a standard laptop CPU this takes **30–90 seconds**. Subsequent page navigations within the same session use Streamlit's `@st.cache_resource` cache — instant.

### Requirements

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

---

## Performance

| Metric | Value |
|---|---|
| RF Cross-Validated Accuracy (5-fold) | ~87–90% |
| Positional classes | AT / MT / DF |
| Archetype count | 8 (GMM) |
| PCA variance retained | 90% |
| PCA dimensions (typical) | 6–8 |
| Minimum minutes filter | 900 |
| Players typically validated | ~2,000–2,500 |

The `Cluster_ID` feature ranks in the **top-3 RF importances** in all tested runs — the single most important validation signal that the unsupervised GMM layer is capturing genuine structure, not noise.

---

## Intended Users

This tool is built for football practitioners, not data scientists. No coding is required to use the Streamlit interface.

| Who | How They Use It |
|---|---|
| **Head of Recruitment / Sporting Director** | Run the Replacement Engine with strict filter OFF to surface tactically equivalent players across all leagues before the transfer window opens |
| **Head Coach / First-Team Manager** | Query the Tactical Manifold to understand which archetype clusters opponents' key players belong to; prepare positional countermeasures |
| **Lead Scout / Regional Scout** | Generate long-lists by searching for replacements in lower leagues with identical PCA latent vectors to a departing senior player |
| **Performance Analyst** | Audit squad archetype distribution post-season; identify tactical gaps relative to the playing system |
| **Academy Director / U21 Coach** | Apply to youth league FBRef data; track which archetype academy players cluster into across seasons |
| **Football Agent** | Demonstrate a client's cosine similarity to higher-profile comparators to support contract and transfer negotiations |

---

## Known Limitations

These are **data constraints**, not model flaws. They are acknowledged transparently in the Methodology page of the app.

1. **No defensive metrics** — FBRef public data excludes tackles, interceptions, pressures, and aerial duels. The three derived ratio features partially compensate but some high-passing centre-backs will cluster with midfielders.

2. **Single-season snapshot** — Rate statistics for players with mid-season role changes or extended injuries will be noisy.

3. **No league-strength adjustment** — A prolific striker in League One and one in the Champions League may have similar per-90 vectors. The model does not currently adjust for competition quality.

4. **Hard cluster label at $k=8$** — The number of archetypes was chosen by domain knowledge. BIC/AIC optimisation across multiple $k$ values would be more rigorous.

5. **Training-set classification report** — The RF metrics shown in the app are on training data. A held-out test split would give a less optimistic accuracy estimate.

---

## Roadmap

- [ ] Integrate StatsBomb open-source event data for defensive metrics (pressures, tackles, interceptions)
- [ ] BIC/AIC-based automatic $k$ selection for GMM
- [ ] League-strength normalisation layer (UEFA coefficients / Elo)
- [ ] Multi-season temporal trajectories — track player archetype drift year-on-year
- [ ] Expose GMM posterior probabilities in the UI to visualise role ambiguity
- [ ] UMAP as an alternative/complement to PCA for non-linear manifold learning
- [ ] REST API layer for integration with video scouting platforms (Wyscout, SciSports)

---

## Academic Context

This project is submitted as a Master's Dissertation in Computer Science (Machine Learning specialisation). The **Methodology & Explainable AI** page of the app is designed to function as an interactive audit trail for academic review — every modelling decision is surfaced with its mathematical rationale, and the supervised RF validator exists specifically to provide an empirical proof-of-concept for the unsupervised clustering layer.

**Pipeline citation**:
```
Nair, P. (2025). TacticalDNA: Beyond Traditional Positions —
A Machine Learning Framework for Tactical Role Identification
in Association Football. MSc Dissertation, SVKM's NMIMS.
```

---

## License

MIT License — see `LICENSE` for details.

---

<div align="center">
  <sub>Built with scikit-learn · Streamlit · Plotly · FBRef data</sub><br/>
  <sub>⚽ <em>Football is a game of positions. Tactics is a game of roles.</em></sub>
</div>
