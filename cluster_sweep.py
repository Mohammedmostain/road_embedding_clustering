"""
Cluster Count Sweep
====================
For each clustering method, sweep over the number of clusters and measure:
  - Standalone R2  : tabular + cluster dummies (no Road Class)
  - Combined R2    : tabular + Road Class + cluster dummies
  - Benefit        : Combined R2 - Road Class-only R2

Methods:
  K-Means  : k = 2 .. 20
  HDBSCAN  : min_cluster_size swept from 5..40, cluster count varies automatically
  GMM      : g = 2 .. 20
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ast, json
from pathlib import Path

from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import hdbscan
import umap
import xgboost as xgb

OUT  = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA  (same as analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_excel("Combined_Features.xlsx")

emb_col = next((c for c in df.columns if "embed" in c.lower()), None)

def parse_embedding(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        x = x.strip()
        try:
            return np.array(json.loads(x), dtype=np.float32)
        except Exception:
            pass
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception:
            pass
    return None

df["_emb"] = df[emb_col].apply(parse_embedding)
df = df[df["_emb"].notnull() & df["AAWDT"].notnull()].reset_index(drop=True)
print(f"  {len(df)} rows after filtering")

EMB = np.stack(df["_emb"].values).astype(np.float32)
y   = df["AAWDT"].values.astype(np.float32)

ROAD_CLASS_COL = next(
    (c for c in df.columns if c.lower().replace(" ", "_") == "road_class"),
    None
)
TABULAR_FEATURES = [c for c in ["Speed","Lanes","Region","Lat","Long",
    "Population2021","PopPerSqKm2021","Employment_rate","Employment_Count",
    "Road_Segment_Type","Year"] if c in df.columns]

# ─────────────────────────────────────────────────────────────────────────────
# 2. PRE-PROCESS TABULAR + ROAD CLASS
# ─────────────────────────────────────────────────────────────────────────────
tab_df = df[TABULAR_FEATURES].copy()
for col in tab_df.columns:
    if tab_df[col].dtype == object:
        tab_df[col] = LabelEncoder().fit_transform(tab_df[col].astype(str))
    tab_df[col] = pd.to_numeric(tab_df[col], errors="coerce")
tab_df = tab_df.fillna(tab_df.median(numeric_only=True))
BASE = tab_df.values.astype(np.float32)

rc_dum = pd.get_dummies(df[ROAD_CLASS_COL].astype(str), prefix="RC").values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3. UMAP REDUCTION  (fixed — same as analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Fitting UMAP for clustering (20D) ...")
EMB_norm = normalize(EMB, norm="l2")
EMB_umap = umap.UMAP(
    n_components=20, metric="cosine", n_neighbors=15,
    min_dist=0.0, random_state=SEED
).fit_transform(EMB_norm)
EMB_umap_norm = normalize(EMB_umap, norm="l2")
print("  UMAP done.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost CV helper
# ─────────────────────────────────────────────────────────────────────────────
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

def xgb_r2(X):
    """5-fold CV R2 for XGBoost regressor."""
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=0, eval_metric="rmse"
    )
    scores = []
    for tr, val in CV.split(X):
        model.fit(X[tr], y[tr],
                  eval_set=[(X[val], y[val])], verbose=False)
        scores.append(r2_score(y[val], model.predict(X[val])))
    return float(np.mean(scores))

# Road Class baseline (fixed reference point)
print("Computing Road Class baseline R2 ...")
RC_BASELINE = xgb_r2(np.hstack([BASE, rc_dum]))
print(f"  Road Class baseline R2 = {RC_BASELINE:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. NOISE REASSIGNMENT helper for HDBSCAN
# ─────────────────────────────────────────────────────────────────────────────
def reassign_noise(labels, space):
    """Reassign noise points (-1) to the nearest cluster centroid."""
    labels = labels.copy()
    noise  = np.where(labels == -1)[0]
    if len(noise) == 0:
        return labels
    unique_c   = [c for c in sorted(set(labels)) if c != -1]
    centroids  = np.array([space[labels == c].mean(axis=0) for c in unique_c])
    dists      = np.linalg.norm(space[noise, None, :] - centroids[None, :, :], axis=2)
    labels[noise] = np.array(unique_c)[dists.argmin(axis=1)]
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# 6. SWEEP K-MEANS  k = 2 .. 20
# ─────────────────────────────────────────────────────────────────────────────
print("\nSweeping K-Means k = 2 .. 20 ...")
km_rows = []
for k in range(2, 21):
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(EMB_umap_norm)
    dum    = pd.get_dummies(labels, prefix="KM").values.astype(np.float32)

    r2_standalone = xgb_r2(np.hstack([BASE, dum]))
    r2_combined   = xgb_r2(np.hstack([BASE, rc_dum, dum]))
    benefit       = r2_combined - RC_BASELINE

    km_rows.append({"k": k, "standalone_r2": r2_standalone,
                    "combined_r2": r2_combined, "benefit": benefit})
    print(f"  k={k:2d}  standalone={r2_standalone:.4f}  combined={r2_combined:.4f}  benefit={benefit:+.4f}")

km_df = pd.DataFrame(km_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 7. SWEEP HDBSCAN  min_cluster_size = 5 .. 40
#    (cluster count varies — we record the actual n_clusters found)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSweeping HDBSCAN min_cluster_size = 5 .. 40 ...")
hdb_rows = []
seen_n = set()   # skip duplicate cluster counts to avoid redundant fits
for mcs in range(5, 41):
    raw = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=3,
        metric="euclidean", cluster_selection_method="eom",
        core_dist_n_jobs=-1
    ).fit_predict(EMB_umap)
    labels  = reassign_noise(raw, EMB_umap)
    n_clust = len(set(labels))

    if n_clust in seen_n:
        continue   # same structure as a previous mcs, skip
    seen_n.add(n_clust)

    dum = pd.get_dummies(labels, prefix="HDB").values.astype(np.float32)
    r2_standalone = xgb_r2(np.hstack([BASE, dum]))
    r2_combined   = xgb_r2(np.hstack([BASE, rc_dum, dum]))
    benefit       = r2_combined - RC_BASELINE

    hdb_rows.append({"mcs": mcs, "n_clusters": n_clust,
                     "standalone_r2": r2_standalone,
                     "combined_r2": r2_combined, "benefit": benefit})
    print(f"  mcs={mcs:2d}  n_clusters={n_clust:2d}  standalone={r2_standalone:.4f}  "
          f"combined={r2_combined:.4f}  benefit={benefit:+.4f}")

hdb_df = pd.DataFrame(hdb_rows).sort_values("n_clusters")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SWEEP GMM  g = 2 .. 20
# ─────────────────────────────────────────────────────────────────────────────
print("\nSweeping GMM g = 2 .. 20 ...")
gmm_rows = []
for g in range(2, 31):
    labels = GaussianMixture(
        n_components=g, covariance_type="full",
        max_iter=300, random_state=SEED
    ).fit_predict(EMB_umap)
    dum = pd.get_dummies(labels, prefix="GMM").values.astype(np.float32)

    r2_standalone = xgb_r2(np.hstack([BASE, dum]))
    r2_combined   = xgb_r2(np.hstack([BASE, rc_dum, dum]))
    benefit       = r2_combined - RC_BASELINE

    gmm_rows.append({"g": g, "standalone_r2": r2_standalone,
                     "combined_r2": r2_combined, "benefit": benefit})
    print(f"  g={g:2d}  standalone={r2_standalone:.4f}  combined={r2_combined:.4f}  benefit={benefit:+.4f}")

gmm_df = pd.DataFrame(gmm_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
km_df.to_csv(OUT / "sweep_kmeans.csv",  index=False)
hdb_df.to_csv(OUT / "sweep_hdbscan.csv", index=False)
gmm_df.to_csv(OUT / "sweep_gmm.csv",    index=False)
print("\nCSVs saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT — 3 rows × 2 cols  (standalone R2 | benefit)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 13))
fig.suptitle(
    f"Cluster Count Sweep — Downstream AAWDT Prediction\n"
    f"Road Class baseline R² = {RC_BASELINE:.3f}  (dashed line)",
    fontsize=13, y=1.01
)

methods = [
    ("K-Means",  km_df,  "k",          "steelblue"),
    ("HDBSCAN",  hdb_df, "n_clusters",  "darkorange"),
    ("GMM",      gmm_df, "g",           "mediumseagreen"),
]

for row, (name, df_m, xcol, col) in enumerate(methods):
    x = df_m[xcol]

    # Left: standalone and combined R2
    ax = axes[row, 0]
    ax.plot(x, df_m["standalone_r2"], "o-", color=col, label="Clusters only")
    ax.plot(x, df_m["combined_r2"],   "s--", color=col, alpha=0.6, label="Road Class + Clusters")
    ax.axhline(RC_BASELINE, color="crimson", linestyle=":", linewidth=1.5, label="Road Class only")
    ax.set_title(f"{name} — R²")
    ax.set_xlabel("Number of clusters" if xcol != "k" else "k")
    ax.set_ylabel("R² (5-fold CV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: benefit over Road Class baseline
    ax = axes[row, 1]
    best_idx  = df_m["benefit"].idxmax()
    best_x    = df_m.loc[best_idx, xcol]
    best_ben  = df_m.loc[best_idx, "benefit"]

    bars = ax.bar(x, df_m["benefit"], color=col, alpha=0.75, width=0.6)
    # colour negative bars red
    for bar, val in zip(bars, df_m["benefit"]):
        bar.set_color("crimson" if val < 0 else col)
        bar.set_alpha(0.75)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(best_x, color="black", linestyle="--", linewidth=1,
               label=f"Best: {xcol}={best_x} (+{best_ben:.4f})")
    ax.set_title(f"{name} — Benefit over Road Class baseline")
    ax.set_xlabel("Number of clusters" if xcol != "k" else "k")
    ax.set_ylabel("Delta R²  (combined - baseline)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(OUT / "sweep_all_methods.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: outputs/sweep_all_methods.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("BEST CONFIGURATIONS PER METHOD")
print("=" * 65)
print(f"Road Class baseline R2 : {RC_BASELINE:.4f}")
print()

for name, df_m, xcol in [("K-Means",  km_df,  "k"),
                          ("HDBSCAN", hdb_df, "n_clusters"),
                          ("GMM",     gmm_df, "g")]:
    best = df_m.loc[df_m["benefit"].idxmax()]
    print(f"{name}")
    print(f"  Best {xcol:<12} = {int(best[xcol])}")
    print(f"  Standalone R2  = {best['standalone_r2']:.4f}")
    print(f"  Combined R2    = {best['combined_r2']:.4f}")
    print(f"  Benefit        = {best['benefit']:+.4f}")
    print()
