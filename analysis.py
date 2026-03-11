"""
Road Segment Clustering Analysis
Goal: Prove embedding-based clusters are better AAWDT predictors than Road Class.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import ast
import json

# -- sklearn ------------------------------------------------------------------─
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# -- third-party --------------------------------------------------------------─
import hdbscan
import umap
import xgboost as xgb
from scipy import stats

# -- output dir ----------------------------------------------------------------
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & PARSE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel("Combined_Features.xlsx")
print(f"Raw shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# -- locate the embedding column ----------------------------------------------─
emb_col = None
for c in df.columns:
    if "embed" in c.lower():
        emb_col = c
        break
if emb_col is None:
    # fallback: find column whose first non-null value looks like a list/array
    for c in df.columns:
        val = df[c].dropna().iloc[0]
        if isinstance(val, (list, np.ndarray)) or (isinstance(val, str) and val.strip().startswith("[")):
            emb_col = c
            break

print(f"Embedding column: '{emb_col}'")

# -- parse embeddings ----------------------------------------------------------
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

# drop rows where embedding parsing failed or AAWDT is null
mask = df["_emb"].notnull() & df["AAWDT"].notnull()
df = df[mask].reset_index(drop=True)
print(f"After filtering nulls: {len(df)} rows")

EMB = np.stack(df["_emb"].values).astype(np.float32)
print(f"Embedding matrix: {EMB.shape}")

# -- identify tabular feature columns ----------------------------------------─
TARGET = "AAWDT"
ROAD_CLASS_COL = "Road Class" if "Road Class" in df.columns else None
for candidate in ["RoadClass", "road_class", "road class", "ROAD_CLASS"]:
    if candidate in df.columns:
        ROAD_CLASS_COL = candidate
        break

TABULAR_FEATURES = []
candidates = ["Speed", "Lanes", "Region", "Lat", "Long",
              "Population2021", "PopPerSqKm2021",
              "Employment_rate", "Employment_Count",
              "Road_Segment_Type", "Year"]
for c in candidates:
    if c in df.columns:
        TABULAR_FEATURES.append(c)

print(f"Road Class column : '{ROAD_CLASS_COL}'")
print(f"Tabular features  : {TABULAR_FEATURES}")
print(f"Target            : {TARGET}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PRE-PROCESS TABULAR FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PRE-PROCESSING")
print("=" * 70)

# Label-encode any object columns; keep numerics
tab_df = df[TABULAR_FEATURES].copy()
for col in tab_df.columns:
    if tab_df[col].dtype == object:
        le = LabelEncoder()
        tab_df[col] = le.fit_transform(tab_df[col].astype(str))
    tab_df[col] = pd.to_numeric(tab_df[col], errors="coerce")

tab_df = tab_df.fillna(tab_df.median(numeric_only=True))

# Road Class one-hot
if ROAD_CLASS_COL:
    rc_dummies = pd.get_dummies(df[ROAD_CLASS_COL].astype(str), prefix="RC")
else:
    rc_dummies = pd.DataFrame(index=df.index)

y = df[TARGET].values.astype(np.float32)

print(f"Tabular feature matrix: {tab_df.shape}")
print(f"Road Class dummies: {rc_dummies.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  EMBEDDING NORMALISATION + UMAP REDUCTION FOR CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
EMB_norm = normalize(EMB, norm="l2")   # unit-length -> cosine == dot product

# UMAP to 20 dims for clustering — preserves local/global structure, kills the
# curse of dimensionality that makes 512-dim distances nearly uniform.
# Using a separate reducer from the 2D viz UMAP so each is tuned for its job.
UMAP_CLUST_DIMS = 20
print(f"Fitting UMAP for clustering ({UMAP_CLUST_DIMS}D) ...")
umap_clust = umap.UMAP(
    n_components=UMAP_CLUST_DIMS,
    metric="cosine",
    n_neighbors=15,
    min_dist=0.0,   # tighter packing -> better cluster separation
    random_state=SEED,
)
EMB_umap = umap_clust.fit_transform(EMB_norm)
print(f"UMAP clustering space: {EMB_umap.shape}")

# Normalise the UMAP output so K-Means still uses cosine-equivalent distances
EMB_umap_norm = normalize(EMB_umap, norm="l2")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CLUSTERING  (all algorithms on UMAP-reduced space)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CLUSTERING  (on UMAP-reduced embeddings)")
print("=" * 70)

# -- 4a  K-Means elbow --------------------------------------------------------
K_RANGE = range(2, 16)
inertias = []
print("K-Means elbow search ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(EMB_umap_norm)
    inertias.append(km.inertia_)
    print(f"  k={k:2d}  inertia={km.inertia_:.4f}")

# elbow via second-derivative (kneedle-like)
inertia_arr = np.array(inertias)
diffs1 = np.diff(inertia_arr)
diffs2 = np.diff(diffs1)
best_k = list(K_RANGE)[np.argmax(np.abs(diffs2)) + 1]
print(f"-> Elbow at k={best_k}")

# save elbow plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_RANGE), inertias, "o-", color="steelblue")
ax.axvline(best_k, color="crimson", linestyle="--", label=f"Elbow k={best_k}")
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Inertia (within-cluster SSE)")
ax.set_title("K-Means Elbow Curve (UMAP-reduced embeddings)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "01_kmeans_elbow.png", dpi=150)
plt.close()

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
km_labels = km_final.fit_predict(EMB_umap_norm)
print(f"K-Means ({best_k} clusters): {np.unique(km_labels, return_counts=True)}")

# -- 4b  HDBSCAN  (canonical McInnes pipeline: UMAP -> HDBSCAN) --------------
print("\nHDBSCAN ...")
hdb = hdbscan.HDBSCAN(
    min_cluster_size=max(5, len(df) // 50),
    min_samples=3,
    metric="euclidean",       # euclidean in UMAP space is meaningful
    cluster_selection_method="eom",
    core_dist_n_jobs=-1,
)
hdb_labels_raw = hdb.fit_predict(EMB_umap)   # raw UMAP (not re-normalised)
n_hdb = len(set(hdb_labels_raw)) - (1 if -1 in hdb_labels_raw else 0)
n_noise = (hdb_labels_raw == -1).sum()
print(f"HDBSCAN: {n_hdb} clusters, {n_noise} noise points ({100*n_noise/len(df):.1f}%)")

# remap noise (-1) to nearest cluster centroid
hdb_labels = hdb_labels_raw.copy()
if n_noise > 0 and n_hdb > 0:
    centroids = {c: EMB_umap[hdb_labels_raw == c].mean(axis=0)
                 for c in set(hdb_labels_raw) if c != -1}
    centroid_mat = np.stack([centroids[c] for c in sorted(centroids)])
    noise_idx = np.where(hdb_labels_raw == -1)[0]
    dists = np.linalg.norm(EMB_umap[noise_idx, None, :] - centroid_mat[None, :, :], axis=2)
    hdb_labels[noise_idx] = np.array(sorted(centroids))[dists.argmin(axis=1)]
    print(f"  Noise reassigned to nearest cluster centroid.")

print(f"HDBSCAN final labels: {np.unique(hdb_labels, return_counts=True)}")

# -- 4c  GMM  (UMAP space is low-dim enough for full covariance) --------------
print("\nGMM ...")
GMM_RANGE = range(2, 16)
bics = []
for g in GMM_RANGE:
    gm = GaussianMixture(n_components=g, random_state=SEED, covariance_type="full", max_iter=300)
    gm.fit(EMB_umap)
    bics.append(gm.bic(EMB_umap))
    print(f"  g={g:2d}  BIC={gm.bic(EMB_umap):.2f}")

best_g = list(GMM_RANGE)[np.argmin(bics)]
print(f"-> Best GMM components: {best_g}")

gmm_final = GaussianMixture(n_components=best_g, random_state=SEED, covariance_type="full", max_iter=500)
gmm_labels = gmm_final.fit_predict(EMB_umap)
print(f"GMM ({best_g} components): {np.unique(gmm_labels, return_counts=True)}")

# save BIC plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(GMM_RANGE), bics, "o-", color="darkorange")
ax.axvline(best_g, color="crimson", linestyle="--", label=f"Best g={best_g}")
ax.set_xlabel("Number of GMM components")
ax.set_ylabel("BIC (lower is better)")
ax.set_title("GMM Component Selection via BIC")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_gmm_bic.png", dpi=150)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  STATISTICAL EVALUATION — ANOVA / ETA-SQUARED
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANOVA / ETA-SQUARED")
print("=" * 70)

def anova_eta(labels, y, name):
    groups = [y[labels == c] for c in np.unique(labels) if len(y[labels == c]) >= 2]
    if len(groups) < 2:
        return {"name": name, "F": np.nan, "p": np.nan, "eta2": np.nan, "n_groups": len(groups)}
    F, p = stats.f_oneway(*groups)
    # eta-squared = SS_between / SS_total
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    print(f"  {name:30s}  F={F:.2f}  p={p:.2e}  eta^2={eta2:.4f}  n_groups={len(groups)}")
    return {"name": name, "F": F, "p": p, "eta2": eta2, "n_groups": len(groups)}

anova_results = []
anova_results.append(anova_eta(km_labels,  y, f"K-Means (k={best_k})"))
anova_results.append(anova_eta(hdb_labels, y, f"HDBSCAN ({n_hdb} clusters)"))
anova_results.append(anova_eta(gmm_labels, y, f"GMM (g={best_g})"))

if ROAD_CLASS_COL:
    rc_enc = LabelEncoder().fit_transform(df[ROAD_CLASS_COL].astype(str))
    anova_results.append(anova_eta(rc_enc, y, "Road Class (original)"))

anova_df = pd.DataFrame(anova_results).set_index("name")
print("\nANOVA Summary:")
print(anova_df.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ML MODEL SHOOTOUT — XGBoost with cross-validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ML MODEL SHOOTOUT (XGBoost, 5-fold CV)")
print("=" * 70)

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

def cluster_dummies(labels, prefix):
    return pd.get_dummies(pd.Series(labels, name=prefix).astype(str), prefix=prefix)

def xgb_cv_scores(X, y, label):
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    r2_scores, rmse_scores = [], []
    for train_idx, val_idx in CV.split(X):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        pred = model.predict(Xval)
        r2_scores.append(r2_score(yval, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(yval, pred)))
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    print(f"  {label:45s}  R^2={r2_mean:.4f}±{r2_std:.4f}  RMSE={rmse_mean:.1f}±{rmse_std:.1f}")
    return {
        "Feature Set": label,
        "R2_mean": r2_mean, "R2_std": r2_std,
        "RMSE_mean": rmse_mean, "RMSE_std": rmse_std,
    }

BASE = tab_df.values.astype(np.float32)

# cluster dummies for each algorithm
km_dum  = cluster_dummies(km_labels,  "KM").values.astype(np.float32)
hdb_dum = cluster_dummies(hdb_labels, "HDB").values.astype(np.float32)
gmm_dum = cluster_dummies(gmm_labels, "GMM").values.astype(np.float32)
rc_dum  = rc_dummies.values.astype(np.float32)

feature_sets = {
    "Baseline (tabular only)":           BASE,
    "+ Road Class":                       np.hstack([BASE, rc_dum]),
    "+ K-Means clusters":                 np.hstack([BASE, km_dum]),
    "+ HDBSCAN clusters":                 np.hstack([BASE, hdb_dum]),
    "+ GMM clusters":                     np.hstack([BASE, gmm_dum]),
    "+ Road Class + K-Means":             np.hstack([BASE, rc_dum, km_dum]),
    "+ Road Class + HDBSCAN":             np.hstack([BASE, rc_dum, hdb_dum]),
    "+ Road Class + GMM":                 np.hstack([BASE, rc_dum, gmm_dum]),
}

ml_results = []
for label, X in feature_sets.items():
    ml_results.append(xgb_cv_scores(X, y, label))

ml_df = pd.DataFrame(ml_results).set_index("Feature Set").sort_values("R2_mean", ascending=False)
print("\nML Results (ranked by R^2):")
print(ml_df.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  UMAP VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("UMAP PROJECTION")
print("=" * 70)

print("Fitting UMAP (this may take a moment) ...")
reducer = umap.UMAP(n_components=2, metric="cosine", random_state=SEED, n_neighbors=15, min_dist=0.1)
emb2d = reducer.fit_transform(EMB_norm)
print("UMAP done.")

def plot_umap_by_cluster(emb2d, labels, title, fname, cmap="tab20"):
    unique = sorted(set(labels))
    n_cls = len(unique)
    cmap_obj = plt.get_cmap(cmap, n_cls)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(unique):
        idx = labels == c
        ax.scatter(emb2d[idx, 0], emb2d[idx, 1],
                   s=8, alpha=0.6, color=cmap_obj(i), label=f"Cluster {c}")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if n_cls <= 20:
        ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close()

def plot_umap_by_aawdt(emb2d, y, title, fname):
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1],
                    c=y, cmap="plasma", s=8, alpha=0.7,
                    norm=mcolors.LogNorm(vmin=max(1, y.min()), vmax=y.max()))
    plt.colorbar(sc, ax=ax, label="AAWDT (log scale)")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150)
    plt.close()

# K-Means
plot_umap_by_cluster(emb2d, km_labels,  f"UMAP — K-Means (k={best_k})",      "03_umap_kmeans_clusters.png")
plot_umap_by_aawdt(  emb2d, y,           "UMAP — Coloured by AAWDT",           "04_umap_aawdt.png")

# HDBSCAN
plot_umap_by_cluster(emb2d, hdb_labels, f"UMAP — HDBSCAN ({n_hdb} clusters)", "05_umap_hdbscan_clusters.png")

# GMM
plot_umap_by_cluster(emb2d, gmm_labels, f"UMAP — GMM (g={best_g})",           "06_umap_gmm_clusters.png")

print("UMAP plots saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MEAN AAWDT PER CLUSTER (bar charts with error bars)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBar charts — mean AAWDT per cluster ...")

def bar_aawdt(labels, y, title, fname):
    stats_rows = []
    for c in sorted(set(labels)):
        g = y[labels == c]
        stats_rows.append({
            "cluster": c, "mean": g.mean(),
            "sem": stats.sem(g) if len(g) > 1 else 0,
            "n": len(g)
        })
    sdf = pd.DataFrame(stats_rows).sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(7, len(sdf) * 0.6), 5))
    bars = ax.bar(sdf["cluster"].astype(str), sdf["mean"],
                  yerr=sdf["sem"], capsize=4,
                  color=plt.get_cmap("tab20")(np.linspace(0, 1, len(sdf))),
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean AAWDT")
    ax.set_title(title)
    for bar, row in zip(bars, sdf.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sdf["sem"].max() * 0.1,
                f"n={row.n}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150)
    plt.close()

bar_aawdt(km_labels,  y, f"Mean AAWDT per K-Means cluster (k={best_k})",      "07_bar_kmeans.png")
bar_aawdt(hdb_labels, y, f"Mean AAWDT per HDBSCAN cluster ({n_hdb} clusters)", "08_bar_hdbscan.png")
bar_aawdt(gmm_labels, y, f"Mean AAWDT per GMM cluster (g={best_g})",           "09_bar_gmm.png")

if ROAD_CLASS_COL:
    rc_str = df[ROAD_CLASS_COL].astype(str).values
    rc_enc2 = LabelEncoder().fit_transform(rc_str)
    bar_aawdt(rc_enc2, y, "Mean AAWDT per Road Class",                         "10_bar_roadclass.png")

print("Bar charts saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  R^2 HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\nR^2 heatmap ...")

# reshape into 2D: rows = models, cols = feature sets (already flat here, so fake 2-axis)
# present as model × feature-group matrix
heatmap_data = ml_df[["R2_mean"]].T
fig, ax = plt.subplots(figsize=(max(10, len(ml_df) * 1.4), 3))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".4f",
    cmap="YlGnBu", vmin=0, vmax=1,
    linewidths=0.5, linecolor="gray",
    ax=ax
)
ax.set_title("R^2 (5-fold CV) — XGBoost Model × Feature Set\n(higher = better)", fontsize=12)
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "11_r2_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Heatmap saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  CLEAN SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)

summary = ml_df.reset_index().rename(columns={
    "Feature Set": "Feature Set",
    "R2_mean": "R^2",
    "R2_std": "R^2 std",
    "RMSE_mean": "RMSE",
    "RMSE_std": "RMSE std",
})
summary["Rank"] = range(1, len(summary) + 1)
summary = summary[["Rank", "Feature Set", "R^2", "R^2 std", "RMSE", "RMSE std"]]
summary["R^2"] = summary["R^2"].round(4)
summary["R^2 std"] = summary["R^2 std"].round(4)
summary["RMSE"] = summary["RMSE"].round(1)
summary["RMSE std"] = summary["RMSE std"].round(1)

print(summary.to_string(index=False))

# -- print the key insight ------------------------------------------------------
rc_r2   = ml_df.loc["+ Road Class",    "R2_mean"] if "+ Road Class" in ml_df.index else np.nan
km_r2   = ml_df.loc["+ K-Means clusters",  "R2_mean"] if "+ K-Means clusters" in ml_df.index else np.nan
hdb_r2  = ml_df.loc["+ HDBSCAN clusters",  "R2_mean"] if "+ HDBSCAN clusters" in ml_df.index else np.nan
gmm_r2  = ml_df.loc["+ GMM clusters",      "R2_mean"] if "+ GMM clusters" in ml_df.index else np.nan
base_r2 = ml_df.loc["Baseline (tabular only)", "R2_mean"]

best_cluster_r2   = max(km_r2, hdb_r2, gmm_r2)
best_cluster_name = {km_r2: "K-Means", hdb_r2: "HDBSCAN", gmm_r2: "GMM"}[best_cluster_r2]

print("\n-- Key Insight --------------------------------------------------------─")
print(f"  Baseline R^2         : {base_r2:.4f}")
print(f"  + Road Class R^2     : {rc_r2:.4f}  (delta = {rc_r2 - base_r2:+.4f})")
print(f"  Best cluster method : {best_cluster_name}  R^2={best_cluster_r2:.4f}  (delta = {best_cluster_r2 - base_r2:+.4f})")
beats = "BEATS" if best_cluster_r2 > rc_r2 else "does NOT beat"
print(f"  -> Embedding clusters {beats} Road Class by deltaR^2={best_cluster_r2 - rc_r2:+.4f}")

print("\n-- ANOVA Eta-Squared (variance explained by group membership) ----------─")
print(anova_df[["eta2", "F", "n_groups"]].to_string())

# save tables to CSV
summary.to_csv(OUT / "summary_ml.csv", index=False)
anova_df.to_csv(OUT / "summary_anova.csv")
print(f"\nAll outputs saved to: {OUT.resolve()}")
print("Done.")
