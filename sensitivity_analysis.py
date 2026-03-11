"""
Sensitivity Analysis — Road Segment Clustering
Tests robustness of UMAP + clustering pipeline across hyperparameter configurations.

What we vary:
  - UMAP n_components: [10, 15, 20, 30, 50]
  - UMAP n_neighbors:  [5, 10, 15, 30]
  - K-Means seeds:     10 different seeds (label stability via NMI)
  - HDBSCAN min_cluster_size: [5, 10, 14, 20, 30]
  - Bootstrap stability: 50 resamples on best config

Outputs:
  - outputs/sens_umap_dims.png       - eta2 / R2 across n_components
  - outputs/sens_umap_neighbors.png  - eta2 / R2 across n_neighbors
  - outputs/sens_kmeans_seeds.png    - NMI spread across 10 seeds
  - outputs/sens_hdbscan_mcs.png     - clusters / eta2 across min_cluster_size
  - outputs/sens_bootstrap.png       - bootstrap R2 distribution
  - outputs/sensitivity_results.csv  - full numeric results table
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import ast, json

from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, normalized_mutual_info_score
from scipy import stats
import hdbscan
import umap
import xgboost as xgb

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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

def eta_squared(labels, y):
    groups = [y[labels == c] for c in np.unique(labels) if len(y[labels == c]) >= 2]
    if len(groups) < 2:
        return np.nan
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total if ss_total > 0 else np.nan

def xgb_r2(X, y, n_splits=5, seed=SEED):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0, n_jobs=-1,
    )
    scores = []
    for tr, val in cv.split(X):
        model.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        scores.append(r2_score(y[val], model.predict(X[val])))
    return np.mean(scores), np.std(scores)

def best_kmeans_k(data, k_range=range(2, 14), seed=SEED):
    inertias = [KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data).inertia_
                for k in k_range]
    d2 = np.diff(np.diff(inertias))
    return list(k_range)[np.argmax(np.abs(d2)) + 1]

def run_hdbscan(umap_data, min_cluster_size):
    h = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=3,
        metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1,
    )
    raw = h.fit_predict(umap_data)
    labels = raw.copy()
    noise_idx = np.where(raw == -1)[0]
    valid = set(raw) - {-1}
    if len(noise_idx) > 0 and len(valid) > 0:
        centroids = {c: umap_data[raw == c].mean(axis=0) for c in valid}
        cmat = np.stack([centroids[c] for c in sorted(centroids)])
        dists = np.linalg.norm(umap_data[noise_idx, None, :] - cmat[None, :, :], axis=2)
        labels[noise_idx] = np.array(sorted(centroids))[dists.argmin(axis=1)]
    return labels, len(valid)

def cluster_dummies(labels, prefix):
    return pd.get_dummies(pd.Series(labels).astype(str), prefix=prefix).values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (same as analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel("Combined_Features.xlsx")
emb_col = next((c for c in df.columns if "embed" in c.lower()), None)
df["_emb"] = df[emb_col].apply(parse_embedding)
df = df[df["_emb"].notnull() & df["AAWDT"].notnull()].reset_index(drop=True)

EMB = np.stack(df["_emb"].values).astype(np.float32)
EMB_norm = normalize(EMB, norm="l2")
y = df["AAWDT"].values.astype(np.float32)

ROAD_CLASS_COL = next((c for c in df.columns if "road" in c.lower() and "class" in c.lower()), None)
TABULAR_FEATURES = [c for c in ["Speed", "Lanes", "Region", "Lat", "Long",
                                  "Population2021", "PopPerSqKm2021",
                                  "Employment_rate", "Employment_Count",
                                  "Road_Segment_Type", "Year"] if c in df.columns]

tab_df = df[TABULAR_FEATURES].copy()
for col in tab_df.columns:
    if tab_df[col].dtype == object:
        tab_df[col] = LabelEncoder().fit_transform(tab_df[col].astype(str))
    tab_df[col] = pd.to_numeric(tab_df[col], errors="coerce")
tab_df = tab_df.fillna(tab_df.median(numeric_only=True))
BASE = tab_df.values.astype(np.float32)

rc_dum = pd.get_dummies(df[ROAD_CLASS_COL].astype(str), prefix="RC").values.astype(np.float32) \
         if ROAD_CLASS_COL else np.zeros((len(df), 0), dtype=np.float32)

print(f"  {len(df)} segments | {EMB.shape[1]}-dim embeddings | {len(TABULAR_FEATURES)} tabular features")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — UMAP n_components sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 1: UMAP n_components sweep [10, 15, 20, 30, 50]")
print("=" * 70)

dims_to_test = [10, 15, 20, 30, 50]
dim_rows = []

for n_dim in dims_to_test:
    print(f"  n_components={n_dim} ...", end=" ", flush=True)

    umap_r = umap.UMAP(n_components=n_dim, metric="cosine",
                        n_neighbors=15, min_dist=0.0, random_state=SEED)
    umap_data = umap_r.fit_transform(EMB_norm)
    umap_norm = normalize(umap_data, norm="l2")

    k = best_kmeans_k(umap_norm)
    km_labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(umap_norm)
    km_eta = eta_squared(km_labels, y)
    km_r2, _ = xgb_r2(np.hstack([BASE, cluster_dummies(km_labels, "KM")]), y)

    hdb_labels, n_hdb = run_hdbscan(umap_data, max(5, len(df)//50))
    hdb_eta = eta_squared(hdb_labels, y)
    hdb_r2, _ = xgb_r2(np.hstack([BASE, cluster_dummies(hdb_labels, "HDB")]), y)

    dim_rows.append({"n_components": n_dim,
                     "km_k": k, "km_eta2": round(km_eta, 4), "km_r2": round(km_r2, 4),
                     "hdb_n": n_hdb, "hdb_eta2": round(hdb_eta, 4), "hdb_r2": round(hdb_r2, 4)})
    print(f"KM k={k} eta2={km_eta:.3f} R2={km_r2:.3f} | "
          f"HDB n={n_hdb} eta2={hdb_eta:.3f} R2={hdb_r2:.3f}")

dim_df = pd.DataFrame(dim_rows)
print("\n", dim_df.to_string(index=False))

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in zip(axes,
                               [("km_eta2","hdb_eta2"), ("km_r2","hdb_r2")],
                               ["eta-squared by n_components", "XGBoost R² by n_components"]):
    ax.plot(dim_df["n_components"], dim_df[metric[0]], "o-", label="K-Means", color="steelblue")
    ax.plot(dim_df["n_components"], dim_df[metric[1]], "s--", label="HDBSCAN", color="darkorange")
    ax.axvline(20, color="crimson", linestyle=":", alpha=0.7, label="Baseline (n=20)")
    ax.set_xlabel("UMAP n_components")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
plt.suptitle("Sensitivity: UMAP Dimensionality", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "sens_umap_dims.png", dpi=150)
plt.close()
print("Saved: sens_umap_dims.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — UMAP n_neighbors sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 2: UMAP n_neighbors sweep [5, 10, 15, 30]")
print("=" * 70)

neighbors_to_test = [5, 10, 15, 30]
nn_rows = []

for nn in neighbors_to_test:
    print(f"  n_neighbors={nn} ...", end=" ", flush=True)

    umap_r = umap.UMAP(n_components=20, metric="cosine",
                        n_neighbors=nn, min_dist=0.0, random_state=SEED)
    umap_data = umap_r.fit_transform(EMB_norm)
    umap_norm = normalize(umap_data, norm="l2")

    k = best_kmeans_k(umap_norm)
    km_labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(umap_norm)
    km_eta = eta_squared(km_labels, y)
    km_r2, _ = xgb_r2(np.hstack([BASE, cluster_dummies(km_labels, "KM")]), y)

    hdb_labels, n_hdb = run_hdbscan(umap_data, max(5, len(df)//50))
    hdb_eta = eta_squared(hdb_labels, y)
    hdb_r2, _ = xgb_r2(np.hstack([BASE, cluster_dummies(hdb_labels, "HDB")]), y)

    nn_rows.append({"n_neighbors": nn,
                    "km_k": k, "km_eta2": round(km_eta, 4), "km_r2": round(km_r2, 4),
                    "hdb_n": n_hdb, "hdb_eta2": round(hdb_eta, 4), "hdb_r2": round(hdb_r2, 4)})
    print(f"KM k={k} eta2={km_eta:.3f} R2={km_r2:.3f} | "
          f"HDB n={n_hdb} eta2={hdb_eta:.3f} R2={hdb_r2:.3f}")

nn_df = pd.DataFrame(nn_rows)
print("\n", nn_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in zip(axes,
                               [("km_eta2","hdb_eta2"), ("km_r2","hdb_r2")],
                               ["eta-squared by n_neighbors", "XGBoost R² by n_neighbors"]):
    ax.plot(nn_df["n_neighbors"], nn_df[metric[0]], "o-", label="K-Means", color="steelblue")
    ax.plot(nn_df["n_neighbors"], nn_df[metric[1]], "s--", label="HDBSCAN", color="darkorange")
    ax.axvline(15, color="crimson", linestyle=":", alpha=0.7, label="Baseline (n=15)")
    ax.set_xlabel("UMAP n_neighbors")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
plt.suptitle("Sensitivity: UMAP n_neighbors", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "sens_umap_neighbors.png", dpi=150)
plt.close()
print("Saved: sens_umap_neighbors.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — K-Means seed stability (NMI between runs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 3: K-Means seed stability (10 seeds, NMI vs seed=42)")
print("=" * 70)

# Use baseline config: 20-dim UMAP, n_neighbors=15
umap_base = umap.UMAP(n_components=20, metric="cosine",
                       n_neighbors=15, min_dist=0.0, random_state=SEED)
umap_base_data = umap_base.fit_transform(EMB_norm)
umap_base_norm = normalize(umap_base_data, norm="l2")

baseline_k = best_kmeans_k(umap_base_norm)
ref_labels = KMeans(n_clusters=baseline_k, random_state=42, n_init=20).fit_predict(umap_base_norm)

seed_rows = []
seeds = [0, 1, 7, 13, 21, 42, 99, 123, 256, 999]
for s in seeds:
    labels_s = KMeans(n_clusters=baseline_k, random_state=s, n_init=20).fit_predict(umap_base_norm)
    nmi = normalized_mutual_info_score(ref_labels, labels_s)
    eta = eta_squared(labels_s, y)
    seed_rows.append({"seed": s, "NMI_vs_42": round(nmi, 4), "eta2": round(eta, 4)})
    print(f"  seed={s:4d}  NMI={nmi:.4f}  eta2={eta:.4f}")

seed_df = pd.DataFrame(seed_rows)
print(f"\n  Mean NMI={seed_df['NMI_vs_42'].mean():.4f}  std={seed_df['NMI_vs_42'].std():.4f}")
print(f"  Mean eta2={seed_df['eta2'].mean():.4f}  std={seed_df['eta2'].std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(seed_df["seed"].astype(str), seed_df["NMI_vs_42"], color="steelblue", edgecolor="black")
axes[0].axhline(1.0, color="crimson", linestyle="--", alpha=0.5, label="Perfect agreement")
axes[0].set_xlabel("Random seed")
axes[0].set_ylabel("NMI vs seed=42")
axes[0].set_title(f"K-Means label stability (k={baseline_k})")
axes[0].legend()
axes[1].bar(seed_df["seed"].astype(str), seed_df["eta2"], color="darkorange", edgecolor="black")
axes[1].set_xlabel("Random seed")
axes[1].set_ylabel("eta-squared")
axes[1].set_title("AAWDT grouping quality across seeds")
plt.suptitle("Sensitivity: K-Means Random Seed", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "sens_kmeans_seeds.png", dpi=150)
plt.close()
print("Saved: sens_kmeans_seeds.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4 — HDBSCAN min_cluster_size sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 4: HDBSCAN min_cluster_size sweep [5, 10, 14, 20, 30]")
print("=" * 70)

mcs_values = [5, 10, 14, 20, 30]
mcs_rows = []

for mcs in mcs_values:
    labels, n_cls = run_hdbscan(umap_base_data, mcs)
    noise_pct = (labels == -1).sum() / len(df) * 100  # after reassignment should be 0
    eta = eta_squared(labels, y)
    r2, r2_std = xgb_r2(np.hstack([BASE, cluster_dummies(labels, "HDB")]), y)
    mcs_rows.append({"min_cluster_size": mcs, "n_clusters": n_cls,
                     "eta2": round(eta, 4), "r2": round(r2, 4), "r2_std": round(r2_std, 4)})
    print(f"  mcs={mcs:2d}  n_clusters={n_cls:2d}  eta2={eta:.4f}  R2={r2:.4f}")

mcs_df = pd.DataFrame(mcs_rows)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(mcs_df["min_cluster_size"], mcs_df["n_clusters"], "o-", color="purple")
axes[0].axvline(14, color="crimson", linestyle=":", alpha=0.7, label="Baseline (mcs=14)")
axes[0].set_xlabel("min_cluster_size")
axes[0].set_ylabel("Number of clusters")
axes[0].set_title("Cluster count")
axes[0].legend()

axes[1].plot(mcs_df["min_cluster_size"], mcs_df["eta2"], "o-", color="steelblue")
axes[1].axvline(14, color="crimson", linestyle=":", alpha=0.7)
axes[1].set_xlabel("min_cluster_size")
axes[1].set_ylabel("eta-squared")
axes[1].set_title("AAWDT grouping quality")
axes[1].set_ylim(bottom=0)

axes[2].errorbar(mcs_df["min_cluster_size"], mcs_df["r2"], yerr=mcs_df["r2_std"],
                  fmt="o-", color="darkorange", capsize=4)
axes[2].axvline(14, color="crimson", linestyle=":", alpha=0.7)
axes[2].set_xlabel("min_cluster_size")
axes[2].set_ylabel("XGBoost R²")
axes[2].set_title("Predictive performance")
axes[2].set_ylim(bottom=0)

plt.suptitle("Sensitivity: HDBSCAN min_cluster_size", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "sens_hdbscan_mcs.png", dpi=150)
plt.close()
print("Saved: sens_hdbscan_mcs.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 5 — Bootstrap stability (best config: 20-dim, nn=15, baseline k)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 5: Bootstrap stability (50 resamples, best config)")
print("=" * 70)

N_BOOTSTRAP = 50
boot_r2_km, boot_r2_hdb, boot_r2_rc = [], [], []

for i in range(N_BOOTSTRAP):
    rng = np.random.RandomState(i)
    idx = rng.choice(len(df), len(df), replace=True)

    emb_b = EMB_norm[idx]
    y_b   = y[idx]
    base_b = BASE[idx]
    rc_b   = rc_dum[idx]

    # UMAP on bootstrap sample
    ur = umap.UMAP(n_components=20, metric="cosine",
                    n_neighbors=15, min_dist=0.0, random_state=i)
    umap_b = ur.fit_transform(emb_b)
    umap_bn = normalize(umap_b, norm="l2")

    # K-Means
    k_b = best_kmeans_k(umap_bn, seed=i)
    km_b = KMeans(n_clusters=k_b, random_state=i, n_init=10).fit_predict(umap_bn)
    km_dum_b = cluster_dummies(km_b, "KM")

    # HDBSCAN
    hdb_b, _ = run_hdbscan(umap_b, max(5, len(df)//50))
    hdb_dum_b = cluster_dummies(hdb_b, "HDB")

    # XGBoost R2 (3-fold on bootstrap to keep it fast)
    r2_km,  _ = xgb_r2(np.hstack([base_b, km_dum_b]),  y_b, n_splits=3, seed=i)
    r2_hdb, _ = xgb_r2(np.hstack([base_b, hdb_dum_b]), y_b, n_splits=3, seed=i)
    r2_rc,  _ = xgb_r2(np.hstack([base_b, rc_b]),       y_b, n_splits=3, seed=i)

    boot_r2_km.append(r2_km)
    boot_r2_hdb.append(r2_hdb)
    boot_r2_rc.append(r2_rc)

    if (i + 1) % 10 == 0:
        print(f"  Bootstrap {i+1}/{N_BOOTSTRAP}  "
              f"KM={np.mean(boot_r2_km):.3f}  HDB={np.mean(boot_r2_hdb):.3f}  RC={np.mean(boot_r2_rc):.3f}")

print(f"\n  Bootstrap R2 summary (n={N_BOOTSTRAP}):")
for name, vals in [("K-Means clusters", boot_r2_km),
                    ("HDBSCAN clusters", boot_r2_hdb),
                    ("Road Class",       boot_r2_rc)]:
    v = np.array(vals)
    print(f"    {name:25s}  mean={v.mean():.4f}  std={v.std():.4f}  "
          f"95%CI=[{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]")

# plot bootstrap distributions
fig, ax = plt.subplots(figsize=(10, 5))
data_to_plot = [boot_r2_km, boot_r2_hdb, boot_r2_rc]
labels_bp    = ["K-Means clusters", "HDBSCAN clusters", "Road Class"]
colors_bp    = ["steelblue", "darkorange", "crimson"]

bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(labels_bp)
ax.set_ylabel("XGBoost R² (3-fold CV)")
ax.set_title(f"Bootstrap Stability — R² Distribution ({N_BOOTSTRAP} resamples)\n"
              "Each resample re-runs UMAP + clustering + XGBoost independently")
ax.set_ylim(bottom=0)
ax.axhline(np.median(boot_r2_rc), color="crimson", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / "sens_bootstrap.png", dpi=150)
plt.close()
print("Saved: sens_bootstrap.png")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SENSITIVITY SUMMARY")
print("=" * 70)

print("\n[1] UMAP n_components impact:")
print(dim_df.to_string(index=False))

print("\n[2] UMAP n_neighbors impact:")
print(nn_df.to_string(index=False))

print("\n[3] K-Means seed stability:")
print(seed_df.to_string(index=False))
print(f"    -> Mean NMI={seed_df['NMI_vs_42'].mean():.4f} (1.0 = identical, >=0.8 = stable)")

print("\n[4] HDBSCAN min_cluster_size impact:")
print(mcs_df.to_string(index=False))

print("\n[5] Bootstrap R2 (50 resamples):")
for name, vals in [("K-Means clusters", boot_r2_km),
                    ("HDBSCAN clusters", boot_r2_hdb),
                    ("Road Class",       boot_r2_rc)]:
    v = np.array(vals)
    print(f"    {name:25s}  mean={v.mean():.4f}  std={v.std():.4f}  "
          f"95%CI=[{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]")

# save all to CSV
all_rows = []
for _, r in dim_df.iterrows():
    all_rows.append({"experiment": "umap_dims", **r})
for _, r in nn_df.iterrows():
    all_rows.append({"experiment": "umap_neighbors", **r})
for _, r in seed_df.iterrows():
    all_rows.append({"experiment": "kmeans_seed", **r})
for _, r in mcs_df.iterrows():
    all_rows.append({"experiment": "hdbscan_mcs", **r})

pd.DataFrame(all_rows).to_csv(OUT / "sensitivity_results.csv", index=False)
print(f"\nAll sensitivity results saved to: {OUT / 'sensitivity_results.csv'}")
print("Done.")
