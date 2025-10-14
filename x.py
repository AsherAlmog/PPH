# train_tabm_pph.py
"""
TabM for PPH binary classification on engineered-window features.

Preserves your existing pipeline:
- Early column pruning by name (metric substrings + leakage/time blocklists)
- Duplicate ID removal (prefer hashed_mother_id)
- Column-pruned parquet reads, immediate pruning after read
- Categorical IDs and numeric downcasts on read
- Label tagging by intersection of snapshot times within (hashed_mother_id, episode_idx)
- Episode-level grouping (mother#episode)
- HARD BLOCK of time features (t_from_birth_sec_*, mins_postpartum, episode_start)
- Leakage guard & constant-feature drop
- Caching of matrices
- ROC / PR / Confusion plots
- Histograms by label

Replaces XGBoost with official `tabm` (parameter-efficient ensembling MLP).
- Trains with BCEWithLogitsLoss and class imbalance pos_weight
- Train-time standardization & median imputation (train-only stats)
- Validation on AUCPR (reported along with ROC-AUC etc.)
- Importance: proxy via first-layer weight norms (+ optional permutation importance)

Run:
    python train_tabm_pph.py
"""

from __future__ import annotations
import os, gc, json, math, random, sys
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

# PyTorch / TabM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tabm import TabM
except Exception as e:
    raise RuntimeError(
        "The 'tabm' package is required.\n"
        "Install with: pip install tabm\n"
        f"Import error: {e}"
    )

try:
    # Optional numeric feature embeddings (good default for TabM)
    from rtdl_num_embeddings import LinearReLUEmbeddings
    _HAS_NUM_EMB = True
except Exception:
    _HAS_NUM_EMB = False

# Reduce accidental copies
pd.options.mode.copy_on_write = True

# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict = {
    # -------- Paths (from builder) --------
    "base_dir": r"D:\PPH",
    "features_all_path": r"D:\PPH\features_all.parquet",
    "labels_all_path":   r"D:\PPH\out_labels\labels_vector.parquet",
    "feature_cols_json": r"D:\PPH\feature_columns.json",

    # Optional: CSV with a column of hashed_mother_id to KEEP
    "keep_mothers_csv": None,          # e.g. r"D:\PPH\keep_mothers.csv"
    "keep_mothers_col": "hashed_mother_id",

    # Label column name inside labels parquet
    "target_col": "label",

    # If labels file doesn't have `snapshot_time`, try this column for event-time labels
    "labels_event_time_col": None,

    # Multiclass → binary mapping
    "target_binarize": True,
    "positive_labels": [1, 2],

    # Drop rows where target is NaN?
    "drop_na_target": True,

    # (Only for diagnostics; NOT fed to model)
    "max_minutes_postpartum": None,

    # --------- metric whitelist (applies to real-time features only) ----------
    "keep_only_name_contains": [
        "sistol", "diastol", "bp - mean", "map", "pulse", "saturation",
        "shock_index", "si_", "shockindex"
    ],

    # --------- drop ALL real-time measurement features (keep all static) ----------
    "drop_realtime_measurements": False,

    # How to detect real-time measurement features (substrings, case-insensitive)
    "realtime_measurement_patterns": [
        # vital signs
        "sistol", "diastol", "bp - mean", "map", "pulse", "saturation", "heat",
        "shock_index", "si_", "shockindex",
        # labs
        "hgb", "hct", "plt", "fibrinogen", "wbc",
        "sodium_blood", "creatinine_blood", "uric_acid_blood",
        # time-resolution hints
        "_fine_", "_coarse_",
    ],

    # ------------- Name-based pruning (drop) -------------
    "drop_if_name_contains": [
        # e.g. "z_last", "iqr", "cov", "drug_", "given_", "pp_stats",
    ],

    # If a canonical ID exists (first in tuple), drop its duplicates listed after it if present
    "duplicate_id_candidates": [
        ("hashed_mother_id", ["mother_id"]),
    ],

    # ------------- Leakage & TIME-feature blocklists -------------
    "use_feature_blocklist": True,
    "hard_time_blocklist_patterns": (
        "t_from_birth_sec",
        "mins_postpartum",
        "episode_start",
    ),
    "leakage_blocklist_patterns": (
        "_recency_s",
        "_measured",
        "_given",
        "time_since_",
    ),
    "leakage_blocklist_columns": (
        "anesthesia_local","anesthesia_epidural","anesthesia_general",
        "anesthesia_spinal","no_anesthesia",
        "amniofusion","oxytocin_administrations",
        "membranes_rupture_type","amniotic_fluid",
    ),

    # Drop constants / near-constants to reduce noise
    "drop_constant_features": True,
    "min_feature_variance": 1e-12,

    # ------------- Validation ---------------
    "validation_strategy": "holdout",   # "holdout" or "group_kfold"
    "holdout_group_frac": 0.2,
    "group_kfold_splits": 5,
    "random_state": 42,

    # ------------- Compute ---------------
    "device": "cpu",            # "cpu" or "cuda"
    "num_workers": 0,
    "torch_num_threads": None,  # e.g., 8 (None = let PyTorch decide)

    # ------------- Training ---------------
    "epochs": 20,
    "batch_size": 1024,
    "val_batch_size": 4096,
    "lr": 2e-3,
    "weight_decay": 3e-4,
    "early_stop_patience": 5,      # None to disable

    # ------------- TabM params ------------
    "tabm_k": 24,                  # ensemble size (heads)
    "use_num_embeddings": True,    # if rtdl_num_embeddings is available
    "num_embedding_type": "linear_relu",  # only LinearReLUEmbeddings used here

    # ------------- Outputs ----------------
    "out_model_pt":          r"D:\PPH\tabm_model.pt",
    "out_used_features_json": r"D:\PPH\used_feature_columns.json",
    "out_metrics_txt":        r"D:\PPH\metrics.txt",
    "out_feat_csv":           r"D:\PPH\feature_importances.csv",
    "out_plot_roc":           r"D:\PPH\roc.png",
    "out_plot_pr":            r"D:\PPH\pr.png",
    "out_plot_cm":            r"D:\PPH\cm.png",
    "out_plot_feat":          r"D:\PPH\feat_importance.png",
    "feat_top_n": 30,

    # ------------- Caching ----------------
    "matrices_cache_enabled": False,
    "matrices_force_recompute": False,
    "matrices_cache_dir":   r"D:\PPH\.cache_tabm",
    "matrices_cache_X":     r"D:\PPH\.cache_tabm\Xdf.parquet",
    "matrices_cache_y":     r"D:\PPH\.cache_tabm\y.parquet",
    "matrices_cache_meta":  r"D:\PPH\.cache_tabm\meta.parquet",
    "matrices_cache_cols":  r"D:\PPH\.cache_tabm\used_cols.json",

    # --------- Balancing (train only) -----
    "downsample_training_to_even": True,
    "downsample_random_state": 123,

    # --------- Histograms -----------------
    "hist_top_n": 50,
    "hist_bins": 40,
    "out_hist_dir": r"D:\PPH\top_feature_hist",
    "hist_plot_all_used": True,

    # --------- Importances ----------------
    "perm_importance_enabled": False,
    "perm_importance_n_repeats": 5,
    "perm_importance_max_features": 200,  # top-K by proxy to keep it fast
}

# =============================================================================
# CONSTANTS / KEYS
# =============================================================================
TR_KEYS = ["hashed_mother_id", "episode_idx", "snapshot_time"]
PAIR_KEYS = ["hashed_mother_id", "episode_idx"]
META_KEYS = ["hashed_mother_id", "episode_idx", "snapshot_time", "pregnancy_index"]
LABEL_KEYSET = ["hashed_mother_id", "episode_idx"]

# =============================================================================
# Utilities
# =============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_torch_threads(n: Optional[int]):
    if n is not None and isinstance(n, int) and n > 0:
        torch.set_num_threads(n)

# ---------------- mother filter ----------------
def _load_keep_mothers_set(cfg: Dict) -> Optional[set[str]]:
    path = cfg.get("keep_mothers_csv")
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    col = cfg.get("keep_mothers_col") or "hashed_mother_id"
    if col not in df.columns:
        col = df.columns[0]
    s = df[col].astype(str).str.strip()
    s = s[s != ""]
    kept = set(s.unique().tolist())
    print(f"[INFO] keep_mothers_csv: keeping {len(kept):,} mothers.")
    return kept if kept else None

# ---------------- duplicate IDs ----------------
def _drop_duplicate_id_columns(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    for canon, dupes in cfg.get("duplicate_id_candidates", []):
        if canon in df.columns:
            to_drop = [d for d in dupes if d in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
    return df

# ---------------- core dtypes ----------------
def _coerce_core_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if "hashed_mother_id" in df:
        if not isinstance(df["hashed_mother_id"].dtype, CategoricalDtype):
            df["hashed_mother_id"] = df["hashed_mother_id"].astype("category")
        else:
            df["hashed_mother_id"] = df["hashed_mother_id"].cat.remove_unused_categories()
    if "episode_idx" in df:
        df["episode_idx"] = pd.to_numeric(df["episode_idx"], errors="coerce").astype("Int32")
    if "pregnancy_index" in df:
        df["pregnancy_index"] = pd.to_numeric(df["pregnancy_index"], errors="coerce").astype("Int64")
    if "snapshot_time" in df and not np.issubdtype(df["snapshot_time"].dtype, np.datetime64):
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    return df

# ---------------- numeric downcast ----------------
def _downcast_numeric_inplace(df: pd.DataFrame) -> None:
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_unsigned_integer_dtype(s):
            if pd.isna(s).any():
                try:
                    df[c] = pd.to_numeric(s, errors="coerce").astype("Int32")
                except Exception:
                    pass
            else:
                df[c] = pd.to_numeric(s, downcast="integer")

# ---------------- real-time vs static detection ----------------
def _is_realtime_feature(name: str, cfg: Dict) -> bool:
    low = name.lower()
    pats = [p.lower() for p in (cfg.get("realtime_measurement_patterns") or [])]
    return any(p in low for p in pats)

# ---------------- Name filters ----------------
def _name_is_dropped_by_metric(name: str, cfg: Dict) -> bool:
    drops = tuple(cfg.get("drop_if_name_contains", []) or [])
    if not drops:
        return False
    low = name.lower()
    return any(s.lower() in low for s in drops)

def _apply_keep_only_filter(cols: List[str], cfg: Dict) -> List[str]:
    keeps = cfg.get("keep_only_name_contains", []) or []
    if not keeps:
        return cols
    ks = [k.lower() for k in keeps]
    kept: List[str] = []
    for c in cols:
        if not _is_realtime_feature(c, cfg):
            kept.append(c)  # always keep static
        else:
            if any(k in c.lower() for k in ks):
                kept.append(c)
    if not kept:
        print(f"[WARN] keep_only_name_contains removed everything; falling back to original set.")
        return cols
    print(f"[INFO] keep_only_name_contains kept {len(kept)}/{len(cols)} columns (static always kept).")
    return kept

def _drop_realtime_from_list(cols: Iterable[str], cfg: Dict) -> List[str]:
    cols = list(cols)
    out = [c for c in cols if not _is_realtime_feature(c, cfg)]
    print(f"[INFO] drop_realtime_measurements removed {len(cols) - len(out)} real-time columns; kept {len(out)} static columns.")
    return out

def _blocklist_filter(cols: List[str], cfg: Dict) -> List[str]:
    if not cfg.get("use_feature_blocklist", True):
        return [c for c in cols if not _name_is_dropped_by_metric(c, cfg)]
    pats = tuple(cfg.get("leakage_blocklist_patterns", ())) + tuple(cfg.get("hard_time_blocklist_patterns", ()))
    bad_cols = set(cfg.get("leakage_blocklist_columns", ()))
    kept = []
    for c in cols:
        if c in bad_cols:
            continue
        if any(p in c for p in pats):
            continue
        if _name_is_dropped_by_metric(c, cfg):
            continue
        kept.append(c)
    return kept

def _plan_feature_columns_to_read(cfg: Dict) -> List[str]:
    with open(cfg["feature_cols_json"], "r") as f:
        engineered_cols: List[str] = json.load(f)
    forbid = {"hashed_mother_id","episode_idx","snapshot_time","episode_start","mins_postpartum","pregnancy_index"}
    candidates = [c for c in engineered_cols if c not in forbid]
    if cfg.get("drop_realtime_measurements", False):
        candidates = _drop_realtime_from_list(candidates, cfg)
    else:
        candidates = _apply_keep_only_filter(candidates, cfg)
    used = _blocklist_filter(candidates, cfg)
    return used

# ---------------- Loaders ----------------
def load_feature_label_tables(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep_ids = _load_keep_mothers_set(cfg)

    # FEATURES
    feat_cols = _plan_feature_columns_to_read(cfg)
    feat_keep = list(dict.fromkeys(feat_cols + META_KEYS))
    try:
        X = pd.read_parquet(cfg["features_all_path"], columns=feat_keep)
    except Exception:
        X = pd.read_parquet(cfg["features_all_path"])

    if keep_ids:
        if "hashed_mother_id" not in X.columns:
            raise ValueError("features missing 'hashed_mother_id' but keep_mothers_csv is set.")
        before = len(X)
        X = X[X["hashed_mother_id"].astype(str).isin(keep_ids)]
        print(f"[INFO] features: kept {len(X):,}/{before:,} rows after mother filter.")

    keepX = [c for c in feat_keep if c in X.columns]
    if len(keepX) != len(X.columns):
        X = X.loc[:, keepX]
    X = _drop_duplicate_id_columns(X, cfg)
    X = _coerce_core_dtypes(X)
    _downcast_numeric_inplace(X)

    # LABELS
    y_cols = list(dict.fromkeys(LABEL_KEYSET + [cfg["target_col"]]))
    tcol = cfg.get("labels_event_time_col") or "snapshot_time"
    y_cols.append(tcol)
    try:
        Y = pd.read_parquet(cfg["labels_all_path"], columns=[c for c in y_cols if c is not None])
    except Exception:
        Y = pd.read_parquet(cfg["labels_all_path"])

    if keep_ids:
        if "hashed_mother_id" not in Y.columns:
            raise ValueError("labels missing 'hashed_mother_id' but keep_mothers_csv is set.")
        before = len(Y)
        Y = Y[Y["hashed_mother_id"].astype(str).isin(keep_ids)]
        print(f"[INFO] labels: kept {len(Y):,}/{before:,} rows after mother filter.")

    keepY = [c for c in y_cols if c in Y.columns]
    if len(keepY) != len(Y.columns):
        Y = Y.loc[:, keepY]
    Y = _drop_duplicate_id_columns(Y, cfg)
    Y = _coerce_core_dtypes(Y)
    _downcast_numeric_inplace(Y)

    # Duplicate diagnostics
    for name, df, keys in (("features", X, TR_KEYS), ("labels", Y, TR_KEYS)):
        if all(k in df.columns for k in keys):
            n_dups = df.duplicated(keys, keep=False).sum()
            if n_dups:
                print(f"[INFO] {name}: {n_dups} duplicated rows on {keys}")
    return X, Y

# =============================================================================
# Time metadata
# =============================================================================
def add_minutes_postpartum_by_episode(X: pd.DataFrame) -> pd.DataFrame:
    need = ["hashed_mother_id", "episode_idx", "snapshot_time"]
    for k in need:
        if k not in X.columns:
            raise ValueError(f"'{k}' missing for postpartum calculation.")
    t0 = (
        X.groupby(PAIR_KEYS, observed=True, sort=False)["snapshot_time"]
         .min()
         .rename("episode_start")
         .reset_index()
    )
    X = X.join(t0.set_index(PAIR_KEYS), on=PAIR_KEYS)
    mins = (X["snapshot_time"] - X["episode_start"]).dt.total_seconds() / 60.0
    X = X.assign(mins_postpartum=mins)
    return X

# =============================================================================
# Labels by intersection
# =============================================================================
def _detect_labels_time_col(Y: pd.DataFrame, cfg: Dict) -> Optional[str]:
    if "snapshot_time" in Y.columns:
        return "snapshot_time"
    et = cfg.get("labels_event_time_col")
    if et and et in Y.columns:
        return et
    cand = [c for c in Y.columns if "time" in str(c).lower()]
    return cand[0] if cand else None

def apply_labels_by_intersection(X: pd.DataFrame, Y: pd.DataFrame, cfg: Dict) -> pd.Series:
    target_col = cfg["target_col"]
    if target_col not in Y.columns:
        raise ValueError(f"Target column '{target_col}' not found in labels parquet")
    pairs = X[PAIR_KEYS].drop_duplicates()
    Y_pairs = Y.merge(pairs, on=PAIR_KEYS, how="inner")
    tcol = _detect_labels_time_col(Y_pairs, cfg)
    if tcol is None:
        lab_per_pair = (
            Y_pairs.groupby(PAIR_KEYS, as_index=False, observed=True, sort=False)[target_col]
                   .max()
        )
        X_lab = X.merge(lab_per_pair, on=PAIR_KEYS, how="left", sort=False)
        y_arr = X_lab[target_col].to_numpy()
        return pd.Series(y_arr, index=X.index, name=target_col).fillna(0).astype(int)

    Y_pairs = Y_pairs.copy()
    Y_pairs[tcol] = pd.to_datetime(Y_pairs[tcol], errors="coerce")
    Y_pairs = Y_pairs.dropna(subset=[tcol])
    if tcol != "snapshot_time":
        Y_pairs = Y_pairs.rename(columns={tcol: "snapshot_time"})
    Y_cand = (
        Y_pairs.groupby(TR_KEYS, as_index=False, observed=True, sort=False)[target_col]
               .max()
    )
    mark = X[TR_KEYS].merge(Y_cand, on=TR_KEYS, how="left", sort=False)
    y_arr = mark[target_col].to_numpy()
    return pd.Series(y_arr, index=X.index, name=target_col).fillna(0).astype(int)

# =============================================================================
# Matrices builder
# =============================================================================
def _compute_matrices(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    X, Y = load_feature_label_tables(cfg)
    X = add_minutes_postpartum_by_episode(X)
    used = [c for c in _plan_feature_columns_to_read(cfg) if c in X.columns]

    Xdf = X[used]
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    for c in Xdf.columns:
        if not (pd.api.types.is_float_dtype(Xdf[c]) or pd.api.types.is_integer_dtype(Xdf[c])):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        Xdf[c] = Xdf[c].astype(np.float32)

    if cfg.get("drop_constant_features", True):
        var = Xdf.var(axis=0, numeric_only=True)
        keep = var.index[var.values > float(cfg.get("min_feature_variance", 1e-12))].tolist()
        if len(keep) < len(used):
            print(f"[INFO] Dropping {len(used) - len(keep)} near-constant features.")
        used = keep
        Xdf = Xdf[used]

    y = apply_labels_by_intersection(X, Y, cfg)

    if cfg.get("drop_na_target", True):
        mask = ~pd.isna(y)
        Xdf = Xdf.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        X = X.loc[mask].reset_index(drop=True)

    if cfg.get("target_binarize", True):
        pos_set = set(cfg.get("positive_labels", [1, 2]))
        y = y.apply(lambda v: 1 if v in pos_set else 0).astype(int)
    else:
        y = y.astype(int)

    meta_cols = ["hashed_mother_id","episode_idx","snapshot_time","mins_postpartum","pregnancy_index"]
    meta_cols = [c for c in meta_cols if c in X.columns]
    meta = X[meta_cols].copy()

    del X, Y
    gc.collect()
    return Xdf, y, meta, used

def build_matrices(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    if not cfg.get("matrices_cache_enabled", True):
        return _compute_matrices(cfg)

    os.makedirs(cfg["matrices_cache_dir"], exist_ok=True)
    Xp, yp, mp, cp = (cfg["matrices_cache_X"], cfg["matrices_cache_y"],
                      cfg["matrices_cache_meta"], cfg["matrices_cache_cols"])

    if (not cfg.get("matrices_force_recompute", False)
        and all(os.path.exists(p) for p in (Xp, yp, mp, cp))):
        Xdf = pd.read_parquet(Xp)
        y = pd.read_parquet(yp)["y"].astype(int)
        meta = pd.read_parquet(mp)
        with open(cp, "r") as f:
            used = json.load(f)
        return Xdf, y, meta, used

    Xdf, y, meta, used = _compute_matrices(cfg)
    tmp = ".tmp"
    Xdf.to_parquet(Xp + tmp, index=False)
    pd.DataFrame({"y": y.values}).to_parquet(yp + tmp, index=False)
    meta.to_parquet(mp + tmp, index=False)
    with open(cp + tmp, "w") as f:
        json.dump(used, f, indent=2)
    for s, d in ((Xp+tmp, Xp), (yp+tmp, yp), (mp+tmp, mp), (cp+tmp, cp)):
        if os.path.exists(d): os.remove(d)
        os.replace(s, d)
    return Xdf, y, meta, used

# =============================================================================
# Metrics & plotting
# =============================================================================
def summarize_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    return "\n".join([
        f"AUC: {auc:.4f}",
        f"Average Precision (PR AUC): {ap:.4f}",
        f"Precision@{threshold:.2f}: {p:.4f}",
        f"Recall/Sensitivity@{threshold:.2f}: {r:.4f}",
        f"Specificity@{threshold:.2f}: {spec:.4f}",
        f"F1@{threshold:.2f}: {f1:.4f}",
        f"Confusion Matrix @{threshold:.2f}: TN={tn} FP={fp} FN={fn} TP={tp}",
    ])

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision (PPV)")
    plt.title("Precision–Recall Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, interpolation='nearest')
    plt.title(f"Confusion Matrix @ {threshold:.2f}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Pred 0", "Pred 1"])
    plt.yticks(ticks, ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(mat[i, j]), ha="center", va="center",
                     color="white" if mat[i, j] > mat.max()/2 else "black")
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
    plt.savefig(out_path, dpi=160); plt.close()

# =============================================================================
# Histograms by label
# =============================================================================
def plot_feature_histograms_by_label(
    Xdf: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    out_dir: str,
    bins: int = 40,
    annotate_min_count: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    yb = y.astype(int).values
    pos_mask = yb == 1
    neg_mask = yb == 0

    for feat in features:
        if feat not in Xdf.columns:
            continue
        s_pos = pd.to_numeric(Xdf.loc[pos_mask, feat], errors="coerce").dropna()
        s_neg = pd.to_numeric(Xdf.loc[neg_mask, feat], errors="coerce").dropna()
        n_pos = len(s_pos); n_neg = len(s_neg)
        if n_pos == 0 and n_neg == 0:
            continue

        all_vals = np.concatenate([s_pos.to_numpy(), s_neg.to_numpy()])
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1] + widths / 2
        cnt_pos, _ = np.histogram(s_pos, bins=bin_edges)
        cnt_neg, _ = np.histogram(s_neg, bins=bin_edges)

        pct_pos = (cnt_pos / max(n_pos, 1)) * 100.0
        pct_neg = (cnt_neg / max(n_neg, 1)) * 100.0

        plt.figure(figsize=(8.5, 4.6))
        plt.bar(bin_edges[:-1], pct_neg, width=widths, alpha=0.5, align="edge",
                label=f"label=0 (non-NaN n={n_neg})")
        plt.bar(bin_edges[:-1], pct_pos, width=widths, alpha=0.5, align="edge",
                label=f"label=1 (non-NaN n={n_pos})")

        off0, off1 = -0.18, 0.18
        for i in range(len(centers)):
            if cnt_neg[i] >= annotate_min_count:
                plt.text(centers[i] + off0*widths[i], pct_neg[i] + 0.3,
                         f"{cnt_neg[i]}", ha="center", va="bottom", fontsize=7, rotation=90)
            if cnt_pos[i] >= annotate_min_count:
                plt.text(centers[i] + off1*widths[i], pct_pos[i] + 0.3,
                         f"{cnt_pos[i]}", ha="center", va="bottom", fontsize=7, rotation=90)

        plt.title(f"Histogram (% within label among non-NaNs): {feat}")
        plt.xlabel(feat)
        plt.ylabel("% of samples in label (non-NaN)")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"hist_{feat.replace('/', '_')}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()

# =============================================================================
# Balancing (train only)
# =============================================================================
def _downsample_to_even(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    yb = y.astype(int).values
    pos_idx = np.flatnonzero(yb == 1)
    neg_idx = np.flatnonzero(yb == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, np.arange(len(y))
    if len(pos_idx) > len(neg_idx):
        maj, mini = pos_idx, neg_idx
    else:
        maj, mini = neg_idx, pos_idx
    rng = np.random.RandomState(random_state)
    keep_maj = rng.choice(maj, size=len(mini), replace=False)
    keep = np.sort(np.concatenate([keep_maj, mini]))
    return X.iloc[keep], y.iloc[keep], keep

# =============================================================================
# Holdout masks
# =============================================================================
def make_holdout_masks(groups: np.ndarray, holdout_frac: float, seed: int = 42):
    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)
    if n_groups < 2:
        va_mask = np.zeros_like(codes, dtype=bool)
        tr_mask = ~va_mask
        return tr_mask, va_mask
    rng = np.random.RandomState(seed)
    n_val = max(1, int(math.ceil(n_groups * float(holdout_frac))))
    val_codes = rng.choice(n_groups, size=n_val, replace=False)
    flags = np.zeros(n_groups, dtype=bool); flags[val_codes] = True
    va_mask = flags[codes]; tr_mask = ~va_mask
    return tr_mask, va_mask

# =============================================================================
# Preprocessing: median imputation + standardization (train-only stats)
# =============================================================================
class StandardizeImputer:
    def __init__(self):
        self.median_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.median_ = np.nanmedian(X, axis=0).astype(np.float32)
        X_imp = np.where(np.isnan(X), self.median_, X)
        self.mean_ = X_imp.mean(axis=0).astype(np.float32)
        self.std_ = X_imp.std(axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = np.where(np.isnan(X), self.median_, X).astype(np.float32)
        return (X_imp - self.mean_) / self.std_

    def to_dict(self) -> Dict[str, list]:
        return {"median": self.median_.tolist(), "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @staticmethod
    def from_dict(d: Dict[str, list]) -> "StandardizeImputer":
        si = StandardizeImputer()
        si.median_ = np.array(d["median"], dtype=np.float32)
        si.mean_   = np.array(d["mean"], dtype=np.float32)
        si.std_    = np.array(d["std"], dtype=np.float32)
        return si

# =============================================================================
# Datasets
# =============================================================================
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i: int): return self.X[i], self.y[i]

# =============================================================================
# TabM model helpers
# =============================================================================
def make_tabm(n_features: int, cfg: Dict) -> TabM:
    """Build a TabM model for numeric-only inputs."""
    use_emb = bool(cfg.get("use_num_embeddings", True) and _HAS_NUM_EMB)
    if use_emb:
        emb = LinearReLUEmbeddings(n_features)
        model = TabM.make(
            n_num_features=n_features,
            num_embeddings=emb,
            d_out=1,
            k=int(cfg.get("tabm_k", 24)),
        )
    else:
        model = TabM.make(
            n_num_features=n_features,
            d_out=1,
            k=int(cfg.get("tabm_k", 24)),
        )
    return model

def train_epoch(model, loader, optimizer, device, pos_weight: Optional[float]):
    model.train()
    if pos_weight and pos_weight > 0:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits_k = model(xb).squeeze(-1)             # (B, K)
        # Train rule: mean of per-head losses (do NOT average logits first)
        loss = loss_fn(logits_k, yb.unsqueeze(1).expand_as(logits_k))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits_k = model(xb).squeeze(-1)             # (B, K)
        p = torch.sigmoid(logits_k).mean(dim=1)      # inference: average probabilities over heads
        outs.append(p.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

# =============================================================================
# Importance (proxy + optional permutation)
# =============================================================================
def weight_norm_importance_tabm(model: TabM, feature_names: List[str]) -> pd.DataFrame:
    """
    Proxy importance: L1 norm of the first numeric block's shared weights
    (works when the first layer is Linear or part of LinearReLUEmbeddings).
    If structure changes, this stays a heuristic proxy.
    """
    # Try to locate a first linear weight matrix
    w = None
    for n, p in model.named_parameters():
        if "weight" in n and p.dim() == 2 and p.shape[1] == len(feature_names):
            w = p.detach().cpu().abs().sum(dim=0).numpy()  # sum over outputs
            break
    if w is None:
        # Fallback: aggregate all 2D weights mapped to input dim
        acc = None
        for n, p in model.named_parameters():
            if "weight" in n and p.dim() == 2 and p.shape[1] == len(feature_names):
                v = p.detach().cpu().abs().sum(dim=0).numpy()
                acc = v if acc is None else acc + v
        if acc is None:
            # give uniform tiny scores to avoid errors
            acc = np.ones(len(feature_names), dtype=float)
        w = acc
    df = pd.DataFrame({"feature": feature_names, "importance": w})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df

def compute_permutation_importance(
    model: TabM,
    X_va: np.ndarray,
    y_va: np.ndarray,
    device: str,
    feature_names: List[str],
    n_repeats: int = 5,
    max_features: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sklearn permutation importance wrapper over TabM predict_proba."""
    class _Wrap:
        def __init__(self, model, device):
            self.model = model; self.device = device
        def predict_proba(self, X):
            ds = ArrayDataset(X.astype(np.float32), np.zeros((X.shape[0],), dtype=np.float32))
            dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
            p = predict_proba(self.model, dl, self.device)
            return np.vstack([1.0 - p, p]).T

    base_imp = weight_norm_importance_tabm(model, feature_names)
    idx = np.arange(len(feature_names))
    if max_features:
        take = min(max_features, len(feature_names))
        top_feats = base_imp.head(take)["feature"].tolist()
        idx = np.array([feature_names.index(f) for f in top_feats], dtype=int)
        X_sel = X_va[:, idx]
        feat_sel = [feature_names[i] for i in idx]
    else:
        X_sel = X_va; feat_sel = feature_names

    wrapper = _Wrap(model, device)
    res = permutation_importance(wrapper, X_sel, y_va, n_repeats=n_repeats,
                                 scoring="average_precision", random_state=seed, n_jobs=1)
    df = pd.DataFrame({
        "feature": feat_sel,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df

def plot_feature_importance_df(imp_df: pd.DataFrame, out_path: str, top_n: int = 30, col="importance"):
    if imp_df is None or imp_df.empty:
        print("[WARN] No feature importances to plot."); return
    key = col if col in imp_df.columns else (imp_df.columns[1] if len(imp_df.columns) > 1 else None)
    if key is None:
        print("[WARN] Importance DF has no plottable column."); return
    top = imp_df.sort_values(key, ascending=False).head(top_n).iloc[::-1]
    if top.empty:
        print("[WARN] Top-N slice empty; skipping importance plot."); return
    plt.figure(figsize=(8, max(4, 0.28 * len(top))))
    plt.barh(top["feature"], top[key])
    plt.xlabel(key)
    plt.title(f"Top {len(top)} Feature Importances ({key})")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# =============================================================================
# Split prep
# =============================================================================
def _prepare_split_tensors(Xdf: pd.DataFrame, y: pd.Series, idx: np.ndarray, scaler: Optional[StandardizeImputer] = None):
    X_np = Xdf.iloc[idx].to_numpy(dtype=np.float32, copy=True)
    y_np = y.iloc[idx].to_numpy(dtype=np.float32, copy=False)
    if scaler is None:
        scaler = StandardizeImputer().fit(X_np)
    X_np = scaler.transform(X_np)
    return X_np, y_np, scaler

# =============================================================================
# Train drivers
# =============================================================================
def train_holdout(cfg: Dict, Xdf: pd.DataFrame, y: pd.Series, groups: np.ndarray, feature_names: List[str]):
    tr_mask, va_mask = make_holdout_masks(groups, cfg["holdout_group_frac"], cfg.get("random_state", 42))
    X_tr, y_tr = Xdf.iloc[tr_mask], y.iloc[tr_mask]
    X_va, y_va = Xdf.iloc[va_mask], y.iloc[va_mask]

    if cfg.get("downsample_training_to_even", False):
        X_tr, y_tr, _ = _downsample_to_even(X_tr, y_tr, cfg.get("downsample_random_state", 42))

    # scaler fit on TRAIN only
    X_tr_np, y_tr_np, scaler = _prepare_split_tensors(X_tr, y_tr, np.arange(len(X_tr)))
    X_va_np, y_va_np, _      = _prepare_split_tensors(X_va, y_va, np.arange(len(X_va)), scaler=scaler)

    device = cfg.get("device", "cpu")
    model = make_tabm(X_tr_np.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    pos = float((y_tr_np == 1).sum()); neg = float((y_tr_np == 0).sum())
    pos_weight = (neg / max(pos, 1.0)) if pos > 0 else None

    train_loader = DataLoader(ArrayDataset(X_tr_np, y_tr_np),
                              batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader   = DataLoader(ArrayDataset(X_va_np, y_va_np),
                              batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    best_ap = -np.inf
    best_state = None
    patience = cfg.get("early_stop_patience", 5)
    no_improve = 0

    set_torch_threads(cfg.get("torch_num_threads"))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr_loss = train_epoch(model, train_loader, opt, device, pos_weight)
        y_prob_va = predict_proba(model, val_loader, device)
        ap = average_precision_score(y_va_np, y_prob_va) if len(np.unique(y_va_np)) > 1 else float("nan")
        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.5f}  val_AP={ap:.6f}")

        if np.isfinite(ap) and ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if patience is not None and no_improve >= patience:
            print(f"[EarlyStop] patience reached ({patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_prob_va = predict_proba(model, val_loader, device)
    report = summarize_metrics(y_va_np, y_prob_va)

    # importance
    imp_proxy = weight_norm_importance_tabm(model, feature_names)
    imp_perm = None
    if cfg.get("perm_importance_enabled", False):
        imp_perm = compute_permutation_importance(
            model, X_va_np, y_va_np, device,
            feature_names=feature_names,
            n_repeats=int(cfg.get("perm_importance_n_repeats", 5)),
            max_features=int(cfg.get("perm_importance_max_features", 200)),
            seed=int(cfg.get("random_state", 42)),
        )

    return model, report, y_va_np, y_prob_va, scaler, imp_proxy, imp_perm

def train_group_kfold(cfg: Dict, Xdf: pd.DataFrame, y: pd.Series, groups: np.ndarray, feature_names: List[str]):
    gkf = GroupKFold(n_splits=cfg["group_kfold_splits"])
    fold_reports = []
    best_model, best_scaler, best_auc = None, None, -np.inf
    best_yva, best_pva = None, None
    best_imp_proxy, best_imp_perm = None, None

    device = cfg.get("device", "cpu")
    set_torch_threads(cfg.get("torch_num_threads"))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xdf, y, groups=groups), start=1):
        X_tr, y_tr = Xdf.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = Xdf.iloc[va_idx], y.iloc[va_idx]

        if cfg.get("downsample_training_to_even", False):
            X_tr, y_tr, _ = _downsample_to_even(X_tr, y_tr, cfg.get("downsample_random_state", 42))

        X_tr_np, y_tr_np, scaler = _prepare_split_tensors(X_tr, y_tr, np.arange(len(X_tr)))
        X_va_np, y_va_np, _      = _prepare_split_tensors(X_va, y_va, np.arange(len(X_va)), scaler=scaler)

        model = make_tabm(X_tr_np.shape[1], cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        pos = float((y_tr_np == 1).sum()); neg = float((y_tr_np == 0).sum())
        pos_weight = (neg / max(pos, 1.0)) if pos > 0 else None

        train_loader = DataLoader(ArrayDataset(X_tr_np, y_tr_np),
                                  batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
        val_loader   = DataLoader(ArrayDataset(X_va_np, y_va_np),
                                  batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"])

        best_ap = -np.inf
        best_state = None
        patience = cfg.get("early_stop_patience", 5)
        no_improve = 0

        for epoch in range(1, int(cfg["epochs"]) + 1):
            tr_loss = train_epoch(model, train_loader, opt, device, pos_weight)
            y_prob_va = predict_proba(model, val_loader, device)
            ap = average_precision_score(y_va_np, y_prob_va) if len(np.unique(y_va_np)) > 1 else float("nan")
            print(f"[FOLD {fold} | Epoch {epoch:03d}] train_loss={tr_loss:.5f}  val_AP={ap:.6f}")

            if np.isfinite(ap) and ap > best_ap:
                best_ap = ap
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if patience is not None and no_improve >= patience:
                print(f"[FOLD {fold}] EarlyStop ({patience}).")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_prob_va = predict_proba(model, val_loader, device)
        auc = roc_auc_score(y_va_np, y_prob_va) if len(np.unique(y_va_np)) > 1 else float("nan")
        fold_reports.append(f"[FOLD {fold}]\n" + summarize_metrics(y_va_np, y_prob_va))

        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_model, best_scaler = auc, model, scaler
            best_yva, best_pva = y_va_np, y_prob_va
            best_imp_proxy = weight_norm_importance_tabm(model, feature_names)
            if cfg.get("perm_importance_enabled", False):
                best_imp_perm = compute_permutation_importance(
                    model, X_va_np, y_va_np, device,
                    feature_names=feature_names,
                    n_repeats=int(cfg.get("perm_importance_n_repeats", 5)),
                    max_features=int(cfg.get("perm_importance_max_features", 200)),
                    seed=int(cfg.get("random_state", 42)),
                )

    assert best_model is not None, "CV failed to produce a model."
    return best_model, "\n\n".join(fold_reports), best_yva, best_pva, best_scaler, best_imp_proxy, best_imp_perm

# =============================================================================
# Main
# =============================================================================
def main(cfg: Dict = CFG):
    seed_everything(int(cfg.get("random_state", 42)))
    set_torch_threads(cfg.get("torch_num_threads"))

    for p in (cfg["out_model_pt"], cfg["out_used_features_json"], cfg["out_metrics_txt"],
              cfg["out_feat_csv"], cfg["out_plot_roc"], cfg["out_plot_pr"],
              cfg["out_plot_cm"], cfg["out_plot_feat"]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(cfg.get("out_hist_dir", os.path.join(cfg["base_dir"], "top_feature_hist")), exist_ok=True)

    # Build matrices
    Xdf, y, meta, used_cols = build_matrices(cfg)

    # Group key: mother#episode
    if not {"hashed_mother_id","episode_idx"}.issubset(meta.columns):
        raise ValueError("meta must contain 'hashed_mother_id' and 'episode_idx' for grouping.")
    groups = (meta["hashed_mother_id"].astype(str) + "#" + meta["episode_idx"].astype(str)).values

    # Train
    if cfg["validation_strategy"] == "holdout":
        model, report, y_va, y_prob, scaler, imp_proxy, imp_perm = train_holdout(cfg, Xdf, y, groups, used_cols)
    elif cfg["validation_strategy"] == "group_kfold":
        model, report, y_va, y_prob, scaler, imp_proxy, imp_perm = train_group_kfold(cfg, Xdf, y, groups, used_cols)
    else:
        raise ValueError("validation_strategy must be 'holdout' or 'group_kfold'")

    # Save artifacts
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
         "used_cols": used_cols,
         "scaler": scaler.to_dict()},
        cfg["out_model_pt"]
    )
    with open(cfg["out_used_features_json"], "w") as f:
        json.dump(used_cols, f, indent=2)
    with open(cfg["out_metrics_txt"], "w") as f:
        f.write(report + "\n")

    # Plots
    plot_roc(y_va, y_prob, cfg["out_plot_roc"])
    plot_pr(y_va, y_prob, cfg["out_plot_pr"])
    plot_confusion(y_va, y_prob, cfg["out_plot_cm"], threshold=0.5)

    # Importances
    if imp_proxy is not None:
        imp_proxy.to_csv(cfg["out_feat_csv"], index=False)
        plot_feature_importance_df(imp_proxy.rename(columns={"importance": "importance"}),
                                   cfg["out_plot_feat"], top_n=int(cfg.get("feat_top_n", 30)), col="importance")

    if imp_perm is not None:
        # Save permutation alongside (append suffix)
        p_csv = cfg["out_feat_csv"].replace(".csv", "_perm.csv")
        imp_perm.to_csv(p_csv, index=False)
        p_png = cfg["out_plot_feat"].replace(".png", "_perm.png")
        plot_feature_importance_df(imp_perm.rename(columns={"importance_mean": "importance"}),
                                   p_png, top_n=int(cfg.get("feat_top_n", 30)), col="importance")

    # Histograms by label
    top_n = int(cfg.get("hist_top_n", 20))
    bins = int(cfg.get("hist_bins", 40))
    out_hist_dir = cfg.get("out_hist_dir", os.path.join(cfg["base_dir"], "top_feature_hist"))

    features_to_plot: List[str] = []
    if cfg.get("hist_plot_all_used", False):
        features_to_plot = [f for f in used_cols if f in Xdf.columns]
    else:
        if imp_proxy is not None and not imp_proxy.empty:
            top_feats = imp_proxy.sort_values("importance", ascending=False)["feature"].tolist()
            seen = set()
            for f in top_feats:
                if f in used_cols and f not in seen:
                    features_to_plot.append(f); seen.add(f)
                if len(features_to_plot) >= top_n:
                    break

    if features_to_plot:
        plot_feature_histograms_by_label(Xdf, y, features_to_plot, out_hist_dir, bins=bins)
        print(f"[OK] Per-class histograms written to: {out_hist_dir}")
    else:
        print("[INFO] No features selected for hist plotting.")

    # Prints
    print("[OK] Model:", cfg["out_model_pt"])
    print("[OK] Used features:", cfg["out_used_features_json"])
    print("[OK] Metrics:", cfg["out_metrics_txt"])
    print("[OK] Feature importances CSV:", cfg["out_feat_csv"])
    print("[OK] Plots:")
    print(" -", cfg["out_plot_roc"])
    print(" -", cfg["out_plot_pr"])
    print(" -", cfg["out_plot_cm"])
    print(" -", cfg["out_plot_feat"])
    print("\n" + report)

if __name__ == "__main__":
    main()
