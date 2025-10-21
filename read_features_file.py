"""
Train XGBoost on PPH features with:
- Early column pruning by name (metric substrings + leakage/time blocklists)
- Duplicate ID removal (prefer hashed_mother_id)
- Column-pruned parquet reads, immediate pruning after read
- Categorical IDs and numeric downcasts on read
- Label tagging by intersection of snapshot times within (hashed_mother_id, episode_idx)
- Episode-level grouping (composite mother#episode)
- HARD BLOCK of time features (t_from_birth_sec_*, mins_postpartum, episode_start)
- Leakage guard & constant-feature drop
- Caching of matrices
- Robust feature importances & plots

NEW:
- keep_only_name_contains: whitelist to select measurement features by name (applies only to real-time features; static features are always kept)
- drop_realtime_measurements: drop ALL real-time measurement features; keep ALL static features
- keep_mothers_csv: optional CSV with hashed_mother_id to slice features & labels immediately after reading
"""

from __future__ import annotations
import json, os, gc
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
from xgboost import XGBClassifier

# Reduce accidental copies
pd.options.mode.copy_on_write = True

# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict = {
    # -------- Paths (from builder) --------
    "base_dir": r"D:\PPH",
    "features_all_path": r"D:\PPH\features_all.parquet",
    "labels_all_path":   r"D:\PPH\labels_all.parquet",
    "feature_cols_json": r"D:\PPH\feature_columns_all.json",

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

    # (Only used for metadata/diagnostics; NOT fed to the model)
    "max_minutes_postpartum": None,

    # --------- NEW: metric whitelist (applies to real-time features only) ----------
    # Only measurement features whose names contain ANY of these substrings (case-insensitive) are kept.
    # Static features are always kept regardless of this list.
    # Example: ["HGB", "sistol", "BP", "MAP"]. Leave empty to disable.
    "keep_only_name_contains": ["sistol", "diastol", "bp - mean", "map", "pulse", "saturation",
        "shock_index", "si_", "shockindex", "HGB", "HCT", "PP"],

    # --------- NEW: drop ALL real-time measurement features (keep all static) ----------
    "drop_realtime_measurements": False,

    # How to detect real-time measurement features (substrings, case-insensitive)
    "realtime_measurement_patterns": [
        # vital signs
        "sistol", "diastol", "bp - mean", "map", "pulse", "saturation", "heat",
        "shock_index", "si_", "shockindex", "PP",
        # labs
        "hgb", "hct", "plt", "fibrinogen", "wbc",
        "sodium_blood", "creatinine_blood", "uric_acid_blood",
        # time-resolution hints
        "_fine_", "_coarse_"
    ],

    # ------------- Name-based pruning (drop) -------------
    # If any of these substrings appear in a feature name, DROP that column (before reading parquet)
    "drop_if_name_contains": ["delta_t"
        # e.g. "z_last", "iqr", "cov", "drug_", "given_", "pp_stats",
    ],

    # If a canonical ID exists (first in tuple), drop its duplicates listed after it if present
    # e.g., keep hashed_mother_id; drop mother_id if both exist
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
    "drop_constant_features": False,
    "min_feature_variance": 1e-12,

    # ------------- Validation ---------------
    "validation_strategy": "group_kfold",   # "group_kfold" or "group_kfold"
    "holdout_group_frac": 0.2,
    "group_kfold_splits": 5,
    "random_state": 42,

    # ------------- XGBoost params -----------
    "xgb_params": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
    },

    # ------------- Outputs ------------------
    "out_model_json":         r"D:\PPH\xgb_model.json",
    "out_used_features_json": r"D:\PPH\used_feature_columns.json",
    "out_metrics_txt":        r"D:\PPH\metrics.txt",
    "out_feat_csv":           r"D:\PPH\feature_importances.csv",
    "out_plot_roc":           r"D:\PPH\roc.png",
    "out_plot_pr":            r"D:\PPH\pr.png",
    "out_plot_cm":            r"D:\PPH\cm.png",
    "out_plot_feat":          r"D:\PPH\feat_importance.png",
    "feat_top_n": 30,

    # ------------- Caching ------------------
    "matrices_cache_enabled": True,
    "matrices_force_recompute": True,
    "matrices_cache_dir":   r"D:\PPH\.cache_xgb",
    "matrices_cache_X":     r"D:\PPH\.cache_xgb\Xdf.parquet",
    "matrices_cache_y":     r"D:\PPH\.cache_xgb\y.parquet",
    "matrices_cache_meta":  r"D:\PPH\.cache_xgb\meta.parquet",
    "matrices_cache_cols":  r"D:\PPH\.cache_xgb\used_cols.json",

    # --------- Balancing (train only) -------
    "downsample_training_to_even": False,
    "downsample_random_state": 123,

    # --------- Histograms -------------------
    "hist_top_n": 50,
    "hist_bins": 40,
    "out_hist_dir": r"D:\PPH\top_feature_hist_1810",
    "hist_plot_all_used": True,   # plot ALL used features when True
}

# =============================================================================
# I/O helpers
# =============================================================================
TR_KEYS = ["hashed_mother_id", "episode_idx", "snapshot_time"]
PAIR_KEYS = ["hashed_mother_id", "episode_idx"]
META_KEYS = ["hashed_mother_id", "episode_idx", "snapshot_time", "pregnancy_index"]
LABEL_KEYSET  = ["hashed_mother_id", "episode_idx"]

# ---------------- Utility: mother filter ----------------
def _load_keep_mothers_set(cfg: Dict) -> Optional[set[str]]:
    path = cfg.get("keep_mothers_csv")
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    col = cfg.get("keep_mothers_col") or "hashed_mother_id"
    if col not in df.columns:
        # fallback to the first column
        col = df.columns[0]
    s = df[col].astype(str).str.strip()
    s = s[s != ""]
    kept = set(s.unique().tolist())
    print(f"[INFO] keep_mothers_csv: keeping {len(kept):,} mothers from '{path}'.")
    return kept if kept else None

# ---------------- Utility: duplicate IDs ----------------
def _drop_duplicate_id_columns(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    for canon, dupes in cfg.get("duplicate_id_candidates", []):
        if canon in df.columns:
            to_drop = [d for d in dupes if d in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
    return df

# ---------------- Utility: core dtypes (no deprecations) ----------------
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

# ---------------- Utility: numeric downcast ----------------
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

# ---------------- Real-time vs static detection ----------------
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
    """
    Whitelist for measurement features. Static features are ALWAYS kept.
    If keep_only list is empty -> return cols unchanged.
    """
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
        print(f"[WARN] keep_only_name_contains={keeps} removed everything; falling back to original set.")
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

    # Drop all real-time measurement columns (static-only mode)
    if cfg.get("drop_realtime_measurements", False):
        candidates = _drop_realtime_from_list(candidates, cfg)
    else:
        # Otherwise, apply keep-only to measurement features (static always kept)
        candidates = _apply_keep_only_filter(candidates, cfg)

    # Then apply leakage/time blocklists and custom drop-by-name
    used = _blocklist_filter(candidates, cfg)
    return used

# ---------------- Loaders (read → prune immediately) ----------------
def load_feature_label_tables(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep_ids = _load_keep_mothers_set(cfg)

    # --- FEATURES ---
    feat_cols = _plan_feature_columns_to_read(cfg)
    feat_keep = list(dict.fromkeys(feat_cols + META_KEYS))

    try:
        X = pd.read_parquet(cfg["features_all_path"], columns=feat_keep)
    except Exception:
        X = pd.read_parquet(cfg["features_all_path"])

    # Slice by mother list immediately (before other processing)
    if keep_ids:
        if "hashed_mother_id" not in X.columns:
            raise ValueError("features missing 'hashed_mother_id' but keep_mothers_csv is set.")
        before = len(X)
        X = X[X["hashed_mother_id"].astype(str).isin(keep_ids)]
        print(f"[INFO] features: kept {len(X):,}/{before:,} rows after mother filter.")

    # RIGHT AWAY prune any extras
    keepX = [c for c in feat_keep if c in X.columns]
    if len(keepX) != len(X.columns):
        X = X.loc[:, keepX]

    # Drop duplicate IDs immediately
    X = _drop_duplicate_id_columns(X, cfg)

    # Dtypes + downcast now (keep RAM low)
    X = _coerce_core_dtypes(X)
    _downcast_numeric_inplace(X)

    # --- LABELS ---
    y_cols = list(dict.fromkeys(LABEL_KEYSET + [cfg["target_col"]]))
    tcol = cfg.get("labels_event_time_col") or "snapshot_time"
    y_cols.append(tcol)

    try:
        Y = pd.read_parquet(cfg["labels_all_path"], columns=[c for c in y_cols if c is not None])
    except Exception:
        Y = pd.read_parquet(cfg["labels_all_path"])

    # Slice labels by mother list immediately
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

    # Optional quick duplicate diagnostics
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

    # observed=True avoids exploding cartesian products for categoricals
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
# Label application via group-wise intersection
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

    # Metadata-only time feature (blocked from model inputs)
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

def feature_importance_df(model: XGBClassifier, used_cols: List[str]) -> pd.DataFrame:
    booster = model.get_booster()
    imp_maps = {
        "gain":        booster.get_score(importance_type="gain"),
        "total_gain":  booster.get_score(importance_type="total_gain"),
        "weight":      booster.get_score(importance_type="weight"),
        "cover":       booster.get_score(importance_type="cover"),
        "total_cover": booster.get_score(importance_type="total_cover"),
    }
    feats = set(); [feats.update(d.keys()) for d in imp_maps.values()]
    if not feats:
        if not used_cols: return pd.DataFrame(columns=["feature_key","feature","gain","weight","cover"])
        rows = [{"feature_key": f"f{i}", "feature": name, "gain": 0.0, "weight": 0.0, "cover": 0.0,
                 "total_gain": 0.0, "total_cover": 0.0} for i, name in enumerate(used_cols)]
        return pd.DataFrame(rows)[["feature_key","feature","gain","weight","cover"]]

    def alias(name: str) -> str:
        if name in used_cols: return name
        if name.startswith("f"):
            try:
                idx = int(name[1:])
                if 0 <= idx < len(used_cols): return used_cols[idx]
            except Exception: pass
        return name

    rows = []
    for fkey in sorted(feats):
        rows.append({
            "feature_key": fkey,
            "feature": alias(fkey),
            "gain": float(imp_maps["gain"].get(fkey, 0.0)),
            "weight": float(imp_maps["weight"].get(fkey, 0.0)),
            "cover": float(imp_maps["cover"].get(fkey, 0.0)),
            "total_gain": float(imp_maps["total_gain"].get(fkey, 0.0)),
            "total_cover": float(imp_maps["total_cover"].get(fkey, 0.0)),
        })
    df = pd.DataFrame(rows)
    for col in ["gain","weight","cover"]:
        if col not in df.columns: df[col] = 0.0
    if df["gain"].sum() == 0:
        df = df.sort_values(["weight","cover"], ascending=False, kind="stable")
    else:
        df = df.sort_values("gain", ascending=False, kind="stable")
    return df.reset_index(drop=True)

def plot_feature_importance(imp_df: pd.DataFrame, out_path: str, top_n: int = 30):
    if imp_df is None or imp_df.empty:
        print("[WARN] No feature importances to plot."); return
    sort_key = "gain" if ("gain" in imp_df.columns and imp_df["gain"].sum() > 0) else (
               "weight" if "weight" in imp_df.columns else None)
    if sort_key is None:
        print("[WARN] Importance DataFrame has no usable columns to plot."); return
    top = imp_df.sort_values(sort_key, ascending=False).head(top_n).iloc[::-1]
    if top.empty:
        print("[WARN] Top-N slice is empty; skipping importance plot."); return
    plt.figure(figsize=(8, max(4, 0.28 * len(top))))
    plt.barh(top["feature"], top[sort_key])
    plt.xlabel(sort_key.capitalize())
    plt.title(f"Top {len(top)} Feature Importances ({sort_key})")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

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

        n_pos = len(s_pos)
        n_neg = len(s_neg)
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
# FAST holdout masks
# =============================================================================
def make_holdout_masks(groups: np.ndarray, holdout_frac: float, seed: int = 42):
    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)
    if n_groups < 2:
        va_mask = np.zeros_like(codes, dtype=bool)
        tr_mask = ~va_mask
        return tr_mask, va_mask
    rng = np.random.RandomState(seed)
    n_val = max(1, int(np.ceil(n_groups * float(holdout_frac))))
    val_codes = rng.choice(n_groups, size=n_val, replace=False)
    flags = np.zeros(n_groups, dtype=bool); flags[val_codes] = True
    va_mask = flags[codes]; tr_mask = ~va_mask
    return tr_mask, va_mask

# =============================================================================
# Training
# =============================================================================
def train_holdout(cfg: Dict, Xdf: pd.DataFrame, y: pd.Series, groups: np.ndarray):
    tr_mask, va_mask = make_holdout_masks(groups, cfg["holdout_group_frac"], cfg.get("random_state", 42))
    X_tr, y_tr = Xdf.iloc[tr_mask], y.iloc[tr_mask]
    X_va, y_va = Xdf.iloc[va_mask], y.iloc[va_mask]

    if cfg.get("downsample_training_to_even", False):
        X_tr, y_tr, _ = _downsample_to_even(X_tr, y_tr, cfg.get("downsample_random_state", 42))

    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    spw = float(neg / max(pos, 1.0))

    params = dict(cfg["xgb_params"]); params["scale_pos_weight"] = spw
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_prob = model.predict_proba(X_va)[:, 1]
    report = f"[VALIDATION - HOLDOUT] (scale_pos_weight={spw:.3f})\n" + summarize_metrics(y_va.values, y_prob)
    return model, report, y_va.values, y_prob

def train_group_kfold(cfg: Dict, Xdf: pd.DataFrame, y: pd.Series, groups: np.ndarray):
    gkf = GroupKFold(n_splits=cfg["group_kfold_splits"])
    fold_reports, best_model, best_auc = [], None, -np.inf
    best_yva, best_pva = None, None
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xdf, y, groups=groups), start=1):
        X_tr, y_tr = Xdf.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = Xdf.iloc[va_idx], y.iloc[va_idx]

        if cfg.get("downsample_training_to_even", False):
            X_tr, y_tr, _ = _downsample_to_even(X_tr, y_tr, cfg.get("downsample_random_state", 42))

        pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
        spw = float(neg / max(pos, 1.0))

        params = dict(cfg["xgb_params"]); params["scale_pos_weight"] = spw
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        y_prob = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_prob) if len(np.unique(y_va)) > 1 else float("nan")
        fold_reports.append(f"[FOLD {fold}] spw={spw:.3f}\n" + summarize_metrics(y_va.values, y_prob))

        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_model = auc, m
            best_yva, best_pva = y_va.values, y_prob
    assert best_model is not None, "CV failed to produce a model."
    return best_model, "\n\n".join(fold_reports), best_yva, best_pva

# =============================================================================
# Main
# =============================================================================
def main(cfg: Dict = CFG):
    for p in (cfg["out_model_json"], cfg["out_used_features_json"], cfg["out_metrics_txt"],
              cfg["out_feat_csv"], cfg["out_plot_roc"], cfg["out_plot_pr"],
              cfg["out_plot_cm"], cfg["out_plot_feat"]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(cfg.get("out_hist_dir", os.path.join(cfg["base_dir"], "top_feature_hist")), exist_ok=True)

    Xdf, y, meta, used_cols = build_matrices(cfg)

    # Build composite episode-group key
    if not {"hashed_mother_id","episode_idx"}.issubset(meta.columns):
        raise ValueError("meta must contain 'hashed_mother_id' and 'episode_idx' for grouping.")
    groups = (meta["hashed_mother_id"].astype(str) + "#" + meta["episode_idx"].astype(str)).values

    # train
    if cfg["validation_strategy"] == "holdout":
        model, report, y_va, y_prob = train_holdout(cfg, Xdf, y, groups)
    elif cfg["validation_strategy"] == "group_kfold":
        model, report, y_va, y_prob = train_group_kfold(cfg, Xdf, y, groups)
    else:
        raise ValueError("validation_strategy must be 'holdout' or 'group_kfold'")

    # save artifacts
    model.save_model(cfg["out_model_json"])
    with open(cfg["out_used_features_json"], "w") as f:
        json.dump(used_cols, f, indent=2)
    with open(cfg["out_metrics_txt"], "w") as f:
        f.write(report + "\n")

    # plots + importances
    plot_roc(y_va, y_prob, cfg["out_plot_roc"])
    plot_pr(y_va, y_prob, cfg["out_plot_pr"])
    plot_confusion(y_va, y_prob, cfg["out_plot_cm"], threshold=0.5)

    imp_df = feature_importance_df(model, used_cols)
    imp_df.to_csv(cfg["out_feat_csv"], index=False)
    plot_feature_importance(imp_df, cfg["out_plot_feat"], top_n=cfg["feat_top_n"])

    # Histograms by label (all used features or top-N)
    top_n = int(cfg.get("hist_top_n", 20))
    bins = int(cfg.get("hist_bins", 40))
    out_hist_dir = cfg.get("out_hist_dir", os.path.join(cfg["base_dir"], "top_feature_hist"))

    features_to_plot: List[str] = []
    if cfg.get("hist_plot_all_used", False):
        features_to_plot = [f for f in used_cols if f in Xdf.columns]
    else:
        if imp_df is not None and not imp_df.empty:
            sort_key = "gain" if ("gain" in imp_df.columns and imp_df["gain"].sum() > 0) else (
                       "weight" if "weight" in imp_df.columns else None)
            if sort_key:
                top_feats = imp_df.sort_values(sort_key, ascending=False)["feature"].tolist()
                seen = set()
                features_to_plot = []
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

    print("[OK] Model:", cfg["out_model_json"])
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
