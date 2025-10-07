"""
Train XGBoost on PPH features (episode-level) with caching, leakage guard, class
balancing (train only), and fast factorize-based holdout/group CV.

Works with the new builder that outputs:
  - features_all.parquet / labels_all.parquet
  - features_doc_subset.parquet / labels_doc_subset.parquet
  - feature_columns.json

Key points
----------
- Uses `episode_id` (not mother_id) as the grouping key.
- Blocks common leakage/prior-knowledge columns (recency/measured/process flags).
- Caches the result of build_matrices() to speed up iterations.
- Downsamples the training split’s majority class to even the classes (optional).
- Fast holdout via factorized group codes (no slow np.isin on strings).
"""

from __future__ import annotations
import json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict = {
    # -------- Paths (from builder) --------
    "base_dir": r"D:\PPH",
    "features_all_path": r"D:\PPH\features_all.parquet",
    "labels_all_path":   r"D:\PPH\labels_all.parquet",
    "features_doc_path": r"D:\PPH\features_doc_subset.parquet",
    "labels_doc_path":   r"D:\PPH\labels_doc_subset.parquet",
    "feature_cols_json": r"D:\PPH\feature_columns.json",

    # Which dataset to train on: "all" or "doc"
    "dataset_choice": "doc",

    # Label column (must exist in labels parquet)
    "target_col": "y_pph_future_60min",

    # Drop rows where target is NaN?
    "drop_na_target": True,

    # Restrict snapshots to within X minutes postpartum (per EPISODE; None = no cap)
    "max_minutes_postpartum": 360,  # e.g., 6 hours

    # ------------- Leakage guard -------------
    # Drop columns that are almost-certain leakage or care-process proxies
    "use_feature_blocklist": True,
    "leakage_blocklist_patterns": (
        "_recency_s",          # recency encodes staff measurement timing
        "_measured",           # whether measured at all (care process)
        "_given",              # drug given flags from stream
        "time_since_",         # time since a drug
    ),
    "leakage_blocklist_columns": (
        # care process / interventions likely after outcome
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
    "holdout_group_frac": 0.2,          # fraction of EPISODES in holdout
    "group_kfold_splits": 5,
    "group_key": "episode_id",          # <- important: use episodes, not mothers
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
        "tree_method": "hist",   # set to "gpu_hist" if you have a CUDA GPU
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
    "matrices_force_recompute": False,
    "matrices_cache_dir":   r"D:\PPH\.cache_xgb",
    "matrices_cache_X":     r"D:\PPH\.cache_xgb\Xdf.parquet",
    "matrices_cache_y":     r"D:\PPH\.cache_xgb\y.parquet",       # single column 'y'
    "matrices_cache_meta":  r"D:\PPH\.cache_xgb\meta.parquet",
    "matrices_cache_cols":  r"D:\PPH\.cache_xgb\used_cols.json",

    # --------- Balancing (train only) -------
    # Downsample MAJORITY class in TRAIN to the MINORITY class size
    "downsample_training_to_even": True,
    "downsample_random_state": 42,
}


# =============================================================================
# I/O helpers
# =============================================================================
def load_feature_label_tables(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = cfg["dataset_choice"].lower()
    if ds == "all":
        X = pd.read_parquet(cfg["features_all_path"])
        Y = pd.read_parquet(cfg["labels_all_path"])
    elif ds == "doc":
        X = pd.read_parquet(cfg["features_doc_path"])
        Y = pd.read_parquet(cfg["labels_doc_path"])
    else:
        raise ValueError("dataset_choice must be 'all' or 'doc'")

    # basic dtype hygiene
    for col in ("snapshot_time",):
        if col in X and not np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = pd.to_datetime(X[col], errors="coerce")
        if col in Y and not np.issubdtype(Y[col].dtype, np.datetime64):
            Y[col] = pd.to_datetime(Y[col], errors="coerce")

    # ensure ids are str
    for key in ("mother_id","episode_id","hashed_mother_id"):
        if key in X: X[key] = X[key].astype(str)
        if key in Y: Y[key] = Y[key].astype(str)

    return X, Y


def merge_features_labels(cfg: Dict, X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    target_col = cfg["target_col"]
    if target_col not in Y.columns:
        raise ValueError(f"Target column '{target_col}' not found in labels parquet")

    # merge on episode_id + snapshot_time (mother_id also present, but episode_id is the grouping key)
    keys = ["episode_id", "snapshot_time"]
    missing = [k for k in keys if k not in X.columns or k not in Y.columns]
    if missing:
        raise ValueError(f"Missing merge keys in X/Y: {missing} (need {keys})")

    XY = X.merge(Y[keys + [target_col]],
                 on=keys, how="inner", validate="one_to_one")
    return XY


def add_minutes_postpartum_by_episode(XY: pd.DataFrame, group_key: str = "episode_id") -> pd.DataFrame:
    # minutes since each episode's first snapshot (builder already constrains to postpartum)
    t0 = XY.groupby(group_key, as_index=False)["snapshot_time"].min().rename(
        columns={"snapshot_time": "episode_start"}
    )
    XY = XY.merge(t0, on=group_key, how="left")
    XY["mins_postpartum"] = (XY["snapshot_time"] - XY["episode_start"]).dt.total_seconds() / 60.0
    return XY


# =============================================================================
# Matrices (with caching)
# =============================================================================
def _blocklist_filter(cols: List[str], cfg: Dict) -> List[str]:
    if not cfg.get("use_feature_blocklist", True):
        return cols
    pats = tuple(cfg.get("leakage_blocklist_patterns", ()))
    bad_cols = set(cfg.get("leakage_blocklist_columns", ()))
    kept = []
    for c in cols:
        if c in bad_cols:  # exact match
            continue
        if any(p in c for p in pats):  # pattern match
            continue
        kept.append(c)
    return kept


def _compute_matrices(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    X, Y = load_feature_label_tables(cfg)
    XY = merge_features_labels(cfg, X, Y)
    XY = add_minutes_postpartum_by_episode(XY, cfg.get("group_key", "episode_id"))

    # Optional postpartum cap
    mpp = cfg.get("max_minutes_postpartum")
    if mpp is not None:
        XY = XY[XY["mins_postpartum"] <= float(mpp)].copy()

    # load engineered feature list produced by builder
    with open(cfg["feature_cols_json"], "r") as f:
        engineered_cols: List[str] = json.load(f)

    # baseline forbidden set
    forbid = {
        "mother_id", "hashed_mother_id", "episode_id",
        "snapshot_time", "episode_start",
        "mins_postpartum", cfg["target_col"],
    }
    # start from engineered list to keep stable column order
    used = [c for c in engineered_cols if (c in XY.columns and c not in forbid)]
    # apply leakage guard
    used = _blocklist_filter(used, cfg)

    # keep only numeric
    used = [c for c in used if pd.api.types.is_numeric_dtype(XY[c])]
    Xdf = XY[used].copy()
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    for c in Xdf.columns:
        Xdf[c] = Xdf[c].astype(np.float32)

    # drop constant/near-constant if requested
    if cfg.get("drop_constant_features", True):
        var = Xdf.var(axis=0, numeric_only=True)
        keep = var.index[var.values > float(cfg.get("min_feature_variance", 1e-12))].tolist()
        dropped = sorted(set(used) - set(keep))
        if dropped:
            print(f"[INFO] Dropping {len(dropped)} near-constant features.")
        used = keep
        Xdf = Xdf[used]

    # target
    if cfg.get("drop_na_target", True):
        XY = XY[~XY[cfg["target_col"]].isna()].copy()
        Xdf = Xdf.loc[XY.index]
    y = XY[cfg["target_col"]].astype(float)

    # meta (for grouping & diagnostics)
    meta_cols = ["mother_id","episode_id","snapshot_time","mins_postpartum"]
    meta_cols = [c for c in meta_cols if c in XY.columns]
    meta = XY[meta_cols].copy()

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
        y = pd.read_parquet(yp)["y"].astype(float)
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
    gain   = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover  = booster.get_score(importance_type="cover")
    feats = sorted(set(gain)|set(weight)|set(cover))

    def alias(name: str) -> str:
        if name in used_cols: return name
        if name.startswith("f"):
            try:
                idx = int(name[1:])
                if 0 <= idx < len(used_cols): return used_cols[idx]
            except Exception:
                pass
        return name

    rows = []
    for f in feats:
        rows.append({
            "feature_key": f,
            "feature": alias(f),
            "gain": gain.get(f, 0.0),
            "weight": weight.get(f, 0.0),
            "cover": cover.get(f, 0.0),
        })
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp_df: pd.DataFrame, out_path: str, top_n: int = 30):
    top = imp_df.sort_values("gain", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.28 * len(top))))
    plt.barh(top["feature"], top["gain"])
    plt.xlabel("Gain"); plt.title(f"Top {len(top)} Feature Importances (gain)")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


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
# FAST holdout masks (factorize)
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
    va_mask = flags[codes]
    tr_mask = ~va_mask
    return tr_mask, va_mask


# =============================================================================
# Training
# =============================================================================
def train_holdout(cfg: Dict, Xdf: pd.DataFrame, y: pd.Series, groups: np.ndarray):
    tr_mask, va_mask = make_holdout_masks(
        groups=groups,
        holdout_frac=cfg["holdout_group_frac"],
        seed=cfg.get("random_state", 42),
    )
    X_tr, y_tr = Xdf.iloc[tr_mask], y.iloc[tr_mask]
    X_va, y_va = Xdf.iloc[va_mask], y.iloc[va_mask]

    if cfg.get("downsample_training_to_even", True):
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

        if cfg.get("downsample_training_to_even", True):
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
    # ensure output dirs
    for p in (cfg["out_model_json"], cfg["out_used_features_json"], cfg["out_metrics_txt"],
              cfg["out_feat_csv"], cfg["out_plot_roc"], cfg["out_plot_pr"],
              cfg["out_plot_cm"], cfg["out_plot_feat"]):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # matrices (cached)
    Xdf, y, meta, used_cols = build_matrices(cfg)
    group_key = cfg.get("group_key", "episode_id")
    if group_key not in meta.columns:
        raise ValueError(f"Group key '{group_key}' not found in meta columns.")
    groups = meta[group_key].astype(str).values

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
