# labels_to_features_parquet.py
# Label snapshot features using blood dose/product timestamps.
# - Join key: (hashed_mother_id, episode_idx)  [episode_idx ≡ pregnancy_index]
# - A snapshot at time T (features over (T-3h, T]) is positive if any dose/product time t
#   for that birth satisfies: 0 <= T - t <= 3h  (inclusive)
# - Deterministic datetime parsing via exact regex/format; non-matching strings are DROPPED.
# - Checkpointed with recompute flags and runtime analysis per block.
# - Uses a per-birth numpy.searchsorted labeling (no merge_asof), avoiding sort errors.
#
# Outputs (in OUT_DIR):
#   - features_with_labels.parquet  (all original feature cols + label + trigger_dose_time)
#   - labels_vector.parquet         (hashed_mother_id, episode_idx, [pregnancy_index], snapshot_time, label)

from __future__ import annotations
import os
import re
import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = r"D:/PPH"

# Inputs
FEATURES_PARQUET = os.path.join(BASE_DIR, "features_all.parquet")
LABELS_CSV       = os.path.join(BASE_DIR, "processed_drugs_new.csv")

# Outputs
OUT_DIR              = os.path.join(BASE_DIR, "out_labels")
OUT_PARQUET_FEATS    = os.path.join(OUT_DIR, "features_with_labels.parquet")
OUT_PARQUET_LABELS   = os.path.join(OUT_DIR, "labels_vector.parquet")

# Cache files (checkpoints)
CACHE_DIR                 = os.path.join(OUT_DIR, "cache")
CACHE_FEATURES_PARSED     = os.path.join(CACHE_DIR, "features_parsed_sorted.parquet")
CACHE_DOSE_TIMES_TIDY     = os.path.join(CACHE_DIR, "dose_times_tidy.parquet")
CACHE_LABELED_PARQUET     = os.path.join(CACHE_DIR, "labeled.parquet")  # optional intermediate

# Recompute policy (toggle per step)
RECOMPUTE: Dict[str, bool] = {
    "features": False,   # parse/sanitize/sort features
    "doses":    True,   # melt/parse/sanitize dose/product times
    "labeling": True,    # label & final writes
}

# Window length (features are over (T - WINDOW_HOURS, T])
WINDOW_HOURS = 3

# Join keys & columns
KEY_COLS = ["hashed_mother_id", "episode_idx"]
SNAPSHOT_TIME_COL = "snapshot_time"
BY_KEY_COL = "_by_key"  # composite grouping key (fast group lookup)

# Allowed datetime formats (exact; non-matching are dropped)
# Labels: "%Y/%m/%d %H:%M:%S.%f" or "%Y/%m/%d %H:%M:%S"
LABELS_FMT_WITH_FRACT = "%Y-%m-%d %H:%M:%S.%f"
LABELS_FMT_NO_FRACT   = "%Y-%m-%d %H:%M:%S"
LABELS_RE_WITH_FRACT  = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$")
LABELS_RE_NO_FRACT    = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

# Features: "%Y-%m-%d %H:%M:%S.%f" or "%Y-%m-%d %H:%M:%S"
FEATS_FMT_WITH_FRACT  = "%Y-%m-%d %H:%M:%S.%f"
FEATS_FMT_NO_FRACT    = "%Y-%m-%d %H:%M:%S"
FEATS_RE_WITH_FRACT   = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$")
FEATS_RE_NO_FRACT     = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


# =========================
# UTILITIES
# =========================
def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

@contextmanager
def _rt(block_name: str):
    t0 = time.perf_counter()
    print(f"[START] {block_name} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[DONE ] {block_name} in {dt:,.2f}s")

def _nonempty_mask(s: pd.Series) -> pd.Series:
    # True for non-null and non-empty strings
    return s.notna() & (s.astype(str).str.strip() != "")

def _add_by_key(df: pd.DataFrame, key_cols: List[str], name: str = BY_KEY_COL) -> pd.DataFrame:
    """Create a single composite key for fast per-birth grouping."""
    df = df.copy()
    for k in key_cols:
        if k not in df.columns:
            raise ValueError(f"Missing key column: {k}")
        df[k] = df[k].astype(str)
    df[name] = df[key_cols[0]]
    for k in key_cols[1:]:
        df[name] = df[name] + "|" + df[k]
    return df

def _parse_features_time_exact(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m1 = s.str.match(FEATS_RE_WITH_FRACT)
    m2 = s.str.match(FEATS_RE_NO_FRACT)
    if m1.any():
        out.loc[m1] = pd.to_datetime(s.loc[m1], format=FEATS_FMT_WITH_FRACT)  # no errors='coerce'
    if m2.any():
        out.loc[m2] = pd.to_datetime(s.loc[m2], format=FEATS_FMT_NO_FRACT)    # no errors='coerce'
    return out  # non-matching remain NaT and will be dropped upstream

def _parse_labels_time_exact(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m1 = s.str.match(LABELS_RE_WITH_FRACT)
    m2 = s.str.match(LABELS_RE_NO_FRACT)
    if m1.any():
        out.loc[m1] = pd.to_datetime(s.loc[m1], format=LABELS_FMT_WITH_FRACT)  # no errors='coerce'
    if m2.any():
        out.loc[m2] = pd.to_datetime(s.loc[m2], format=LABELS_FMT_NO_FRACT)    # no errors='coerce'
    return out  # non-matching remain NaT and will be dropped upstream


# =========================
# LABELS: melt wide → tidy times
# =========================
def _melt_label_times(df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy dose/product timeline:
      returns columns: [hashed_mother_id, episode_idx, dose_time]
    Non-matching datetime strings are dropped (never coerced).
    """
    # Normalize episode_idx
    if "episode_idx" not in df_labels.columns:
        if "pregnancy_index" in df_labels.columns:
            df_labels = df_labels.copy()
            df_labels["episode_idx"] = df_labels["pregnancy_index"]
        else:
            raise ValueError("Labels must include 'episode_idx' or 'pregnancy_index'.")

    dose_cols = [c for c in df_labels.columns if c.lower().startswith("blood_dose_")]
    prod_cols = [c for c in df_labels.columns if c.lower().startswith("blood_product_")]
    time_cols = dose_cols + prod_cols

    if not time_cols:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    long = df_labels[["hashed_mother_id", "episode_idx"] + time_cols] \
        .melt(id_vars=["hashed_mother_id", "episode_idx"], var_name="kind", value_name="time_str")

    long = long.loc[_nonempty_mask(long["time_str"])].copy()
    if long.empty:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    dt = _parse_labels_time_exact(long["time_str"])
    kept = int(dt.notna().sum())
    dropped = int(len(dt) - kept)
    print(f"  [labels] parsed OK: {kept:,}; dropped (format mismatch): {dropped:,}")

    long = long.loc[dt.notna()].copy()
    if long.empty:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    long["dose_time"] = dt.loc[long.index]
    long = (long.drop(columns=["time_str", "kind"])
                .drop_duplicates(subset=["hashed_mother_id", "episode_idx", "dose_time"])
                .reset_index(drop=True))
    return long


# =========================
# CHECKPOINTED LOADERS
# =========================
def _load_or_compute_features() -> pd.DataFrame:
    if os.path.exists(CACHE_FEATURES_PARSED) and not RECOMPUTE["features"]:
        with _rt("Load features from cache"):
            return pd.read_parquet(CACHE_FEATURES_PARSED)

    with _rt("Read features parquet"):
        feats = pd.read_parquet(FEATURES_PARQUET)

    # Normalize episode_idx if needed
    if "episode_idx" not in feats.columns and "pregnancy_index" in feats.columns:
        feats = feats.rename(columns={"pregnancy_index": "episode_idx"})

    missing = set(KEY_COLS + [SNAPSHOT_TIME_COL]) - set(feats.columns)
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    with _rt("Parse & sanitize snapshot_time (drop non-matching formats)"):
        parsed = _parse_features_time_exact(feats[SNAPSHOT_TIME_COL])
        before = len(feats)
        feats = feats.loc[parsed.notna()].copy()
        feats[SNAPSHOT_TIME_COL] = parsed.loc[feats.index]
        print(f"  [features] kept: {len(feats):,}; dropped (bad time format): {before - len(feats):,}")

    with _rt("Normalize keys, mirror pregnancy_index, add composite key"):
        feats[KEY_COLS[0]] = feats[KEY_COLS[0]].astype(str)
        feats[KEY_COLS[1]] = feats[KEY_COLS[1]].astype(str)
        if "pregnancy_index" not in feats.columns:
            feats["pregnancy_index"] = feats["episode_idx"]
        feats = _add_by_key(feats, KEY_COLS, BY_KEY_COL)

    with _rt("Sort features by (by_key, snapshot_time)"):
        feats = feats.sort_values([BY_KEY_COL, SNAPSHOT_TIME_COL], kind="mergesort").reset_index(drop=True)

    with _rt("Save features cache"):
        feats.to_parquet(CACHE_FEATURES_PARSED, index=False)

    return feats

def _load_or_compute_dose_times() -> pd.DataFrame:
    if os.path.exists(CACHE_DOSE_TIMES_TIDY) and not RECOMPUTE["doses"]:
        with _rt("Load dose times from cache"):
            return pd.read_parquet(CACHE_DOSE_TIMES_TIDY)

    with _rt("Read labels CSV"):
        labels = pd.read_csv(LABELS_CSV, low_memory=False)

    with _rt("Normalize episode_idx from pregnancy_index if needed"):
        if "episode_idx" not in labels.columns and "pregnancy_index" in labels.columns:
            labels = labels.copy()
            labels["episode_idx"] = labels["pregnancy_index"]

    with _rt("Build tidy dose/product times (drop bad formats)"):
        doses = _melt_label_times(labels)

    with _rt("Add composite key + sort doses"):
        if not doses.empty:
            doses = _add_by_key(doses, KEY_COLS, BY_KEY_COL)
            doses = doses.sort_values([BY_KEY_COL, "dose_time"], kind="mergesort").reset_index(drop=True)

    with _rt("Save dose times cache"):
        doses.to_parquet(CACHE_DOSE_TIMES_TIDY, index=False)

    return doses


# =========================
# LABELING (searchsorted per birth)
# =========================
def _label_with_window(feats: pd.DataFrame, doses: pd.DataFrame) -> pd.DataFrame:
    """
    Per-birth labeling with numpy.searchsorted:
      For each birth (by_key):
        - Find index of most recent dose_time <= each snapshot T
        - Label 1 if 0 <= (T - dose_time) <= 3h
    """
    if doses.empty:
        out = feats.copy()
        out["label"] = np.int8(0)
        out["trigger_dose_time"] = pd.NaT
        return out

    with _rt("Prepare per-birth arrays"):
        # Build fast lookup: by_key -> dose times (ns and datetime)
        d_groups = doses.groupby(BY_KEY_COL, sort=False)["dose_time"]
        dose_map_ns = {}
        dose_map_dt = {}
        for k, s in d_groups:
            arr_dt = s.to_numpy(copy=False)
            dose_map_dt[k] = arr_dt
            dose_map_ns[k] = arr_dt.view("i8")  # datetime64[ns] -> int64 ns

        # Features index arrays per key
        f_groups = feats.groupby(BY_KEY_COL, sort=False).indices  # dict: key -> index positions (np.ndarray)

    with _rt("Compute labels via searchsorted"):
        n = len(feats)
        labels = np.zeros(n, dtype=np.int8)
        trigger_ns = np.full(n, np.iinfo("i8").min, dtype="i8")  # sentinel for NaT
        win_ns = np.int64(pd.Timedelta(hours=WINDOW_HOURS).value)

        processed = 0
        for k, idxs in f_groups.items():
            ts_dt = feats.loc[idxs, SNAPSHOT_TIME_COL].to_numpy(copy=False)
            ts_ns = ts_dt.view("i8")

            ds_ns = dose_map_ns.get(k, None)
            if ds_ns is None or ds_ns.size == 0:
                processed += len(idxs)
                continue

            # For each snapshot time, find rightmost dose_time <= T
            pos = np.searchsorted(ds_ns, ts_ns, side="right") - 1  # shape (len(idxs),)
            valid = pos >= 0
            if not valid.any():
                processed += len(idxs)
                continue

            prior_ds_ns = ds_ns[pos[valid]]
            delta = ts_ns[valid] - prior_ds_ns  # ns
            pos_mask = (delta >= 0) & (delta <= win_ns)

            if pos_mask.any():
                set_idx = np.asarray(idxs)[valid][pos_mask]
                labels[set_idx] = 1
                # store trigger dose time (ns)
                trigger_ns[set_idx] = prior_ds_ns[pos_mask]

            processed += len(idxs)

        # convert trigger_ns sentinel to NaT
        trigger_dt = trigger_ns.view("datetime64[ns]")
        # spots not set remain very negative; replace with NaT
        trigger_dt = pd.Series(trigger_dt)
        trigger_dt[trigger_ns == np.iinfo("i8").min] = pd.NaT

    out = feats.copy()
    out["label"] = labels.astype("int8")
    out["trigger_dose_time"] = trigger_dt.values
    return out


# =========================
# MAIN
# =========================
def main():
    _ensure_dirs()

    feats = _load_or_compute_features()
    doses = _load_or_compute_dose_times()

    if os.path.exists(CACHE_LABELED_PARQUET) and not RECOMPUTE["labeling"]:
        with _rt("Load labeled from cache"):
            labeled = pd.read_parquet(CACHE_LABELED_PARQUET)
    else:
        with _rt("Label snapshots by dose window (searchsorted)"):
            labeled = _label_with_window(feats, doses)
        with _rt("Save labeled cache"):
            labeled.to_parquet(CACHE_LABELED_PARQUET, index=False)

    # ---------- Final writes ----------
    with _rt("Write features_with_labels parquet"):
        labeled.to_parquet(OUT_PARQUET_FEATS, index=False)

    with _rt("Write labels_vector parquet"):
        cols = ["hashed_mother_id", "episode_idx", "pregnancy_index", SNAPSHOT_TIME_COL, "label"]
        cols = [c for c in cols if c in labeled.columns]
        labels_vector = labeled[cols].copy()
        labels_vector.to_parquet(OUT_PARQUET_LABELS, index=False)

    # ---------- Summary ----------
    n = len(labeled)
    pos = int(labeled["label"].sum())
    births = labeled[KEY_COLS].drop_duplicates().shape[0]
    print("\nSummary:")
    print(f"  Rows: {n:,}")
    print(f"  Births (unique {KEY_COLS}): {births:,}")
    print(f"  Positives: {pos:,} ({(pos / max(1, n)) * 100:.3f}%)")
    print(f"  Wrote: {OUT_PARQUET_FEATS}")
    print(f"  Wrote: {OUT_PARQUET_LABELS}")
    uniq = set(labeled["label"].unique().tolist())
    assert uniq.issubset({0, 1}), f"Unexpected labels present: {uniq}"


if __name__ == "__main__":
    main()
