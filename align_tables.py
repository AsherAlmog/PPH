# align_tables.py
# End-to-end alignment:
#   - Build wide vitals/labs/drugs
#   - Episode split by 183 days
#   - Align each episode to birth
#   - Align labels (by pregnancy_index) to birth + episode
#   - Save timeline (PKL) and labels (CSV) separately
#   - Cache results for every major timed step

from __future__ import annotations

import os
import re
import time as _time
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = r"D:\PPH"

# Inputs
PATH_MEASUREMENTS_LONG = os.path.join(BASE_DIR, "MF_mother_Measurements_20250812.csv")
PATH_LABS              = os.path.join(BASE_DIR, "MF_mother_labs_20250812.csv")
PATH_DRUGS             = os.path.join(BASE_DIR, "MF_mother_drugs_20250812.csv")
PATH_BIRTHS            = os.path.join(BASE_DIR, "MF_FETAL_TABL_20250812_132000.csv")
PATH_LABELS            = os.path.join(BASE_DIR, "labels_drugs.csv")  # labels CSV with pregnancy_index + start_date + label

# Outputs
OUT_TIMELINE_PKL       = os.path.join(BASE_DIR, "pph_wide_timeline.pkl")
OUT_LABELS_ALIGNED_CSV = os.path.join(BASE_DIR, "pph_labels_aligned.csv")

# Caches for each major step
CACHE_VITALS_WIDE      = os.path.join(BASE_DIR, ".cache_vitals_wide.pkl")
CACHE_LABS_WIDE        = os.path.join(BASE_DIR, ".cache_labs_wide.pkl")
CACHE_DRUGS_WIDE       = os.path.join(BASE_DIR, ".cache_drugs_wide.pkl")
CACHE_BIRTHS_ALL       = os.path.join(BASE_DIR, ".cache_births_all.pkl")
CACHE_MERGED_WIDE      = os.path.join(BASE_DIR, ".cache_merged_wide.pkl")
CACHE_EP_BOUNDS        = os.path.join(BASE_DIR, ".cache_episode_bounds.pkl")
CACHE_TIMELINE_ALIGNED = os.path.join(BASE_DIR, ".cache_timeline_aligned.pkl")
CACHE_LABELS_ALIGNED   = os.path.join(BASE_DIR, ".cache_labels_aligned.csv")

# Global switches
FORCE_RECOMPUTE        = False    # rebuild everything ignoring caches
EPISODE_GAP_DAYS       = 183      # ~6 months
TIME_WINDOW_BEFORE_H   = 24       # prune window around birth (optional). None to disable
TIME_WINDOW_AFTER_H    = 24*5

# =============================================================================
# Strict datetime formats (project-wide)
# =============================================================================
# Non-label timestamp format (everywhere except labels)
DT_FMT_MAIN = "%Y/%m/%d %H:%M:%S.%f"
# Label timestamp accepted formats (two-digit year, hyphens)
DT_FMT_LABELS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")

# =============================================================================
# Timing helper
# =============================================================================
_TIMINGS: Dict[str, float] = {}

@contextmanager
def _timer(name: str):
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        dt = _time.perf_counter() - t0
        _TIMINGS[name] = _TIMINGS.get(name, 0.0) + dt
        print(f"[TIMER] {name}: {dt:.3f}s")

# =============================================================================
# IO helpers
# =============================================================================
def _read_csv_usecols(path: str, wanted_cols: List[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    use = [c for c in wanted_cols if c in header.columns]
    return pd.read_csv(path, usecols=use)

def save_pkl(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)

def load_pkl(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

# =============================================================================
# Misc helpers
# =============================================================================
def fmt_signed_hms_tenths(delta_seconds):
    if pd.isna(delta_seconds): return np.nan
    sign = "-" if delta_seconds < 0 else ""
    x = abs(delta_seconds)
    h = int(x // 3600)
    rem = x - 3600 * h
    m = int(rem // 60)
    s = rem - 60 * m
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"

def safe_col(name: str) -> str:
    s = "unknown" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name).strip()
    s = re.sub(r"[^0-9A-Za-z_+]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "unknown"

# =============================================================================
# Load raw
# =============================================================================
def load_measurements(path: str) -> pd.DataFrame:
    cols = ["hashed_mother_id", "er_admission_date", "department_admission", "department_discharge",
            "parameter_time", "Parameter_Name", "ResultNumeric", "flag"]
    return _read_csv_usecols(path, cols)

def load_labs(path: str) -> pd.DataFrame:
    cols = ["hashed_mother_id", "ResultTime", "TestName", "Result"]
    return _read_csv_usecols(path, cols)

def load_drugs(path: str) -> pd.DataFrame:
    cols = ["hashed_mother_id", "ExecutionTime", "DrugName"]
    return _read_csv_usecols(path, cols)

def load_births(path: str) -> pd.DataFrame:
    cols = ["hashed_mother_id", "child_birth_date", "birth_time"]
    return _read_csv_usecols(path, cols)

def load_labels(path: str) -> pd.DataFrame:
    # Expect: hashed_mother_id, pregnancy_index, start_date, (optional) blood dose/product, label
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

# =============================================================================
# Vitals → wide
# =============================================================================
WANTED_VITALS = ["BP - Mean", "diastol", "sistol", "saturation", "heat", "pulse"]
HE_TO_EN_VITALS = {
    "לחץ סיסטולי": "sistol",
    "לחץ דיאסטולי": "diastol",
    "לחץ דם ממוצע": "BP - Mean",
    "סטורציה": "saturation",
    "חום": "heat",
    "דופק": "pulse",
}

def get_or_build_vitals_wide(meas: pd.DataFrame, force: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (not force) and os.path.exists(CACHE_VITALS_WIDE):
        vitals_wide = load_pkl(CACHE_VITALS_WIDE)
        anchors = (vitals_wide[["hashed_mother_id","event_time_abs"]]
                   .sort_values(["hashed_mother_id","event_time_abs"])
                   .groupby("hashed_mother_id", as_index=False).first()
                   .rename(columns={"event_time_abs":"anchor_er"}))
        return vitals_wide, anchors

    df = meas.copy()

    # Strict datetime parsing for measurement timestamp
    df["parameter_time"] = pd.to_datetime(df["parameter_time"], format=DT_FMT_MAIN)

    # Basic cleaning
    df = df.dropna(subset=["hashed_mother_id","parameter_time","ResultNumeric"])
    if "flag" in df.columns:
        df = df[df["flag"].fillna(0).eq(0)]

    df["param_norm"] = df["Parameter_Name"].map(HE_TO_EN_VITALS).fillna(df["Parameter_Name"])
    df = df[df["param_norm"].isin(WANTED_VITALS)]

    df = df.sort_values(["hashed_mother_id","parameter_time"]).drop_duplicates(
        subset=["hashed_mother_id","parameter_time","param_norm"], keep="last"
    )

    vitals_wide = (df.set_index(["hashed_mother_id","parameter_time","param_norm"])["ResultNumeric"]
                     .unstack("param_norm")
                     .reset_index()
                     .rename(columns={"parameter_time":"event_time_abs"}))

    for v in WANTED_VITALS:
        if v not in vitals_wide.columns:
            vitals_wide[v] = np.nan

    vitals_wide = vitals_wide[["hashed_mother_id","event_time_abs"] + WANTED_VITALS]

    anchors = (vitals_wide[["hashed_mother_id","event_time_abs"]]
               .sort_values(["hashed_mother_id","event_time_abs"])
               .groupby("hashed_mother_id", as_index=False).first()
               .rename(columns={"event_time_abs":"anchor_er"}))

    save_pkl(vitals_wide, CACHE_VITALS_WIDE)
    return vitals_wide, anchors

# =============================================================================
# Births → all + pregnancy_index
# =============================================================================
def compute_births_all(births: pd.DataFrame, anchors: pd.DataFrame, force: bool) -> pd.DataFrame:
    """
    Returns: ['hashed_mother_id','birth_datetime','pregnancy_index'].
    'pregnancy_index' enumerates births chronologically per mother (1..K).
    """
    if (not force) and os.path.exists(CACHE_BIRTHS_ALL):
        df = load_pkl(CACHE_BIRTHS_ALL)
        if "pregnancy_index" not in df.columns:
            if df.empty:
                df["pregnancy_index"] = pd.Series(dtype="Int32")
            else:
                df = df.dropna(subset=["hashed_mother_id"]).copy()
                df["pregnancy_index"] = (
                    df.sort_values(["hashed_mother_id", "birth_datetime"])
                      .groupby("hashed_mother_id").cumcount() + 1
                ).astype("Int32")
        return df

    b = births.copy()

    # Strict parsing: both columns are absolute datetimes in DT_FMT_MAIN
    if "child_birth_date" in b.columns:
        b["child_birth_date_dt"] = pd.to_datetime(b["child_birth_date"], format=DT_FMT_MAIN)
    else:
        b["child_birth_date_dt"] = pd.NaT

    if "birth_time" in b.columns:
        b["birth_time_dt"] = pd.to_datetime(b["birth_time"], format=DT_FMT_MAIN)
    else:
        b["birth_time_dt"] = pd.NaT

    # Choose birth_time when present; else use child_birth_date
    b["birth_datetime"] = b["birth_time_dt"].where(b["birth_time_dt"].notna(), b["child_birth_date_dt"])

    births_all = (
        b.dropna(subset=["hashed_mother_id", "birth_datetime"])
         .assign(hashed_mother_id=lambda df: df["hashed_mother_id"].astype(str))
         .sort_values(["hashed_mother_id", "birth_datetime"])
         [["hashed_mother_id", "birth_datetime"]]
         .reset_index(drop=True)
    )

    if births_all.empty:
        births_all["pregnancy_index"] = pd.Series(dtype="Int32")
    else:
        births_all["pregnancy_index"] = (
            births_all.groupby("hashed_mother_id").cumcount() + 1
        ).astype("Int32")

    save_pkl(births_all, CACHE_BIRTHS_ALL)
    return births_all

# =============================================================================
# Labs → wide
# =============================================================================
def get_or_build_labs_wide(labs: pd.DataFrame, anchors: pd.DataFrame, force: bool) -> pd.DataFrame:
    if (not force) and os.path.exists(CACHE_LABS_WIDE):
        return load_pkl(CACHE_LABS_WIDE)

    L = labs.copy()
    L["event_time_abs"] = pd.to_datetime(L["ResultTime"], format=DT_FMT_MAIN)

    L["TestName_safe"] = L.get("TestName").map(lambda x: safe_col(str(x).upper()))
    L["Result_num"] = pd.to_numeric(L.get("Result"))

    L = L.sort_values(["hashed_mother_id","event_time_abs"]).drop_duplicates(
        subset=["hashed_mother_id","event_time_abs","TestName_safe"], keep="last"
    )
    labs_wide = (L.set_index(["hashed_mother_id","event_time_abs","TestName_safe"])["Result_num"]
                   .unstack("TestName_safe").reset_index())

    save_pkl(labs_wide, CACHE_LABS_WIDE)
    return labs_wide

# =============================================================================
# Drugs → wide binary (PPH-related)
# =============================================================================
def normalize_pph_drug_name(drug: str) -> Optional[str]:
    if drug is None or (isinstance(drug, float) and np.isnan(drug)):
        return None
    s = str(drug).upper()
    if "OXYTOCIN" in s: return "OXYTOCIN"
    if "MISOPROSTOL" in s or "CYTOTEC" in s: return "MISOPROSTOL"
    if "METHYLERGONOVINE" in s or "METHERGIN" in s: return "METHYLERGONOVINE"
    if "LACTATED RINGERS" in s or "HARTMAN" in s or "HARTMAN`S" in s: return "LACTATED_RINGERS"
    if "SODIUM CHLORIDE" in s and "0.9%" in s: return "SODIUM_CHLORIDE_0_9"
    return None

def get_or_build_drugs_wide(drugs: pd.DataFrame, anchors: pd.DataFrame, force: bool) -> pd.DataFrame:
    if (not force) and os.path.exists(CACHE_DRUGS_WIDE):
        return load_pkl(CACHE_DRUGS_WIDE)

    d = drugs.copy()
    d["pph_norm"] = d.get("DrugName").map(normalize_pph_drug_name)
    d = d[d["pph_norm"].notna()].copy()
    if d.empty:
        save_pkl(pd.DataFrame(columns=["hashed_mother_id","event_time_abs"]), CACHE_DRUGS_WIDE)
        return pd.DataFrame(columns=["hashed_mother_id","event_time_abs"])

    d["event_time_abs"] = pd.to_datetime(d["ExecutionTime"], format=DT_FMT_MAIN)

    d["flag"] = 1
    d = d.sort_values(["hashed_mother_id","event_time_abs"]).drop_duplicates(
        subset=["hashed_mother_id","event_time_abs","pph_norm"], keep="last"
    )
    drugs_wide = (d.set_index(["hashed_mother_id","event_time_abs","pph_norm"])["flag"]
                    .unstack("pph_norm").fillna(0).astype("int8").reset_index())

    save_pkl(drugs_wide, CACHE_DRUGS_WIDE)
    return drugs_wide

# =============================================================================
# Merge wide
# =============================================================================
def outer_merge_wide(vitals_wide: pd.DataFrame, labs_wide: pd.DataFrame, drugs_wide: pd.DataFrame,
                     force: bool) -> pd.DataFrame:
    if (not force) and os.path.exists(CACHE_MERGED_WIDE):
        return load_pkl(CACHE_MERGED_WIDE)
    merged = vitals_wide.merge(labs_wide, how="outer", on=["hashed_mother_id","event_time_abs"])
    merged = merged.merge(drugs_wide, how="outer", on=["hashed_mother_id","event_time_abs"])
    save_pkl(merged, CACHE_MERGED_WIDE)
    return merged

# =============================================================================
# Episode split + birth alignment (threaded)
# =============================================================================
def add_episodes_and_align_births_threaded(
    merged: pd.DataFrame,
    births_all: pd.DataFrame,
    episode_gap_days: int = EPISODE_GAP_DAYS,
    n_jobs: int = -1,
    batch_size: str | int = "auto",
    force: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - timeline with ['episode_idx','birth_datetime','t_from_birth_sec','t_from_birth_str']
      - episode bounds table: ['hashed_mother_id','episode_idx','ep_start','ep_end','birth_datetime']
    Caches the aligned timeline and episode bounds.
    """
    if (not force) and os.path.exists(CACHE_TIMELINE_ALIGNED) and os.path.exists(CACHE_EP_BOUNDS):
        return load_pkl(CACHE_TIMELINE_ALIGNED), load_pkl(CACHE_EP_BOUNDS)

    df = merged.copy()
    df["hashed_mother_id"] = df["hashed_mother_id"].astype(str)
    # event_time_abs already parsed strictly upstream

    df = df.dropna(subset=["event_time_abs"])
    df = df.sort_values(["hashed_mother_id","event_time_abs"], kind="stable")

    # compute episode_idx
    slim = df[["hashed_mother_id","event_time_abs"]].copy()
    gap = pd.to_timedelta(int(episode_gap_days), unit="D")
    dt = slim.groupby("hashed_mother_id")["event_time_abs"].diff()
    new_ep = (dt.isna()) | (dt > gap)
    slim["episode_idx"] = new_ep.groupby(slim["hashed_mother_id"]).cumsum().astype("int32")

    # join episode_idx back
    df = df.join(
        slim.set_index(["hashed_mother_id","event_time_abs"])["episode_idx"],
        on=["hashed_mother_id","event_time_abs"],
        how="left",
    )

    # bounds
    bounds = (slim.groupby(["hashed_mother_id","episode_idx"], as_index=False)["event_time_abs"]
                 .agg(ep_start="min", ep_end="max"))

    # births per mother arrays
    B = births_all.copy()
    B["hashed_mother_id"] = B["hashed_mother_id"].astype(str)
    # birth_datetime already strict parsed upstream
    B = B.dropna(subset=["birth_datetime"]).sort_values(
        ["hashed_mother_id","birth_datetime"], kind="stable"
    )
    births_by_mother: Dict[str, np.ndarray] = {
        mid: grp["birth_datetime"].to_numpy(dtype="datetime64[ns]")
        for mid, grp in B.groupby("hashed_mother_id", sort=False)
    }

    # per-mother tasks
    tasks: List[Tuple[str, pd.DataFrame]] = [
        (mid, grp[["episode_idx","ep_start","ep_end"]].reset_index(drop=True))
        for mid, grp in bounds.groupby("hashed_mother_id", sort=False)
    ]

    def _align_for_mother(mid: str, eps: pd.DataFrame) -> pd.DataFrame:
        out = eps.copy()
        b = births_by_mother.get(mid, None)
        if b is None or b.size == 0:
            out["birth_datetime"] = pd.NaT
            out["hashed_mother_id"] = mid
            return out
        s = eps["ep_start"].to_numpy(dtype="datetime64[ns]")
        e = eps["ep_end"].to_numpy(dtype="datetime64[ns]")

        left  = np.searchsorted(b, s, side="left")
        right = np.searchsorted(b, e, side="right")

        # global nearest (fallback)
        left_clip = np.clip(left, 0, len(b) - 1)
        leftm1    = np.clip(left - 1, 0, len(b) - 1)
        d_left    = np.abs(b[left_clip] - s)
        d_leftm1  = np.abs(b[leftm1]   - s)
        idx_global = np.where(d_left <= d_leftm1, left_clip, leftm1)

        # inside-window candidates
        cand_hi = left
        cand_lo = left - 1
        valid_hi = (cand_hi >= left) & (cand_hi < right)
        valid_lo = (cand_lo >= left) & (cand_lo < right)

        chi = np.clip(cand_hi, 0, len(b) - 1)
        clo = np.clip(cand_lo, 0, len(b) - 1)

        diff_hi = np.where(valid_hi, np.abs(b[chi] - s), np.timedelta64("NaT","ns"))
        diff_lo = np.where(valid_lo, np.abs(b[clo] - s), np.timedelta64("NaT","ns"))

        idx_inside = chi.copy()
        only_lo = (~valid_hi) & valid_lo
        both = valid_hi & valid_lo
        closer_lo = both & (diff_lo < diff_hi)
        idx_inside[only_lo] = clo[only_lo]
        idx_inside[closer_lo] = clo[closer_lo]
        valid_inside_any = valid_hi | valid_lo

        chosen_idx = np.where(valid_inside_any, idx_inside, idx_global)
        out["birth_datetime"] = pd.to_datetime(b[chosen_idx])
        out["hashed_mother_id"] = mid
        return out

    # threaded parallel
    if len(tasks) == 0:
        df["birth_datetime"] = pd.NaT
        df["t_from_birth_sec"] = np.nan
        df["t_from_birth_str"] = np.nan
        save_pkl(df, CACHE_TIMELINE_ALIGNED)
        save_pkl(bounds.assign(birth_datetime=pd.NaT), CACHE_EP_BOUNDS)
        return df, bounds.assign(birth_datetime=pd.NaT)

    chosen_parts = Parallel(
        n_jobs=n_jobs,
        backend="threading",
        batch_size=batch_size,
        prefer="threads",
    )(delayed(_align_for_mother)(mid, eps) for mid, eps in tasks)

    ep_births = pd.concat(chosen_parts, ignore_index=True)
    bounds2 = bounds.merge(
        ep_births[["hashed_mother_id","episode_idx","birth_datetime"]],
        on=["hashed_mother_id","episode_idx"], how="left", validate="one_to_one"
    )

    # attach to rows
    df = df.merge(
        bounds2[["hashed_mother_id","episode_idx","birth_datetime"]],
        on=["hashed_mother_id","episode_idx"], how="left", validate="many_to_one"
    )

    # deltas vs birth
    df["t_from_birth_sec"] = (df["event_time_abs"] - df["birth_datetime"]).dt.total_seconds()
    df["t_from_birth_str"] = df["t_from_birth_sec"].apply(fmt_signed_hms_tenths)

    save_pkl(df, CACHE_TIMELINE_ALIGNED)
    save_pkl(bounds2, CACHE_EP_BOUNDS)
    return df, bounds2

# =============================================================================
# Optional pruning around birth
# =============================================================================
def prune_to_birth_window(df: pd.DataFrame, before_h: Optional[float], after_h: Optional[float]) -> pd.DataFrame:
    if before_h is None and after_h is None:
        return df
    lo = -float(before_h) * 3600 if before_h is not None else -np.inf
    hi = float(after_h) * 3600 if after_h is not None else np.inf
    mask = df["t_from_birth_sec"].between(lo, hi)
    return df.loc[mask].copy()

# =============================================================================
# Order & classify columns (for pretty output ordering)
# =============================================================================
def order_and_classify_columns(merged: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    key_cols = ["hashed_mother_id","event_time_abs","birth_datetime","t_from_birth_sec","t_from_birth_str","episode_idx"]
    vitals_cols = [c for c in WANTED_VITALS if c in merged.columns]
    exclude = set(key_cols + vitals_cols)
    other_cols = [c for c in merged.columns if c not in exclude]

    lab_cols, drug_cols = [], []
    for c in other_cols:
        vals = merged[c]
        uniq = set(pd.unique(vals.dropna()))
        if len(uniq) > 0 and uniq.issubset({0,1}):
            drug_cols.append(c)
        else:
            lab_cols.append(c)

    ordered = merged[[c for c in key_cols if c in merged.columns] + vitals_cols + sorted(lab_cols) + sorted(drug_cols)]
    return ordered, vitals_cols, lab_cols, drug_cols

# =============================================================================
# Collapse near-duplicates (<1s apart)
# =============================================================================
def collapse_near_duplicates_fast(df: pd.DataFrame,
                                  vitals_cols: List[str],
                                  lab_cols: List[str],
                                  drug_cols: List[str],
                                  threshold_seconds: float = 1.0) -> pd.DataFrame:
    df = df.sort_values(["hashed_mother_id","event_time_abs"], kind="stable").reset_index(drop=True)
    dt = df.groupby("hashed_mother_id")["event_time_abs"].diff().dt.total_seconds()
    new_cluster = (dt.isna()) | (dt >= threshold_seconds)
    df["__cluster__"] = new_cluster.groupby(df["hashed_mother_id"]).cumsum()

    gk = ["hashed_mother_id","__cluster__"]
    cols_fill = [c for c in vitals_cols + lab_cols if c in df.columns]
    if cols_fill:
        df[cols_fill] = df.groupby(gk, sort=False)[cols_fill].ffill()

    agg = {
        "event_time_abs": "min",
        "birth_datetime": "first",
        "episode_idx": "first",
    }
    for c in cols_fill:
        agg[c] = "last"
    for c in drug_cols:
        if c in df.columns:
            agg[c] = "max"

    compact = (df.groupby(gk, as_index=False, sort=False).agg(agg)
                 .drop(columns="__cluster__", errors="ignore"))
    compact["t_from_birth_sec"] = (compact["event_time_abs"] - compact["birth_datetime"]).dt.total_seconds()
    compact["t_from_birth_str"] = compact["t_from_birth_sec"].apply(fmt_signed_hms_tenths)

    key_cols = ["hashed_mother_id","event_time_abs","birth_datetime","t_from_birth_sec","t_from_birth_str","episode_idx"]
    final_cols = [c for c in key_cols + vitals_cols + lab_cols + drug_cols if c in compact.columns]
    return compact[final_cols].sort_values(["hashed_mother_id","event_time_abs"], kind="stable").reset_index(drop=True)

# =============================================================================
# Labels alignment (separate file; do NOT merge into timeline)
# =============================================================================
def _parse_labels_start_date_strict(series: pd.Series) -> pd.Series:
    """
    Parse labels' start_date with EXACT accepted formats:
      - "%y-%m-%d %H:%M:%S.%f"
      - "%y-%m-%d %H:%M:%S"
    No errors='coerce'. Raises if any non-null value fails both formats.
    """
    fmts = DT_FMT_LABELS
    try:
        return pd.to_datetime(series, format=fmts[0])
    except Exception:
        # fallback per-element to the second format; raise on any failure
        out = pd.Series(index=series.index, dtype="datetime64[ns]")
        failed = []
        for i, v in series.items():
            if pd.isna(v):
                out.at[i] = pd.NaT
                continue
            ok = False
            for fmt in fmts:
                try:
                    out.at[i] = pd.to_datetime(v, format=fmt)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                failed.append((i, v))
        if failed:
            sample = ", ".join(f"idx={i}, value={val!r}" for i, val in failed[:5])
            raise ValueError(
                f"Failed to parse some label start_date values. Accepted formats: {fmts}. "
                f"Examples: {sample}"
            )
        return out

from joblib import Parallel, delayed

from joblib import Parallel, delayed

def build_labels_aligned(labels_raw: pd.DataFrame,
                         births_all: pd.DataFrame,   # unused, kept for signature compatibility
                         episode_bounds: pd.DataFrame,
                         force: bool,
                         n_jobs: int = -1,
                         batch_size: str | int = "auto",
                         backend: str = "threading") -> pd.DataFrame:
    """
    Align labels to episodes using ONLY time containment and enforce episode_idx == pregnancy_index per mother.

    Rules:
      - Parse labels start_date strictly (DT_FMT_LABELS).
      - For each mother, normalize episodes in episode_bounds so that:
            episode_idx := order by ep_start (1..K)  == pregnancy_index
      - A label row is assigned to the unique episode whose [ep_start, ep_end] contains event_time_abs.
      - No birth equality checks, no nearest fallback.

    Output columns:
      ['hashed_mother_id','pregnancy_index','episode_idx','event_time_abs','birth_datetime',
       'label','label_pos','blood_dose','blood_product']
    """
    # Cache short-circuit
    if (not force) and os.path.exists(CACHE_LABELS_ALIGNED):
        return pd.read_csv(CACHE_LABELS_ALIGNED, parse_dates=["event_time_abs","birth_datetime"])

    # Empty labels -> empty aligned
    if labels_raw is None or labels_raw.empty:
        out = pd.DataFrame(columns=[
            "hashed_mother_id","pregnancy_index","episode_idx","event_time_abs","birth_datetime",
            "label","label_pos","blood_dose","blood_product"
        ])
        save_csv(out, CACHE_LABELS_ALIGNED)
        return out

    # --- Prepare labels (strict parsing) ---
    L = labels_raw.copy()

    # rename optional columns
    rename_map = {}
    for c in L.columns:
        cl = c.lower()
        if cl == "blood dose":    rename_map[c] = "blood_dose"
        if cl == "blood product": rename_map[c] = "blood_product"
    if rename_map:
        L = L.rename(columns=rename_map)

    req = {"hashed_mother_id","pregnancy_index","start_date","label"}
    missing = req - set(L.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    # IDs + strict datetime for labels
    L["hashed_mother_id"] = L["hashed_mother_id"].astype(str)
    L["pregnancy_index"]  = pd.to_numeric(L["pregnancy_index"]).astype("Int64")  # numeric coercion is fine
    # strict: DT_FMT_LABELS must be defined at module top as ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    def _parse_labels(series: pd.Series) -> pd.Series:
        fmts = DT_FMT_LABELS
        try:
            return pd.to_datetime(series, format=fmts[0])
        except Exception:
            out = pd.Series(index=series.index, dtype="datetime64[ns]")
            failed = []
            for i, v in series.items():
                if pd.isna(v):
                    out.at[i] = pd.NaT
                    continue
                ok = False
                for fmt in fmts:
                    try:
                        out.at[i] = pd.to_datetime(v, format=fmt)
                        ok = True
                        break
                    except Exception:
                        continue
                if not ok:
                    failed.append((i, v))
            if failed:
                sample = ", ".join(f"idx={i}, value={val!r}" for i, val in failed[:5])
                raise ValueError(
                    f"Failed to parse some label start_date values. Accepted formats: {fmts}. "
                    f"Examples: {sample}"
                )
            return out

    L["event_time_abs"] = _parse_labels(L["start_date"])

    L["label"] = pd.to_numeric(L["label"]).fillna(0).astype("int16")
    L["label_pos"] = L["label"].isin([1,2]).astype("int8")
    if "blood_dose" in L.columns:
        L["blood_dose"] = pd.to_numeric(L["blood_dose"]).astype("float32")
    else:
        L["blood_dose"] = np.nan
    if "blood_product" in L.columns:
        L["blood_product"] = pd.to_numeric(L["blood_product"]).astype("float32")
    else:
        L["blood_product"] = np.nan

    # We only keep rows with a valid event_time_abs
    L = L.dropna(subset=["event_time_abs"])

    # --- Prepare episode bounds, enforce episode_idx == pregnancy_index per mother ---
    E = episode_bounds.copy()
    E["hashed_mother_id"] = E["hashed_mother_id"].astype(str)

    # Normalize episode_idx: per mother, sort by ep_start and set episode_idx = 1..K (pregnancy index)
    E = E.sort_values(["hashed_mother_id", "ep_start"]).copy()
    E["episode_idx"] = E.groupby("hashed_mother_id").cumcount() + 1  # 1..K per mother (this is pregnancy_index)
    # Keep birth_datetime if present; else create NaT for consistent output
    if "birth_datetime" not in E.columns:
        E["birth_datetime"] = pd.NaT

    # Build fast lookup dict per mother
    e_by_mid: Dict[str, Dict[str, np.ndarray]] = {}
    for mid, grp in E.groupby("hashed_mother_id", sort=False):
        g = grp.sort_values("ep_start")
        e_by_mid[mid] = {
            "episode_idx": g["episode_idx"].to_numpy(copy=False),                     # 1..K
            "ep_start": g["ep_start"].values.astype("datetime64[ns]"),
            "ep_end": g["ep_end"].values.astype("datetime64[ns]"),
            "birth": g["birth_datetime"].values.astype("datetime64[ns]"),
        }

    # --- Per-mother assignment by pure containment ---
    def _assign_for_mother(mid: str, grp: pd.DataFrame) -> pd.DataFrame:
        if mid not in e_by_mid:
            tmp = grp.copy()
            tmp["episode_idx"] = pd.NA
            tmp["birth_datetime"] = pd.NaT
            return tmp

        arrs = e_by_mid[mid]
        idxs  = arrs["episode_idx"]
        start = arrs["ep_start"]
        end   = arrs["ep_end"]
        bdt   = arrs["birth"]

        g = grp.copy()
        tvals = g["event_time_abs"].values.astype("datetime64[ns]")

        # Interval membership using searchsorted on non-overlapping sorted intervals:
        # cand = index of rightmost start ≤ t
        pos = np.searchsorted(start, tvals, side="right") - 1
        valid = (pos >= 0) & (tvals <= end[np.clip(pos, 0, len(end)-1)])
        chosen = np.where(valid, pos, -1)

        ep_idx_out = np.full(len(g), pd.NA, dtype="object")
        birth_out  = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")

        ok_mask = chosen != -1
        if ok_mask.any():
            ci = chosen[ok_mask]
            ep_idx_out[ok_mask] = idxs[ci].astype(object)
            birth_out[ok_mask]  = bdt[ci]

        g["episode_idx"]   = ep_idx_out
        g["birth_datetime"] = birth_out
        return g

    groups = list(L.groupby("hashed_mother_id", sort=False))
    if not groups:
        out = L.copy()
        out["episode_idx"] = pd.NA
        out["birth_datetime"] = pd.NaT
        save_csv(out, CACHE_LABELS_ALIGNED)
        return out

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        batch_size=batch_size,
        prefer="threads" if backend == "threading" else None,
    )(delayed(_assign_for_mother)(mid, grp) for mid, grp in groups)

    labels_aligned = pd.concat(results, ignore_index=True)

    # Reorder/limit columns for output
    keep_cols = [
        "hashed_mother_id","pregnancy_index","episode_idx","event_time_abs","birth_datetime",
        "label","label_pos","blood_dose","blood_product"
    ]
    for c in keep_cols:
        if c not in labels_aligned.columns:
            labels_aligned[c] = np.nan if c not in ("hashed_mother_id","pregnancy_index","label","label_pos") else labels_aligned.get(c, np.nan)

    labels_aligned = labels_aligned[keep_cols]

    # Cache + return
    save_csv(labels_aligned, CACHE_LABELS_ALIGNED)
    return labels_aligned



# =============================================================================
# Build pipeline
# =============================================================================
def build_wide_timeline(base_dir: str, force_recompute: bool = False) -> pd.DataFrame:
    # Load raw CSVs
    with _timer("load_csvs"):
        births = load_births(PATH_BIRTHS)
        meas   = load_measurements(PATH_MEASUREMENTS_LONG)
        labs   = load_labs(PATH_LABS)
        drugs  = load_drugs(PATH_DRUGS)
        labels = load_labels(PATH_LABELS)

    # Vitals → wide + anchors
    with _timer("vitals_wide"):
        vitals_wide, anchors = get_or_build_vitals_wide(meas, force=force_recompute)

    # Births (ALL + pregnancy_index)
    with _timer("births_all"):
        births_all = compute_births_all(births, anchors, force=force_recompute)

    # Labs → wide
    with _timer("labs_wide"):
        labs_w = get_or_build_labs_wide(labs, anchors, force=force_recompute)

    # Drugs → wide
    with _timer("drugs_wide"):
        drugs_w = get_or_build_drugs_wide(drugs, anchors, force=force_recompute)

    # Merge wide parts
    with _timer("outer_merge"):
        merged_wide = outer_merge_wide(vitals_wide, labs_w, drugs_w, force=force_recompute)

    # Episodes + per-episode birth alignment (threaded)
    with _timer("episodes_align_births"):
        merged_wide, ep_bounds = add_episodes_and_align_births_threaded(
            merged_wide, births_all, episode_gap_days=EPISODE_GAP_DAYS, n_jobs=-1,
            batch_size="auto", force=force_recompute
        )

    # Order & classify columns (for nice output)
    with _timer("order_classify"):
        merged_wide, vitals_cols, lab_cols, drug_cols = order_and_classify_columns(merged_wide)

    # Optional pruning around birth
    # if TIME_WINDOW_BEFORE_H is not None or TIME_WINDOW_AFTER_H is not None:
    #     with _timer("prune_birth_window"):
    #         merged_wide = prune_to_birth_window(merged_wide, TIME_WINDOW_BEFORE_H, TIME_WINDOW_AFTER_H)

    # Collapse near-duplicates (<1s)
    with _timer("collapse_near_dups"):
        merged_wide = collapse_near_duplicates_fast(merged_wide, vitals_cols, lab_cols, drug_cols, threshold_seconds=1.0)

    # Align labels to birth + episode (SEPARATE FILE; NOT merged into timeline)
    force_recompute = True
    with _timer("labels_align"):
        labels_aligned = build_labels_aligned(labels, births_all, ep_bounds, force=force_recompute)

    # Save final outputs
    with _timer("save_outputs"):
        os.makedirs(base_dir, exist_ok=True)
        save_pkl(merged_wide, OUT_TIMELINE_PKL)
        save_csv(labels_aligned, OUT_LABELS_ALIGNED_CSV)

    # Print label summary by pregnancy (mother + pregnancy_index)
    with _timer("labels_summary"):
        if not labels_aligned.empty:
            per_preg = (labels_aligned.groupby(["hashed_mother_id","pregnancy_index"], as_index=False)["label"]
                        .max())
            summary = per_preg["label"].value_counts().sort_index()
            print("\n[Labels summary: pregnancies per label]")
            for lab, cnt in summary.items():
                print(f"  label={lab}: {int(cnt):,}")
        else:
            print("\n[Labels summary] no labels found.")

    return merged_wide

# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    with _timer("TOTAL"):
        df = build_wide_timeline(BASE_DIR, force_recompute=FORCE_RECOMPUTE)
    print(f"\nSaved timeline PKL: {OUT_TIMELINE_PKL}")
    print(f"Saved labels CSV  : {OUT_LABELS_ALIGNED_CSV}")
    print("\n[TIMING SUMMARY]")
    for k, v in sorted(_TIMINGS.items(), key=lambda x: (x[0] != "TOTAL", x[0])):
        print(f"{k:>28s}: {v:.3f}s")

if __name__ == "__main__":
    main()
