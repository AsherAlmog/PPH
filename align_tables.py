# ============================================================
# PPH wide timeline (birth-aligned) — fast & pickle-based
#   - Inputs: CSVs (paths below)
#   - Caches: Pickle (.pkl) for vitals/labs/drugs/birth map
#   - Output: Pickle (.pkl) single file with the merged timeline
#
#   Measurements: parameter_time → wide (ONLY: BP-Mean, diastol, sistol, saturation, heat, pulse)
#   Labs: ResultTime → wide (one col per TestName, numeric)
#   Drugs: ExecutionTime → FILTER to PPH-relevant, wide binary flags
#   Births: robust birth_datetime per mother
#   Merge: outer join on (hashed_mother_id, event_time_abs)
#   Dedup: per mother, merge rows < 1 second apart (vectorized)
#   Speed-ups: column-pruned CSV reads, no .apply in hot paths
# ============================================================

import os
import re
import math  # (not strictly needed now, but left in case of future tweaks)
from datetime import time
import numpy as np
import pandas as pd

# -----------------------------
# Config (your real paths)
# -----------------------------
BASE_DIR = r"D:\PPH"

PATH_MEASUREMENTS_LONG = os.path.join(BASE_DIR, "measurements.csv")
PATH_LABS              = os.path.join(BASE_DIR, "MF_mother_labs_20250812.csv")
PATH_DRUGS             = os.path.join(BASE_DIR, "MF_mother_drugs_20250812.csv")
PATH_BIRTHS            = os.path.join(BASE_DIR, "MF_FETAL_TABL_20250812_132000.csv")

# Caches (PKL)
CACHE_VITALS_WIDE      = os.path.join(BASE_DIR, "cache_vitals_wide.pkl")
CACHE_LABS_WIDE        = os.path.join(BASE_DIR, "cache_labs_wide.pkl")
CACHE_DRUGS_WIDE       = os.path.join(BASE_DIR, "cache_drugs_wide.pkl")
CACHE_BIRTHS_MAP       = os.path.join(BASE_DIR, "cache_birth_per_mother.pkl")

# Final Output (PKL)
OUT_TIMELINE_PKL       = os.path.join(BASE_DIR, "pph_wide_timeline.pkl")

# OPTIONAL: prune events to a window around birth to speed up massively (set to numbers to enable)
TIME_WINDOW_BEFORE_H = None  # e.g., 24
TIME_WINDOW_AFTER_H  = None  # e.g., 72


# -----------------------------
# Helpers: parsing/formatting
# -----------------------------
def parse_clock_like(x):
    """Parse duration/clock strings like '21:00.0', '31:43.7', '12:00:15.3' → Timedelta or None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    td = pd.to_timedelta(s, errors="coerce")
    if pd.isna(td) and s.count(":") == 1:
        try:
            h, mm = s.split(":")
            if "." in mm:
                mins = float(mm)
                whole = int(mins)
                secs = (mins - whole) * 60.0
                s2 = f"{h}:{whole:02d}:{secs:04.1f}"
            else:
                s2 = f"{s}:00"
            td = pd.to_timedelta(s2, errors="coerce")
        except Exception:
            td = pd.NaT
    return None if pd.isna(td) else td


def parse_time_of_day(x):
    """If x looks like a time-of-day (0<=hours<24), return datetime.time; else None."""
    td = parse_clock_like(x)
    if td is None:
        return None
    hours = td.total_seconds() / 3600.0
    if 0 <= hours < 24:
        secs_total = td.total_seconds()
        secs_int = int(secs_total)
        frac = secs_total - secs_int
        hh = secs_int // 3600
        mm = (secs_int % 3600) // 60
        ss = (secs_int % 60) + frac
        return time(int(hh), int(mm), int(ss), int((ss % 1) * 1_000_000))
    return None


def fmt_signed_hms_tenths(delta_seconds):
    if pd.isna(delta_seconds):
        return np.nan
    sign = "-" if delta_seconds < 0 else ""
    x = abs(delta_seconds)
    h = int(x // 3600)
    rem = x - 3600 * h
    m = int(rem // 60)
    s = rem - 60 * m
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"


def safe_col(name: str) -> str:
    """Sanitize a label into a safe column name (letters/digits/_/+ only)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "unknown"
    s = str(name).strip()
    s = re.sub(r"[^0-9A-Za-z_+]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "unknown"


# -----------------------------
# IO helpers (column-pruned CSV in, PKL cache out)
# -----------------------------
def _read_csv_usecols(path: str, wanted_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    use = [c for c in wanted_cols if c in header.columns]
    return pd.read_csv(path, usecols=use)

def load_measurements(path: str) -> pd.DataFrame:
    usecols = [
        "hashed_mother_id", "er_admission_date", "department_admission", "department_discharge",
        "parameter_time", "Parameter_Name", "ResultNumeric", "flag"
    ]
    return _read_csv_usecols(path, usecols)

def load_labs(path: str) -> pd.DataFrame:
    usecols = ["hashed_mother_id", "ResultTime", "TestName", "Result"]
    return _read_csv_usecols(path, usecols)

def load_drugs(path: str) -> pd.DataFrame:
    usecols = ["hashed_mother_id", "ExecutionTime", "DrugName"]
    return _read_csv_usecols(path, usecols)

def load_births(path: str) -> pd.DataFrame:
    usecols = ["hashed_mother_id", "child_birth_date", "birth_time"]
    return _read_csv_usecols(path, usecols)

def save_pkl(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)

def load_pkl(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


# -----------------------------
# Births → birth_datetime per mother
# -----------------------------
def compute_birth_per_mother(births: pd.DataFrame, anchors: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and os.path.exists(CACHE_BIRTHS_MAP):
        return load_pkl(CACHE_BIRTHS_MAP)

    births = births.copy()
    births["child_birth_date_dt"] = pd.to_datetime(births.get("child_birth_date"), errors="coerce")
    births["birth_time_td"] = births.get("birth_time").apply(parse_clock_like)
    births["birth_time_tod"] = births.get("birth_time").apply(parse_time_of_day)

    b = births.merge(anchors, on="hashed_mother_id", how="left")

    def _compute(row):
        date_dt = row["child_birth_date_dt"]
        tod = row["birth_time_tod"]
        td = row["birth_time_td"]
        anch = row["anchor_er"]
        if pd.notna(date_dt):
            if tod is not None:
                return date_dt.normalize() + pd.Timedelta(
                    hours=tod.hour, minutes=tod.minute, seconds=tod.second, microseconds=tod.microsecond
                )
            return date_dt
        if (td is not None) and pd.notna(anch):
            return anch + td
        return pd.NaT

    b["birth_datetime"] = b.apply(_compute, axis=1)

    birth_per_mother = (
        b.dropna(subset=["birth_datetime"])
         .sort_values(["hashed_mother_id", "birth_datetime"])
         .groupby("hashed_mother_id", as_index=False)
         .first()[["hashed_mother_id", "birth_datetime"]]
    )

    if use_cache:
        save_pkl(birth_per_mother, CACHE_BIRTHS_MAP)
    return birth_per_mother


# -----------------------------
# Measurements: parameter_time → wide (ONLY the 6 vitals)
# -----------------------------
WANTED_VITALS = ["BP - Mean", "diastol", "sistol", "saturation", "heat", "pulse"]
HE_TO_EN_VITALS = {
    "לחץ סיסטולי": "sistol",
    "לחץ דיאסטולי": "diastol",
    "לחץ דם ממוצע": "BP - Mean",
    "סטורציה": "saturation",
    "חום": "heat",
    "דופק": "pulse",
}

def get_or_build_vitals_wide(meas: pd.DataFrame, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_cache and os.path.exists(CACHE_VITALS_WIDE):
        vitals_wide = load_pkl(CACHE_VITALS_WIDE)
        anchors = (
            vitals_wide[["hashed_mother_id", "event_time_abs"]]
            .dropna()
            .sort_values(["hashed_mother_id", "event_time_abs"])
            .groupby("hashed_mother_id", as_index=False)
            .first()
            .rename(columns={"event_time_abs": "anchor_er"})
        )
        return vitals_wide, anchors

    meas = meas.copy()
    for c in ["er_admission_date", "department_admission", "department_discharge", "parameter_time"]:
        if c in meas.columns:
            meas[c] = pd.to_datetime(meas[c], errors="coerce")

    meas = meas.dropna(subset=["hashed_mother_id", "parameter_time", "ResultNumeric"])
    if "flag" in meas.columns:
        meas = meas[meas["flag"].fillna(0).eq(0)]

    meas["param_norm"] = meas["Parameter_Name"].map(HE_TO_EN_VITALS).fillna(meas["Parameter_Name"])
    meas = meas[meas["param_norm"].isin(WANTED_VITALS)]

    # Faster pivot: drop duplicates keep last, then unstack
    meas = meas.sort_values(["hashed_mother_id", "parameter_time"]).drop_duplicates(
        subset=["hashed_mother_id", "parameter_time", "param_norm"], keep="last"
    )
    vitals_wide = (
        meas.set_index(["hashed_mother_id", "parameter_time", "param_norm"])["ResultNumeric"]
            .unstack("param_norm")
            .reset_index()
            .rename(columns={"parameter_time": "event_time_abs"})
    )

    for v in WANTED_VITALS:
        if v not in vitals_wide.columns:
            vitals_wide[v] = np.nan

    vitals_wide = vitals_wide[["hashed_mother_id", "event_time_abs"] + WANTED_VITALS]

    anchors = (
        vitals_wide[["hashed_mother_id", "event_time_abs"]]
        .dropna()
        .sort_values(["hashed_mother_id", "event_time_abs"])
        .groupby("hashed_mother_id", as_index=False)
        .first()
        .rename(columns={"event_time_abs": "anchor_er"})
    )

    if use_cache:
        save_pkl(vitals_wide, CACHE_VITALS_WIDE)

    return vitals_wide, anchors


# -----------------------------
# Labs: ResultTime → wide (one col per TestName)
# -----------------------------
def get_or_build_labs_wide(labs: pd.DataFrame, anchors: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and os.path.exists(CACHE_LABS_WIDE):
        return load_pkl(CACHE_LABS_WIDE)

    labs = labs.copy()
    labs["event_time_abs"] = pd.to_datetime(labs["ResultTime"], errors="coerce")

    missing = labs["event_time_abs"].isna() & labs["ResultTime"].notna()
    if missing.any():
        tmp = labs.loc[missing, ["hashed_mother_id", "ResultTime"]].merge(anchors, on="hashed_mother_id", how="left")
        td = tmp["ResultTime"].apply(parse_clock_like)
        labs.loc[missing, "event_time_abs"] = tmp["anchor_er"] + td

    labs["TestName_safe"] = labs.get("TestName").map(lambda x: safe_col(str(x).upper()))
    labs["Result_num"] = pd.to_numeric(labs.get("Result"), errors="coerce")

    labs = labs.sort_values(["hashed_mother_id", "event_time_abs"]).drop_duplicates(
        subset=["hashed_mother_id", "event_time_abs", "TestName_safe"], keep="last"
    )
    labs_wide = (
        labs.set_index(["hashed_mother_id", "event_time_abs", "TestName_safe"])["Result_num"]
            .unstack("TestName_safe")
            .reset_index()
    )

    if use_cache:
        save_pkl(labs_wide, CACHE_LABS_WIDE)
    return labs_wide


# -----------------------------
# Drugs: ExecutionTime → FILTER PPH relevant → wide binary flags
# -----------------------------
def normalize_pph_drug_name(drug: str):
    """
    Map raw DrugName to PPH-relevant normalized columns.
    Returns None for non-PPH drugs (filtered out).
    """
    if drug is None or (isinstance(drug, float) and np.isnan(drug)):
        return None
    s = str(drug).upper()

    if "OXYTOCIN" in s:
        return "OXYTOCIN"
    if "MISOPROSTOL" in s or "CYTOTEC" in s:
        return "MISOPROSTOL"
    if "METHYLERGONOVINE" in s or "METHERGIN" in s:
        return "METHYLERGONOVINE"
    if "LACTATED RINGERS" in s or "HARTMAN" in s or "HARTMAN`S" in s:
        return "LACTATED_RINGERS"
    if "SODIUM CHLORIDE" in s and "0.9%" in s:
        return "SODIUM_CHLORIDE_0_9"
    # (Add TXA here if you have it.)
    return None


def get_or_build_drugs_wide(drugs: pd.DataFrame, anchors: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and os.path.exists(CACHE_DRUGS_WIDE):
        return load_pkl(CACHE_DRUGS_WIDE)

    d = drugs.copy()
    d["pph_norm"] = d.get("DrugName").map(normalize_pph_drug_name)
    d = d[d["pph_norm"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["hashed_mother_id", "event_time_abs"])

    d["event_time_abs"] = pd.to_datetime(d["ExecutionTime"], errors="coerce")

    missing = d["event_time_abs"].isna() & d["ExecutionTime"].notna()
    if missing.any():
        tmp = d.loc[missing, ["hashed_mother_id", "ExecutionTime"]].merge(anchors, on="hashed_mother_id", how="left")
        td = tmp["ExecutionTime"].apply(parse_clock_like)
        d.loc[missing, "event_time_abs"] = tmp["anchor_er"] + td

    d["flag"] = 1

    d = d.sort_values(["hashed_mother_id", "event_time_abs"]).drop_duplicates(
        subset=["hashed_mother_id", "event_time_abs", "pph_norm"], keep="last"
    )
    drugs_wide = (
        d.set_index(["hashed_mother_id", "event_time_abs", "pph_norm"])["flag"]
         .unstack("pph_norm")
         .fillna(0)
         .astype("int8")
         .reset_index()
    )

    if use_cache:
        save_pkl(drugs_wide, CACHE_DRUGS_WIDE)
    return drugs_wide


# -----------------------------
# Alignment, merging, ordering
# -----------------------------
def attach_birth_and_delta(df: pd.DataFrame, birth_per_mother: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(birth_per_mother, on="hashed_mother_id", how="left")
    out["event_time_abs"] = pd.to_datetime(out["event_time_abs"], errors="coerce")
    out["t_from_birth_sec"] = (out["event_time_abs"] - out["birth_datetime"]).dt.total_seconds()
    out["t_from_birth_str"] = out["t_from_birth_sec"].apply(fmt_signed_hms_tenths)
    return out


def outer_merge_wide(vitals_wide: pd.DataFrame, labs_wide: pd.DataFrame, drugs_wide: pd.DataFrame) -> pd.DataFrame:
    merged = vitals_wide.merge(labs_wide, how="outer", on=["hashed_mother_id", "event_time_abs"])
    merged = merged.merge(drugs_wide, how="outer", on=["hashed_mother_id", "event_time_abs"])
    return merged


def order_and_classify_columns(merged: pd.DataFrame):
    key_cols = ["hashed_mother_id", "event_time_abs", "birth_datetime", "t_from_birth_sec", "t_from_birth_str"]
    vitals_cols = [c for c in WANTED_VITALS if c in merged.columns]
    exclude = set(key_cols + vitals_cols)
    other_cols = [c for c in merged.columns if c not in exclude]

    lab_cols, drug_cols = [], []
    for c in other_cols:
        vals = merged[c]
        uniq = set(pd.unique(vals.dropna()))
        if len(uniq) > 0 and uniq.issubset({0, 1}):
            drug_cols.append(c)
        else:
            lab_cols.append(c)

    ordered = merged[key_cols + vitals_cols + sorted(lab_cols) + sorted(drug_cols)]
    return ordered, vitals_cols, lab_cols, drug_cols


# -----------------------------
# Optional pruning around birth (huge speed-up if enabled)
# -----------------------------
def prune_to_birth_window(df: pd.DataFrame, before_h: float | None, after_h: float | None) -> pd.DataFrame:
    if before_h is None and after_h is None:
        return df
    lower = -float(before_h) * 3600 if before_h is not None else -np.inf
    upper = float(after_h) * 3600 if after_h is not None else np.inf
    mask = df["t_from_birth_sec"].between(lower, upper)
    return df.loc[mask].copy()


# -----------------------------
# Deduplicate near-duplicates (<1s apart) — vectorized
# -----------------------------
def collapse_near_duplicates_fast(df: pd.DataFrame,
                                  vitals_cols: list,
                                  lab_cols: list,
                                  drug_cols: list,
                                  threshold_seconds: float = 1.0) -> pd.DataFrame:
    # Sort first
    df = df.sort_values(["hashed_mother_id", "event_time_abs"], kind="stable").reset_index(drop=True)

    # Cluster rows per mother when gaps >= threshold
    dt = df.groupby("hashed_mother_id")["event_time_abs"].diff().dt.total_seconds()
    new_cluster = (dt.isna()) | (dt >= threshold_seconds)
    df["__cluster__"] = new_cluster.groupby(df["hashed_mother_id"]).cumsum()

    g_keys = ["hashed_mother_id", "__cluster__"]

    # Last non-null within cluster for vitals/labs via ffill → take 'last'
    cols_fill = [c for c in vitals_cols + lab_cols if c in df.columns]
    if cols_fill:
        df[cols_fill] = df.groupby(g_keys, sort=False)[cols_fill].ffill()

    agg_dict = {
        "event_time_abs": "min",     # earliest time in cluster
        "birth_datetime": "first",   # same per mother
    }
    for c in cols_fill:
        agg_dict[c] = "last"
    for c in drug_cols:
        if c in df.columns:
            agg_dict[c] = "max"      # any 1 wins

    # IMPORTANT: keep group keys as columns (as_index=False), then drop __cluster__
    compact = (
        df.groupby(g_keys, as_index=False, sort=False)
          .agg(agg_dict)
          .drop(columns="__cluster__", errors="ignore")
    )

    # Recompute deltas
    compact["t_from_birth_sec"] = (compact["event_time_abs"] - compact["birth_datetime"]).dt.total_seconds()
    compact["t_from_birth_str"] = compact["t_from_birth_sec"].apply(fmt_signed_hms_tenths)

    # Order columns and sort
    key_cols = ["hashed_mother_id", "event_time_abs", "birth_datetime", "t_from_birth_sec", "t_from_birth_str"]
    final_cols = [c for c in key_cols + vitals_cols + lab_cols + drug_cols if c in compact.columns]
    compact = compact[final_cols].sort_values(["hashed_mother_id", "event_time_abs"], kind="stable").reset_index(drop=True)

    return compact


# -----------------------------
# Build pipeline
# -----------------------------
def build_wide_timeline(base_dir: str, use_cache: bool = True) -> pd.DataFrame:
    # Load raw CSVs (column-pruned)
    meas   = load_measurements(PATH_MEASUREMENTS_LONG)
    labs   = load_labs(PATH_LABS)
    drugs  = load_drugs(PATH_DRUGS)
    births = load_births(PATH_BIRTHS)

    # Vitals-wide + anchors
    vitals_wide, anchors = get_or_build_vitals_wide(meas, use_cache=use_cache)

    # Birth map (per mother)
    birth_per_mother = compute_birth_per_mother(births, anchors, use_cache=use_cache)

    # Labs-wide
    labs_wide = get_or_build_labs_wide(labs, anchors, use_cache=use_cache)

    # Drugs-wide (PPH-only)
    drugs_wide = get_or_build_drugs_wide(drugs, anchors, use_cache=use_cache)

    # Merge the three wide tables
    merged_wide = outer_merge_wide(vitals_wide, labs_wide, drugs_wide)

    # Attach birth & deltas
    merged_wide = attach_birth_and_delta(merged_wide, birth_per_mother)

    # Order & classify columns
    merged_wide, vitals_cols, lab_cols, drug_cols = order_and_classify_columns(merged_wide)

    # Optional pruning around birth to shrink (massive speed-up if enabled)
    merged_wide = prune_to_birth_window(merged_wide, TIME_WINDOW_BEFORE_H, TIME_WINDOW_AFTER_H)

    # Collapse near-duplicates (<1s apart) per mother (fast)
    merged_wide = collapse_near_duplicates_fast(
        merged_wide, vitals_cols, lab_cols, drug_cols, threshold_seconds=1.0
    )

    return merged_wide


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    df = build_wide_timeline(BASE_DIR, use_cache=True)
    df.to_pickle(OUT_TIMELINE_PKL)
    print(f"Saved pickle: {OUT_TIMELINE_PKL}")


if __name__ == "__main__":
    main()
