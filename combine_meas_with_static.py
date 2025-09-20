"""
Unified builder (config-in-script):
From raw CSVs/XLSX (mothers static, fetus/newborn, realtime vitals) → engineered
snapshot features for XGBoost (no labels yet).

This version uses a single in-file `CFG` dictionary (no argparse). Each config
entry explains:
  • What it means
  • Where it is used

Outputs
-------
- `features.parquet`         : snapshot-level feature matrix (row = [mother_id, snapshot_time])
- `feature_columns.json`     : exact feature list used by training/inference
- `static_slim.csv`          : the pruned per-mother static table actually used

Notes
-----
- No labels are produced here; we will join them later by [mother_id, snapshot_time].
- Vitals baselines = mean of first 30 min postpartum (configurable) with fallback
  to the first available postpartum value if nothing in the window.
- We retain only clinically useful columns from the huge mothers table; edit
  `CFG['mothers_keep_columns']` to add/remove items.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =====================================================================================
# CONFIG (Edit here)
# -------------------------------------------------------------------------------------
# Every key below states:
#   • What it means
#   • Where it is used in code
# =====================================================================================
CFG: Dict = {
    # --- BASE DIRECTORY --------------------------------------------------------------
    # A single place to point at your data root. Use a raw string for Windows paths.
    # You can move files without changing the rest of the config.
    # Used by: we join this with relative filenames below.
    "base_dir": r"D:\PPH",

    # --- PATHS: STATIC TABLES --------------------------------------------------------
    # Path to the very wide **mothers static** CSV.
    # Used by: parse_mothers_static() → load and slim columns
    "mothers_csv": r"D:\PPH\MF_Maternal_table_20250812.csv",

    # Path to the **fetus/newborn** CSV (one row per child).
    # Used by: parse_fetus_agg() → aggregate per mother
    "fetus_csv": r"D:\PPH\MF_FETAL_TABL_20250812_132000.csv",

    # --- PATHS: REALTIME (MANY PARTS) -----------------------------------------------
    # One or more realtime files (CSV/XLSX) containing the event stream.
    # Accepts a list of paths. XLSX sheets default to the first sheet.
    # Used by: load_realtime_multi() → read, concatenate, standardize.
    "realtime_files": [
        r"D:\PPH\pph_wide_timeline_part01.xlsx",
        r"D:\PPH\pph_wide_timeline_part02.xlsx",
        # add more parts here (xlsx or csv)
    ],

    # Optional: a cached unified Parquet for the realtime stream. If this file exists,
    # loader will read it directly (much faster than reparsing multiple XLSX files).
    # Set to None to disable caching.
    # Used by: load_realtime_multi()
    "realtime_cache_parquet": r"D:\PPH\realtime_unified.parquet",

    # --- OUTPUTS ---------------------------------------------------------------------
    # Where to write the engineered snapshot features table.
    # Used by: main (build_dataset) when saving artifacts
    "out_features_parquet": r"D:\PPH\features.parquet",

    # Where to write the JSON list of feature column names (excluding id/time).
    # Used by: main (build_dataset)
    "out_feature_cols_json": r"D:\PPH\feature_columns.json",

    # Where to write the pruned per-mother static table used by the pipeline.
    # Used by: main (build_dataset)
    "out_static_csv": r"D:\PPH\static_slim.csv",

    # --- IDENTIFIERS & CORE TIME COLUMNS --------------------------------------------
    # Canonical mother identifier name used **everywhere** inside the pipeline.
    # Used by: all functions; every table is renamed to this id column
    "id_col": "mother_id",

    # The column name of the mother id in the **mothers static** CSV (before rename).
    # Used by: parse_mothers_static() → rename to id_col
    "mothers_id_col": "hashed_mother_id",

    # The column name of the mother id in the **fetus** CSV (before rename).
    # Used by: parse_fetus_agg() → rename to id_col
    "fetus_id_col": "hashed_mother_id",

    # Timestamp column in realtime files for **measurement time**.
    # Used by: load_realtime_multi() → renamed to 'ts' (datetime)
    "rt_time_col": "event_time_abs",

    # Column in realtime files for **delivery (birth) time** of the mother.
    # Used by: load_realtime_multi() → renamed to 'delivery_time' (datetime)
    "rt_birth_col": "birth_datetime",

    # --- SIGNAL DEFINITIONS ----------------------------------------------------------
    # Mapping from source realtime column names → canonical vital names.
    # Downstream code expects **canonical names**: sbp, dbp, map, hr, temp, spo2
    # Used by: load_realtime_multi()
    "vitals_map": {
        "sistol": "sbp",            # systolic BP
        "diastol": "dbp",           # diastolic BP
        "BP - Mean": "map",         # mean arterial pressure
        "pulse": "hr",              # heart rate
        "heat": "temp",             # body temperature (°C)
        "saturation": "spo2",       # oxygen saturation (%)
    },

    # List of **sparse lab** columns in realtime stream to treat as time-series.
    # Used by: build_features_for_mother() → last/recency/baseline-delta per snapshot
    "labs_list": (
        "CREATININE_BLOOD",
        "FIBRINOGEN",
        "HCT",
        "HGB",
        "PLT",
        "SODIUM_BLOOD",
        "URIC_ACID_BLOOD",
        "WBC",
    ),

    # List of **PPH-related intervention** columns in realtime stream. Values will be
    # coerced to 0/1 and we compute ever-given + time-since-last.
    # Used by: load_realtime_multi() and build_features_for_mother()
    "drug_cols": (
        "LACTATED_RINGERS",
        "METHYLERGONOVINE",
        "MISOPROSTOL",
        "OXYTOCIN",
        "SODIUM_CHLORIDE_0_9",
    ),

    # --- FEATURE WINDOWS & SNAPSHOTS -------------------------------------------------
    # Cadence for creating **snapshot rows** after delivery.
    # Used by: build_features_for_mother() → pd.date_range(..., freq=this)
    "snapshot_every": "15min",

    # Rolling lookback window length for computing window stats (mean/std/min/max)
    # and slope features.
    # Used by: build_features_for_mother()
    "lookback_window": "120min",

    # Duration after delivery used to compute **baselines** (mean value in this
    # window). If nothing is measured in this window, fallback to first postpartum value.
    # Used by: build_features_for_mother()
    "baseline_window": "30min",

    # Max postpartum duration to engineer snapshots (prevents overly long tails).
    # Used by: build_features_for_mother()
    "max_monitor_duration": "6h",

    # --- RECENCY CAPS (to limit huge numbers) ----------------------------------------
    # Clip recency in seconds for vitals (e.g., if nothing measured yet, set to cap).
    # Used by: build_features_for_mother()
    "vitals_recency_cap_seconds": 12 * 3600,   # 12 hours

    # Clip recency in seconds for labs (they may be very sparse).
    # Used by: build_features_for_mother()
    "labs_recency_cap_seconds": 7 * 24 * 3600, # 7 days

    # Clip time-since-drug in minutes.
    # Used by: build_features_for_mother()
    "drug_recency_cap_minutes": 7 * 24 * 60,   # 7 days

    # --- MOTHERS STATIC: WHICH COLUMNS TO KEEP --------------------------------------
    # From the very wide mothers table, keep only these columns (if present).
    # This prunes non-important fields and reduces leakage risk. You can add/remove
    # items per site practice. All of these become **per-snapshot** static features.
    # Used by: parse_mothers_static()
    "mothers_keep_columns": [
        # demographics & anthropometrics
        "age_on_date", "height", "weight_before_birth", "weight_before_pregnancy",
        # obstetric history / parity
        "number_of_pregnancies_G", "number_of_births_P", "number_of_abortions_AB",
        "number_of_cesareans_CS", "number_of_vaginal_birthes_after_cesarean_VBAC",
        # hypertensive/diabetes disorders
        "chronic_hypertension", "pregnancy_induced_hypertension", "preeclampsia", "super_imposed_preeclampsia",
        "pregestational_diabetes", "gestational_diabetes_a1", "gestational_diabetes_a2", "gestational_diabetes_unspecified",
        # anesthesia types (intrapartum)
        "anesthesia_local", "anesthesia_epidural", "anesthesia_general", "anesthesia_spinal", "no_anesthesia",
        # pre-birth vitals summaries (24h window)
        "temp_24h_last_before_birth", "temp_24h_max_before_birth", "temp_24h_min_before_birth",
        "systolic_pressure_24h_last_before_birth", "systolic_pressure_24h_max_before_birth", "systolic_pressure_24h_min_before_birth",
        "diastolic_pressure_24h_last_before_birth", "diastolic_pressure_24h_max_before_birth", "diastolic_pressure_24h_min_before_birth",
        "saturation_24h_last_before_birth", "saturation_24h_max_before_birth", "saturation_24h_min_before_birth",
        # labor stage durations (we convert to minutes)
        "first_stage_hours", "second_stage_hours", "first_stage_begin", "first_stage_end",
        # labor onset characteristics
        "birth_start_spontaneous", "birth_start_cesarean", "birth_start_labor_induction", "birth_start_augmentation",
        # VBAC now
        "vbac_now",
    ],

    # --- FETUS/NEWBORN AGGREGATION DEFAULTS ------------------------------------------
    # Defaults applied when fetus table is missing for a mother (assume singleton).
    # Used by: build_dataset() after merging static pieces
    "fetus_defaults": {
        "fetus_count": 1,
        "multiple_gestation": 0,
    },
}

# =====================================================================================
# Helper utilities
# =====================================================================================

def _to_minutes(x: str) -> float:
    """Parse duration strings like 'HH:MM', 'HH:MM:SS', '27:13.2', '00:00.0' → minutes.
    Returns np.nan on failure."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 60 + m + sec / 60
        if len(parts) == 2:
            a, b = float(parts[0]), float(parts[1])
            # Heuristic: if first >= 24 and second < 60, treat as MM:SS; else HH:MM
            if a >= 24 and b < 60:
                return a + b / 60
            return a * 60 + b
        if len(parts) == 1:
            return float(parts[0])
    except Exception:
        return np.nan
    return np.nan


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _read_realtime_part(path: str, time_col: str, birth_col: str) -> pd.DataFrame:
    """Read a realtime part (xlsx or csv) and normalize time columns to datetime."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Requires openpyxl installed for .xlsx
        df = pd.read_excel(path, sheet_name=0)
    else:
        # Let pandas sniff CSV dialect; you can pass sep=... if needed
        df = pd.read_csv(path)
    # Ensure required columns exist
    missing = [c for c in ("hashed_mother_id", time_col, birth_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Realtime file {path} missing required columns: {missing}")
    # Standardize datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")
    return df


# =====================================================================================
# Mothers static: slim + engineered
# =====================================================================================

def parse_mothers_static(cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["mothers_csv"])  # load raw wide table
    if cfg["mothers_id_col"] not in df.columns:
        raise ValueError("Mothers CSV missing id column")
    df = df.rename(columns={cfg["mothers_id_col"]: cfg["id_col"]})

    # Keep only the curated set of columns if present
    keep_cols = [cfg["id_col"]] + [c for c in cfg["mothers_keep_columns"] if c in df.columns]
    df = df[keep_cols].copy()

    # Numeric coercion (except duration strings we parse separately)
    for c in df.columns:
        if c in ("first_stage_begin", "first_stage_end", cfg["id_col"]):
            continue
        if df[c].dtype == object:
            df[c] = _coerce_num(df[c])

    # Parse durations → minutes
    for dur in ("first_stage_begin", "first_stage_end"):
        if dur in df.columns:
            df[dur] = df[dur].apply(_to_minutes)

    # Labor stage minutes: prefer hours fields; fallback to begin/end if sensible
    def stage_minutes(row, hours_col, begin_col=None, end_col=None):
        if hours_col in row and pd.notna(row[hours_col]):
            return float(row[hours_col]) * 60.0
        if begin_col and end_col and (pd.notna(row.get(begin_col)) and pd.notna(row.get(end_col))):
            b, e = row.get(begin_col), row.get(end_col)
            return e - b if e >= b else e
        return np.nan

    df["labor_stage1_dur_min"] = df.apply(
        lambda r: stage_minutes(r, "first_stage_hours", "first_stage_begin", "first_stage_end"), axis=1
    )
    df["labor_stage2_dur_min"] = df.apply(
        lambda r: stage_minutes(r, "second_stage_hours"), axis=1
    )

    # BMI from weight & height (cm → m)
    if "height" in df.columns:
        h_m = _coerce_num(df["height"]) / 100.0
        if "weight_before_birth" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_birth"]) / (h_m ** 2)).replace([np.inf, -np.inf], np.nan)
        elif "weight_before_pregnancy" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_pregnancy"]) / (h_m ** 2)).replace([np.inf, -np.inf], np.nan)

    # Derived flags
    if "number_of_births_P" in df.columns:
        df["parity_primip"] = (df["number_of_births_P"].fillna(0) == 0).astype(int)
    if "number_of_cesareans_CS" in df.columns:
        df["prior_csection"] = (df["number_of_cesareans_CS"].fillna(0) > 0).astype(int)

    # Ensure 0/1 for anesthesia / onset / VBAC booleans
    for c in [
        "anesthesia_local", "anesthesia_epidural", "anesthesia_general", "anesthesia_spinal", "no_anesthesia",
        "birth_start_spontaneous", "birth_start_cesarean", "birth_start_labor_induction", "birth_start_augmentation",
        "vbac_now",
    ]:
        if c in df.columns:
            df[c] = (_coerce_num(df[c]).fillna(0) > 0).astype(int)

    # Final slim columns to emit
    slim_cols = [c for c in [
        cfg["id_col"],
        # engineered/cleaned
        "age_on_date", "bmi", "parity_primip", "number_of_pregnancies_G", "number_of_births_P",
        "prior_csection",
        # HTN/DM
        "chronic_hypertension", "pregnancy_induced_hypertension", "preeclampsia", "super_imposed_preeclampsia",
        "pregestational_diabetes", "gestational_diabetes_a1", "gestational_diabetes_a2", "gestational_diabetes_unspecified",
        # anesthesia
        "anesthesia_local", "anesthesia_epidural", "anesthesia_general", "anesthesia_spinal", "no_anesthesia",
        # pre-birth vitals summaries
        "temp_24h_last_before_birth", "temp_24h_max_before_birth", "temp_24h_min_before_birth",
        "systolic_pressure_24h_last_before_birth", "systolic_pressure_24h_max_before_birth", "systolic_pressure_24h_min_before_birth",
        "diastolic_pressure_24h_last_before_birth", "diastolic_pressure_24h_max_before_birth", "diastolic_pressure_24h_min_before_birth",
        "saturation_24h_last_before_birth", "saturation_24h_max_before_birth", "saturation_24h_min_before_birth",
        # labor durations
        "labor_stage1_dur_min", "labor_stage2_dur_min",
        # onset
        "birth_start_spontaneous", "birth_start_cesarean", "birth_start_labor_induction", "birth_start_augmentation",
        # VBAC
        "vbac_now",
    ] if c in df.columns]

    df = df[slim_cols].drop_duplicates(subset=[cfg["id_col"]]).reset_index(drop=True)
    return df

# =====================================================================================
# Fetus/newborn aggregation (per mother)
# =====================================================================================

def parse_fetus_agg(cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["fetus_csv"])  # load child-level rows
    if cfg["fetus_id_col"] not in df.columns:
        raise ValueError("Fetus CSV missing mother id column")
    df = df.rename(columns={cfg["fetus_id_col"]: cfg["id_col"]})

    # Minimal set needed for per-mother aggregation
    rename_map = {
        "weight": "birthweight_g",
        "apgar1": "apgar1",
        "apgar5": "apgar5",
        "child_gender": "child_gender",
        "nicu_hospitalization": "nicu",
    }
    for s, d in rename_map.items():
        if s in df.columns:
            df = df.rename(columns={s: d})

    # Coerce numeric / binary
    for c in ["birthweight_g", "apgar1", "apgar5"]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])
    if "nicu" in df.columns:
        df["nicu"] = (_coerce_num(df["nicu"]).fillna(0) > 0).astype(int)
    else:
        df["nicu"] = 0

    g = df.groupby(cfg["id_col"])
    agg = g.agg(
        fetus_count=("child_gender", "count"),
        birthweight_min_g=("birthweight_g", "min"),
        birthweight_max_g=("birthweight_g", "max"),
        birthweight_mean_g=("birthweight_g", "mean"),
        apgar1_min=("apgar1", "min"),
        apgar5_min=("apgar5", "min"),
        any_nicu=("nicu", "max"),
    ).reset_index()

    agg["multiple_gestation"] = (agg["fetus_count"] >= 2).astype(int)
    return agg

# =====================================================================================
# Realtime vitals → engineered features
# =====================================================================================

def load_realtime_multi(cfg: Dict) -> pd.DataFrame:
    """
    Read and unify multiple realtime files:
      - If `realtime_cache_parquet` exists, read it and return.
      - Else, read each part (xlsx/csv), concat, standardize columns, write cache (optional).
    Returns a single dataframe with:
      id_col, ts (datetime), delivery_time (datetime), vitals (canonical), labs/drugs as given.
    """
    cache_path = cfg.get("realtime_cache_parquet")
    if cache_path and os.path.exists(cache_path):
        rt = pd.read_parquet(cache_path)
        # Enforce datetime dtype on cached columns
        rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce")
        rt["delivery_time"] = pd.to_datetime(rt["delivery_time"], errors="coerce")
        return rt

    parts = []
    for path in cfg["realtime_files"]:
        df = _read_realtime_part(path, cfg["rt_time_col"], cfg["rt_birth_col"])
        parts.append(df)

    if not parts:
        raise RuntimeError("No realtime files found/loaded. Check CFG['realtime_files'].")

    rt_raw = pd.concat(parts, ignore_index=True)

    # Standardize id/time columns
    rt = rt_raw.rename(columns={
        cfg["rt_time_col"]: "ts",
        cfg["rt_birth_col"]: "delivery_time",
        "hashed_mother_id": cfg["id_col"],
    })
    rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce")
    rt["delivery_time"] = pd.to_datetime(rt["delivery_time"], errors="coerce")

    # Map vitals to canonical names and ensure missing canonical columns exist
    for src, dst in cfg["vitals_map"].items():
        if src in rt.columns:
            rt = rt.rename(columns={src: dst})
    for v in cfg["vitals_map"].values():
        if v not in rt.columns:
            rt[v] = np.nan

    # Coerce drug columns to 0/1; create if absent
    for d in cfg["drug_cols"]:
        if d in rt.columns:
            rt[d] = (_coerce_num(rt[d]).fillna(0) > 0).astype(int)
        else:
            rt[d] = 0

    # Keep **postpartum** rows for feature engineering
    rt = rt[rt["ts"] >= rt["delivery_time"]].copy()
    rt = rt.sort_values([cfg["id_col"], "ts"]).reset_index(drop=True)

    # Write cache parquet if configured
    if cache_path:
        # Ensure target dir exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        rt.to_parquet(cache_path, index=False)

    return rt


def _window_stats(series: pd.Series) -> Tuple[float, float, float, float]:
    s = series.dropna()
    if s.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(s.mean()), float(s.std(ddof=0)), float(s.min()), float(s.max()))


def build_features_for_mother(m_id: str, static_row: pd.Series, meas: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Compute snapshot features for a single mother.

    Uses:
      • CFG['snapshot_every'], ['lookback_window'], ['baseline_window'], ['max_monitor_duration']
      • Canonical vitals = values of CFG['vitals_map']
      • Labs list = CFG['labs_list']
      • Recency caps = CFG['vitals_recency_cap_seconds'], CFG['labs_recency_cap_seconds'], CFG['drug_recency_cap_minutes']
    """
    t0 = meas["delivery_time"].iloc[0]
    t_end = min(meas["ts"].max(), t0 + pd.to_timedelta(CFG["max_monitor_duration"]))
    snaps = pd.date_range(t0, t_end, freq=CFG["snapshot_every"], inclusive="left")

    # Baselines: first baseline_window after delivery
    base_end = t0 + pd.to_timedelta(CFG["baseline_window"])
    early = meas[(meas["ts"] >= t0) & (meas["ts"] <= base_end)]
    vitals = list(CFG["vitals_map"].values())
    baselines = {
        v: (early[v].dropna().mean() if not early[v].dropna().empty else meas[v].dropna().iloc[0] if not meas[v].dropna().empty else np.nan)
        for v in vitals + list(CFG["labs_list"])
        if v in meas.columns
    }

    rows = []
    lookback = pd.to_timedelta(CFG["lookback_window"])
    for snap in snaps:
        row = {CFG["id_col"]: m_id, "snapshot_time": snap}
        # static passthrough
        for col, val in static_row.items():
            row[col] = val
        upto = meas[meas["ts"] <= snap]
        in_win = upto[upto["ts"] >= snap - lookback]

        # --- VITALS FEATURES ---
        for v in vitals:
            if v not in meas.columns:
                continue
            sub = upto.dropna(subset=[v])
            if sub.empty:
                last, last_time, recency_s, measured = (np.nan, None, np.inf, 0)
            else:
                last = float(sub[v].iloc[-1])
                last_time = sub["ts"].iloc[-1]
                recency_s = (snap - last_time).total_seconds()
                measured = 1
            win_vals = in_win[v].dropna()
            mean_abs, std_abs, vmin_abs, vmax_abs = _window_stats(win_vals)
            if not win_vals.empty:
                oldest_val = float(win_vals.iloc[0])
                oldest_time = in_win.loc[win_vals.index[0], "ts"]
                dt = (snap - oldest_time).total_seconds()
                slope_abs = (last - oldest_val) / dt if (dt > 0 and not np.isnan(last)) else np.nan
            else:
                slope_abs = np.nan
            base = baselines.get(v, np.nan)
            delta_last = (last - base) if (not np.isnan(last) and not np.isnan(base)) else np.nan
            pct_last = (delta_last / base * 100.0) if (not np.isnan(delta_last) and base not in (0.0, np.nan)) else np.nan
            delta_mean = (mean_abs - base) if (not np.isnan(mean_abs) and not np.isnan(base)) else np.nan

            row.update({
                f"{v}_measured": measured,
                f"{v}_last_abs": last,
                f"{v}_mean_abs": mean_abs,
                f"{v}_std_abs": std_abs,
                f"{v}_min_abs": vmin_abs,
                f"{v}_max_abs": vmax_abs,
                f"{v}_slope_abs": slope_abs,
                f"{v}_recency_s": np.clip(recency_s, 0, CFG["vitals_recency_cap_seconds"]),
                f"{v}_baseline": base,
                f"{v}_delta_last": delta_last,
                f"{v}_pct_last": pct_last,
                f"{v}_delta_mean": delta_mean,
            })

        # --- LAB FEATURES (sparse) ---
        for lab in CFG["labs_list"]:
            if lab not in meas.columns:
                continue
            sub = upto.dropna(subset=[lab])
            if sub.empty:
                last, last_time, recency_s, measured = (np.nan, None, np.inf, 0)
            else:
                last = float(sub[lab].iloc[-1])
                last_time = sub["ts"].iloc[-1]
                recency_s = (snap - last_time).total_seconds()
                measured = 1
            base = baselines.get(lab, np.nan)
            delta_last = (last - base) if (not np.isnan(last) and not np.isnan(base)) else np.nan
            key = lab.lower()
            row.update({
                f"{key}_measured": measured,
                f"{key}_last": last,
                f"{key}_recency_s": np.clip(recency_s, 0, CFG["labs_recency_cap_seconds"]),
                f"{key}_baseline": base,
                f"{key}_delta_last": delta_last,
            })

        # --- DRUG INTERVENTION FEATURES ---
        for d in CFG["drug_cols"]:
            if d not in meas.columns:
                continue
            subd = upto[upto[d] == 1]
            ever = int(not subd.empty)
            if subd.empty:
                tsl = np.inf
            else:
                last_time = subd["ts"].iloc[-1]
                tsl = (snap - last_time).total_seconds() / 60.0  # minutes
            row[f"{d.lower()}_given"] = ever
            row[f"time_since_{d.lower()}_min"] = (
                np.clip(tsl, 0, CFG["drug_recency_cap_minutes"]) if np.isfinite(tsl) else np.inf
            )

        rows.append(row)

    return pd.DataFrame(rows)

# =====================================================================================
# Main builder
# =====================================================================================

def build_dataset(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1) Static pieces
    mothers = parse_mothers_static(cfg)
    fetus = parse_fetus_agg(cfg)
    static_merged = mothers.merge(fetus, on=cfg["id_col"], how="left")
    static_merged = static_merged.fillna(cfg["fetus_defaults"])  # e.g., singleton defaults

    # 2) Realtime rows (multi-part)
    rt = load_realtime_multi(cfg)

    # 3) Per-mother feature construction
    static_idx = static_merged.set_index(cfg["id_col"])
    parts: List[pd.DataFrame] = []
    for m_id, grp in rt.groupby(cfg["id_col"]):
        srow = static_idx.loc[m_id] if m_id in static_idx.index else pd.Series(dtype=object)
        feats = build_features_for_mother(m_id, srow, grp, cfg)
        if not feats.empty:
            parts.append(feats)
    if not parts:
        raise RuntimeError(
            "No features were built. Check that realtime files have postpartum rows and IDs match static tables."
        )

    X = pd.concat(parts, ignore_index=True)

    # 4) Save artifacts
    os.makedirs(os.path.dirname(cfg["out_static_csv"]), exist_ok=True)
    static_merged.reset_index().to_csv(cfg["out_static_csv"], index=False)

    non_feat = {cfg["id_col"], "snapshot_time"}
    feat_cols = [c for c in X.columns if c not in non_feat]

    os.makedirs(os.path.dirname(cfg["out_features_parquet"]), exist_ok=True)
    X.to_parquet(cfg["out_features_parquet"], index=False)
    with open(cfg["out_feature_cols_json"], "w") as f:
        json.dump(feat_cols, f, indent=2)

    return X, static_merged

# =====================================================================================
# Run
# =====================================================================================
if __name__ == "__main__":
    X, static_df = build_dataset(CFG)
    print(f"Built {len(X):,} snapshot rows for {X['mother_id'].nunique():,} mothers.")
    print("Saved:")
    print(" -", CFG["out_features_parquet"])
    print(" -", CFG["out_feature_cols_json"])
    print(" -", CFG["out_static_csv"])
