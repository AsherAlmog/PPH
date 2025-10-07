# build_features.py
# PPH dataset builder (optimized: rolling stats, processes/threads, indices-or-key dispatch, cached, doc-aware)

from __future__ import annotations
import os, json, tempfile
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# Optional but recommended when using process mode:
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    _HAVE_PA = True
except Exception:
    _HAVE_PA = False

# =========================
# CONFIG
# =========================
CFG: Dict = {
    # ----------------- PATHS -----------------
    "base_dir": r"D:\PPH",

    "mothers_csv": r"D:\PPH\MF_Maternal_table_20250812.csv",
    "fetus_csv":   r"D:\PPH\MF_FETAL_TABL_20250812_132000.csv",

    # Preferred realtime source (wide timeline → pickle)
    "realtime_pickle": r"D:\PPH\pph_wide_timeline.pkl",

    # Optional parts (xlsx/csv/parquet/pickle)
    "realtime_files": [
        # r"D:\PPH\pph_wide_timeline_part01.xlsx",
    ],

    # Cache for standardized realtime (pre-episode)
    "realtime_cache_parquet": r"D:\PPH\realtime_unified.parquet",

    # ---- Documented PPH (doc wins) ----
    "pph_doc_csv": r"D:\PPH\MF_mother_pph_20250812.csv",
    "pph_doc_id_col": "hashed_mother_id",
    "pph_doc_hebrew_col": "PPH",          # "כן"/"לא"
    "pph_doc_time_col": "Entry_Date",

    # ----------------- OUTPUTS -----------------
    "out_features_all_parquet":  r"D:\PPH\features_all.parquet",
    "out_labels_all_parquet":    r"D:\PPH\labels_all.parquet",
    "out_features_doc_parquet":  r"D:\PPH\features_doc_subset.parquet",
    "out_labels_doc_parquet":    r"D:\PPH\labels_doc_subset.parquet",
    "out_feature_cols_json":     r"D:\PPH\feature_columns.json",

    # Caches
    "static_cache_parquet":      r"D:\PPH\.cache_builder\static_merged.parquet",
    "rt_with_episodes_cache":    r"D:\PPH\.cache_builder\rt_with_episodes.parquet",

    # When using processes: write rt once to a parquet dataset so workers load only their slice
    "rt_parquet_dataset_dir":    r"D:\PPH\.cache_builder\rt_dataset",  # will be created/overwritten

    # ----------------- ID & TIME -----------------
    "id_col": "mother_id",
    "mothers_id_col": "hashed_mother_id",
    "fetus_id_col": "hashed_mother_id",

    # realtime source column names
    "rt_time_col":  "event_time_abs",
    "rt_birth_col": "birth_datetime",

    # ----------------- SIGNAL MAPPING -----------------
    "vitals_map": {
        "sistol":     "sbp",   # systolic BP
        "diastol":    "dbp",   # diastolic BP
        "BP - Mean":  "map",   # mean arterial pressure
        "pulse":      "hr",    # heart rate
        "heat":       "temp",  # °C
        "saturation": "spo2",  # %
    },

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

    # context drugs (NOT used in labeling)
    "drug_feature_cols": tuple(),

    # fallback labeling for non-doc mothers
    "pph_trigger_drugs": (
        "OXYTOCIN",
        "MISOPROSTOL",
        "METHYLERGONOVINE",
    ),
    "include_fluids_as_label": False,
    "fluid_drugs": ("LACTATED_RINGERS", "SODIUM_CHLORIDE_0_9"),

    # ----------------- SNAPSHOTS -----------------
    "snapshot_every": "15min",
    "lookback_window": "120min",
    "baseline_window": "30min",
    "max_monitor_duration": "6h",
    "max_snapshots_per_mother": 5000,

    # recency caps (seconds / minutes)
    "vitals_recency_cap_seconds": 12 * 3600,
    "labs_recency_cap_seconds":   7 * 24 * 3600,

    # labeling (used only for non-doc fallback)
    "pph_prophylaxis_grace_min": 30,
    "pph_repeat_dose_threshold": 2,
    "pph_repeat_dose_window_min": 60,
    "pph_future_horizon_min": 60,

    # ----------------- STATIC SLIM -----------------
    "mothers_keep_columns": [
        # IDs / dates
        "child_birth_date", "department_admission", "department_discharge",
        # History
        "number_of_pregnancies_G","number_of_births_P","number_of_abortions_AB",
        "number_of_ectopic_pregnancies_EUP","number_of_cesareans_CS",
        "number_of_vaginal_birthes_after_cesarean_VBAC","number_of_live_children_LC",
        "previous_cesarean_sheba","years_from_last_cesarean",
        "previous_OVD_sheba","years_from_last_ovd",
        # Labor & delivery
        "first_stage_hours","second_stage_hours","vbac_now",
        "baloon_inserting","amniofusion","oxytocin_administrations",
        "amniotic_fluid","membranes_rupture_type",
        # Maternal background
        "age_on_date","weight_before_pregnancy","weight_before_birth","height",
        "smoking","alcohol","drugs","hospitalizations_high_risk_pregnancy",
        "other_hospitalizations_during_pregnancy",
        # Pregnancy details
        "pregnancy_start_date","pregnancy_days_overall","pregnancy_weeks",
        "pregnancy_days_after_weeks","pregnancy_spontaneous","pregnancy_ivf",
        "pregnancy_iui","pregnancy_oi","pregnancy_ivf_ed","pregnancy_unknown",
        "last_celestone_pregnancy_weeks","fetus_count","max_child_weight","min_child_weight",
        "twins_mm","twins_mb","twins_bb","twins_unspecified",
        # Comorbidities
        "diabetes_type_1","diabetes_type_2","diabetes_unspecified","pregestational_diabetes",
        "gestational_diabetes_a1","gestational_diabetes_a2","gestational_diabetes_unspecified",
        "chronic_hypertension","pregnancy_induced_hypertension","preeclampsia","super_imposed_preeclampsia",
        "coagulopathy","pprom","cholestasis","background_diagnoses_count",
        # Infections / labs / vitals (24h before birth)
        "gbs_vagina","gbs_urine",
        "temp_24h_max_before_birth","temp_24h_min_before_birth","temp_24h_last_before_birth",
        "systolic_pressure_24h_last_before_birth","systolic_pressure_24h_max_before_birth","systolic_pressure_24h_min_before_birth",
        "diastolic_pressure_24h_last_before_birth","diastolic_pressure_24h_max_before_birth","diastolic_pressure_24h_min_before_birth",
        "saturation_24h_last_before_birth","saturation_24h_max_before_birth","saturation_24h_min_before_birth",
        # Baseline labs (keep them)
        "WBC","HGB","PLT","Fibrinogen","Creatinine -Blood",
        # Interventions
        "clexane_in_regular_drugs","aspirin_in_regular_drugs","antibiotic_during_delivery",
        # Anesthesia
        "anesthesia_local","anesthesia_epidural","anesthesia_general","anesthesia_spinal","no_anesthesia",
        # Outcomes
        "death_date",
    ],

    # fetus defaults if missing
    "fetus_defaults": {"fetus_count": 1, "multiple_gestation": 0},

    # ---- Parallelization ----
    "pph_doc_only": True,            # fast dev mode: keep doc mothers only
    "n_jobs": 8,                    # >1/-1 enables parallel
    "parallel_backend": "loky",      # *** processes by default ***
    # Set to "threading" to use indices with in-RAM slicing (no pyarrow required)

    # ---- Cache behavior ----
    "force_rebuild_static": False,
    "force_rebuild_rt_episodes": False,
}

# --- helper to map to joblib args ---
def _joblib_backend_args(cfg):
    be = cfg.get("parallel_backend", "loky").lower()
    if be in ("loky", "process", "processes"):
        return dict(backend="loky", prefer="processes")
    elif be in ("threading", "threads", "thread"):
        return dict(backend="threading", prefer="threads")
    else:
        return dict(backend="loky", prefer="processes")

# =========================
# Globals for workers
# =========================
_RT_MEM: Optional[pd.DataFrame] = None        # used only in threading mode (to avoid copies)
_RT_DS_PATH: Optional[str] = None             # parquet dataset path for process mode
_STATIC_IDX: Optional[pd.DataFrame] = None
_DOC_FLAG_MAP: Optional[Dict[str, float]] = None
_PPH_DOC_IDS: Optional[set] = None
_CFG: Optional[Dict] = None

def _init_worker(rt_in_memory: Optional[pd.DataFrame],
                 rt_dataset_path: Optional[str],
                 static_idx: pd.DataFrame,
                 doc_flag_map: Dict[str, float],
                 pph_doc_ids: set,
                 cfg: Dict):
    """
    Called once per worker. For processes:
    - keep rt_in_memory=None to avoid pickling the big frame
    - use rt_dataset_path + pyarrow filters to read tiny slices
    For threads:
    - pass rt_in_memory to share the same object without copies
    """
    global _RT_MEM, _RT_DS_PATH, _STATIC_IDX, _DOC_FLAG_MAP, _PPH_DOC_IDS, _CFG
    _RT_MEM = rt_in_memory
    _RT_DS_PATH = rt_dataset_path
    _STATIC_IDX = static_idx
    _DOC_FLAG_MAP = doc_flag_map
    _PPH_DOC_IDS = pph_doc_ids
    _CFG = cfg

# =========================
# Helpers
# =========================
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_minutes_duration(x: str) -> float:
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = float(parts[0]), float(parts[1]), float(parts[2]); return h*60 + m + sec/60
        if len(parts) == 2:
            a, b = float(parts[0]), float(parts[1])
            if a >= 24 and b < 60: return a + b/60
            return a*60 + b
        if len(parts) == 1:
            return float(parts[0])
    except:
        return np.nan
    return np.nan

def _read_realtime_part(path: str, time_col: str, birth_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=0)
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported realtime part: {ext}")
    need = ["hashed_mother_id", time_col, birth_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df[time_col]  = pd.to_datetime(df[time_col], errors="coerce")
    df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")
    return df

# =========================
# Static parsing (with cache)
# =========================
def parse_mothers_static(cfg: Dict) -> pd.DataFrame:
    cache_p = cfg["static_cache_parquet"]
    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
    if (not cfg.get("force_rebuild_static", False)) and os.path.exists(cache_p):
        return pd.read_parquet(cache_p)

    df = pd.read_csv(cfg["mothers_csv"], low_memory=False)
    df = df.rename(columns={cfg["mothers_id_col"]: cfg["id_col"]})
    keep = [cfg["id_col"]] + [c for c in cfg["mothers_keep_columns"] if c in df.columns]
    df = df[keep].copy()

    for c in df.columns:
        if c == cfg["id_col"]: continue
        if df[c].dtype == object and ("date" not in c.lower()):
            df[c] = _coerce_num(df[c])

    for dur in ("first_stage_begin", "first_stage_end"):
        if dur in df.columns:
            df[dur] = df[dur].apply(_to_minutes_duration)

    def stage_minutes(row, hours_col, begin=None, end=None):
        if hours_col in row and pd.notna(row[hours_col]): return float(row[hours_col])*60
        if begin and end and pd.notna(row.get(begin)) and pd.notna(row.get(end)):
            b, e = row.get(begin), row.get(end)
            return e - b if e >= b else e
        return np.nan

    if "first_stage_hours" in df.columns:
        df["labor_stage1_dur_min"] = df.apply(
            lambda r: stage_minutes(r, "first_stage_hours", "first_stage_begin", "first_stage_end"),
            axis=1
        )
    if "second_stage_hours" in df.columns:
        df["labor_stage2_dur_min"] = df.apply(lambda r: stage_minutes(r, "second_stage_hours"), axis=1)

    if "height" in df.columns:
        h_m = _coerce_num(df["height"]) / 100.0
        if "weight_before_birth" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_birth"]) / (h_m**2)).replace([np.inf, -np.inf], np.nan).astype(np.float32)
        elif "weight_before_pregnancy" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_pregnancy"]) / (h_m**2)).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    if "number_of_births_P" in df.columns:
        df["parity_primip"] = (df["number_of_births_P"].fillna(0) == 0).astype(np.int8)
    if "number_of_cesareans_CS" in df.columns:
        df["prior_csection"] = (df["number_of_cesareans_CS"].fillna(0) > 0).astype(np.int8)

    for c in ["anesthesia_local","anesthesia_epidural","anesthesia_general","anesthesia_spinal","no_anesthesia",
              "vbac_now"]:
        if c in df.columns:
            df[c] = (_coerce_num(df[c]).fillna(0) > 0).astype(np.int8)

    df = df.drop_duplicates(subset=[cfg["id_col"]]).reset_index(drop=True)
    df["hashed_mother_id"] = df[cfg["id_col"]]

    df.to_parquet(cache_p, index=False)
    return df

def parse_fetus_agg(cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["fetus_csv"], low_memory=False).rename(columns={cfg["fetus_id_col"]: cfg["id_col"]})
    ren = {"weight":"birthweight_g","apgar1":"apgar1","apgar5":"apgar5","child_gender":"child_gender","nicu_hospitalization":"nicu"}
    for s,d in ren.items():
        if s in df.columns: df = df.rename(columns={s:d})
    for c in ["birthweight_g","apgar1","apgar5"]:
        if c in df.columns: df[c] = _coerce_num(df[c])
    df["nicu"] = (_coerce_num(df.get("nicu", 0)).fillna(0) > 0).astype(np.int8)

    g = df.groupby(cfg["id_col"])
    agg = g.agg(
        fetus_count=("child_gender","count"),
        birthweight_min_g=("birthweight_g","min"),
        birthweight_max_g=("birthweight_g","max"),
        birthweight_mean_g=("birthweight_g","mean"),
        apgar1_min=("apgar1","min"),
        apgar5_min=("apgar5","min"),
        any_nicu=("nicu","max"),
    ).reset_index()
    agg["multiple_gestation"] = (agg["fetus_count"] >= 2).astype(np.int8)
    agg["hashed_mother_id"] = agg[cfg["id_col"]]
    return agg

# =========================
# Realtime loading & standardization
# =========================
def load_realtime_multi(cfg: Dict) -> pd.DataFrame:
    cache = cfg.get("realtime_cache_parquet")
    if cache and os.path.exists(cache):
        rt = pd.read_parquet(cache)
        rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce")
        rt["delivery_time"] = pd.to_datetime(rt["delivery_time"], errors="coerce")
        return _standardize_rt(rt, cfg)

    parts: List[pd.DataFrame] = []
    pkl = cfg.get("realtime_pickle")
    if pkl and os.path.exists(pkl):
        parts.append(_read_realtime_part(pkl, cfg["rt_time_col"], cfg["rt_birth_col"]))

    if not parts:
        for path in cfg.get("realtime_files", []):
            if os.path.exists(path):
                parts.append(_read_realtime_part(path, cfg["rt_time_col"], cfg["rt_birth_col"]))

    if not parts:
        raise RuntimeError("No realtime data located. Set 'realtime_pickle' or 'realtime_files'.")

    rt_raw = pd.concat(parts, ignore_index=True)
    rt = rt_raw.rename(columns={
        cfg["rt_time_col"]: "ts",
        cfg["rt_birth_col"]: "delivery_time",
        "hashed_mother_id": cfg["id_col"]
    })
    rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce")
    rt["delivery_time"] = pd.to_datetime(rt["delivery_time"], errors="coerce")

    rt = _standardize_rt(rt, cfg)

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        rt.to_parquet(cache, index=False)
    return rt

def _standardize_rt(rt: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    for src, dst in cfg["vitals_map"].items():
        if src in rt.columns and src != dst:
            rt = rt.rename(columns={src: dst})
    for v in cfg["vitals_map"].values():
        if v not in rt.columns: rt[v] = np.nan

    for lab in cfg["labs_list"]:
        if lab not in rt.columns:
            rt[lab] = np.nan
        else:
            rt[lab] = _coerce_num(rt[lab])

    for d in cfg.get("drug_feature_cols", []):
        if d in rt.columns:
            rt[d] = (_coerce_num(rt[d]).fillna(0) > 0).astype(np.int8)

    trig_cols = list(cfg["pph_trigger_drugs"])
    if cfg.get("include_fluids_as_label", False):
        trig_cols += list(cfg["fluid_drugs"])
    for d in trig_cols:
        if d in rt.columns:
            rt[d] = (_coerce_num(rt[d]).fillna(0) > 0).astype(np.int8)

    rt = rt[rt["ts"] >= rt["delivery_time"]].copy()
    rt = rt.sort_values([cfg["id_col"], "delivery_time", "ts"]).reset_index(drop=True)
    rt["hashed_mother_id"] = rt[cfg["id_col"]]
    return rt

# =========================
# Episode split (vectorized) + cache
# =========================
def add_episode_ids(rt: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """New episode when delivery_time changes OR any gap > 183 days within same mother."""
    cache_p = cfg["rt_with_episodes_cache"]
    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
    if (not cfg.get("force_rebuild_rt_episodes", False)) and os.path.exists(cache_p):
        return pd.read_parquet(cache_p)

    rt = rt.sort_values([cfg["id_col"], "delivery_time", "ts"]).reset_index(drop=True)

    six_months = pd.to_timedelta(183, unit="D")
    grp = rt.groupby(cfg["id_col"], sort=False)

    def _ep_flags(g: pd.DataFrame) -> pd.Series:
        dt = g["delivery_time"]
        ts = g["ts"]
        change = (dt.ne(dt.shift(1))) | ((ts - ts.shift(1)) > six_months) | ((dt - dt.shift(1)) > six_months)
        change.iloc[0] = True
        return change.cumsum().astype("int32")

    rt["episode_idx"] = grp.apply(_ep_flags).reset_index(level=0, drop=True)

    rt.to_parquet(cache_p, index=False)
    return rt

# =========================
# Vectorized windowing primitives
# =========================
def _compute_baselines(f: pd.DataFrame, t0: pd.Timestamp, baseline_td: pd.Timedelta,
                       vitals: List[str], labs: Tuple[str, ...]) -> Dict[str, float]:
    early = f.loc[(f.index >= t0) & (f.index <= t0 + baseline_td)]
    def _baseline(col):
        if col not in f.columns:
            return np.nan
        s = early[col].dropna()
        if not s.empty:
            return float(s.mean())
        s2 = f[col].dropna()
        return float(s2.iloc[0]) if not s2.empty else np.nan
    baselines = {v: _baseline(v) for v in vitals}
    for lab in labs:
        baselines[lab] = _baseline(lab)
    return baselines

def build_features_for_mother_fast(m_id, srow, frame, cfg):
    """
    Optimized version:
    - vectorized time-based rolling (no Python loops)
    - single set_index for 'frame'
    - reindex/ffill to grid for last values and rolling windows
    """
    t0 = frame["delivery_time"].iloc[0]
    t_end = min(frame["ts"].max(), t0 + pd.to_timedelta(cfg["max_monitor_duration"]))
    grid = pd.date_range(t0, t_end, freq=cfg["snapshot_every"], inclusive="left")
    n = len(grid)
    if n == 0:
        return pd.DataFrame(columns=[cfg["id_col"], "hashed_mother_id", "snapshot_time"])

    base_block = pd.DataFrame({
        cfg["id_col"]:        np.repeat(m_id, n),
        "hashed_mother_id":   np.repeat(m_id, n),
        "snapshot_time":      grid,
    })

    if len(srow) > 0:
        srow_cast = {k: (np.float32(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in srow.to_dict().items()}
        static_block = pd.DataFrame({k: [v] for k, v in srow_cast.items()})
        static_block = pd.concat([static_block] * n, ignore_index=True)
    else:
        static_block = pd.DataFrame(index=range(n))

    vitals      = list(cfg["vitals_map"].values())
    lookback    = pd.to_timedelta(cfg["lookback_window"])
    lookback_s  = lookback.total_seconds()
    baseline_td = pd.to_timedelta(cfg["baseline_window"])

    f = frame.set_index("ts", drop=True).sort_index()

    baselines = _compute_baselines(f, t0, baseline_td, vitals, cfg["labs_list"])

    grid_idx = pd.Index(grid)
    grid_ns  = grid_idx.astype("int64")

    vit_dict = {}
    for v in vitals:
        if v not in f.columns:
            vit_dict.update({
                f"{v}_measured":    np.zeros(n, dtype=np.int8),
                f"{v}_last_abs":    np.full(n, np.nan, np.float32),
                f"{v}_mean_abs":    np.full(n, np.nan, np.float32),
                f"{v}_std_abs":     np.full(n, np.nan, np.float32),
                f"{v}_min_abs":     np.full(n, np.nan, np.float32),
                f"{v}_max_abs":     np.full(n, np.nan, np.float32),
                f"{v}_slope_abs":   np.full(n, np.nan, np.float32),
                f"{v}_recency_s":   np.full(n, np.inf, np.float32),
                f"{v}_baseline":    np.full(n, np.float32(baselines.get(v, np.nan)), np.float32),
                f"{v}_delta_last":  np.full(n, np.nan, np.float32),
                f"{v}_pct_last":    np.full(n, np.nan, np.float32),
                f"{v}_delta_mean":  np.full(n, np.nan, np.float32),
            })
            continue

        s = f[v].astype("float32")

        s_grid    = s.reindex(s.index.union(grid)).sort_index().ffill()
        s_on_grid = s_grid.reindex(grid)
        measured  = (~s_on_grid.isna()).to_numpy(dtype=np.int8)

        last_seen = s.dropna().index.astype("int64").to_numpy()
        if last_seen.size == 0:
            recency_s = np.full(n, np.inf, dtype=np.float32)
        else:
            pos   = np.searchsorted(last_seen, grid_ns, side="right") - 1
            valid = pos >= 0
            prev_ns = np.where(valid, last_seen[pos], -1)
            rec     = np.where(valid, (grid_ns - prev_ns) / 1e9, np.inf)
            recency_s = np.clip(rec, 0, cfg["vitals_recency_cap_seconds"]).astype(np.float32)

        roll = s_grid.rolling(lookback)
        mean_abs = roll.mean().reindex(grid).to_numpy(dtype=np.float32)
        std_abs  = roll.std(ddof=0).reindex(grid).to_numpy(dtype=np.float32)
        vmin_abs = roll.min().reindex(grid).to_numpy(dtype=np.float32)
        vmax_abs = roll.max().reindex(grid).to_numpy(dtype=np.float32)

        start_idx   = grid_idx - lookback
        start_vals  = s_grid.reindex(s_grid.index.union(start_idx)).ffill().reindex(start_idx)
        slope_abs   = (s_on_grid.to_numpy(dtype=np.float32) - start_vals.to_numpy(dtype=np.float32)) / np.float32(lookback_s)

        last_abs  = s_on_grid.to_numpy(dtype=np.float32)
        base      = np.float32(baselines.get(v, np.nan))
        delta_last = (last_abs - base).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_last = ((last_abs - base) / base * 100.0).astype(np.float32)
        delta_mean = (mean_abs - base).astype(np.float32)

        vit_dict.update({
            f"{v}_measured":  measured,
            f"{v}_last_abs":  last_abs,
            f"{v}_mean_abs":  mean_abs,
            f"{v}_std_abs":   std_abs,
            f"{v}_min_abs":   vmin_abs,
            f"{v}_max_abs":   vmax_abs,
            f"{v}_slope_abs": slope_abs.astype(np.float32),
            f"{v}_recency_s": recency_s,
            f"{v}_baseline":  np.full(n, base, np.float32),
            f"{v}_delta_last":delta_last,
            f"{v}_pct_last":  pct_last,
            f"{v}_delta_mean":delta_mean,
        })

    vitals_block = pd.DataFrame(vit_dict, index=range(n))

    labs_dict = {}
    for lab in cfg["labs_list"]:
        if lab not in f.columns:
            labs_dict.update({
                f"{lab.lower()}_measured":   np.zeros(n, np.int8),
                f"{lab.lower()}_last":       np.full(n, np.nan, np.float32),
                f"{lab.lower()}_recency_s":  np.full(n, np.inf, np.float32),
                f"{lab.lower()}_baseline":   np.full(n, np.float32(baselines.get(lab, np.nan)), np.float32),
                f"{lab.lower()}_delta_last": np.full(n, np.nan, np.float32),
            })
            continue

        s_lab     = f[lab].astype("float32")
        s_grid    = s_lab.reindex(s_lab.index.union(grid)).sort_index().ffill()
        last_grid = s_grid.reindex(grid)
        measured  = (~last_grid.isna()).to_numpy(dtype=np.int8)

        seen = s_lab.dropna().index.astype("int64").to_numpy()
        if seen.size == 0:
            recency_s = np.full(n, np.inf, dtype=np.float32)
        else:
            pos   = np.searchsorted(seen, grid_ns, side="right") - 1
            valid = pos >= 0
            prev_ns = np.where(valid, seen[pos], -1)
            rec     = np.where(valid, (grid_ns - prev_ns) / 1e9, np.inf)
            recency_s = np.clip(rec, 0, cfg["labs_recency_cap_seconds"]).astype(np.float32)

        last_vals  = last_grid.to_numpy(dtype=np.float32)
        base       = np.float32(baselines.get(lab, np.nan))
        delta_last = (last_vals - base).astype(np.float32)

        labs_dict.update({
            f"{lab.lower()}_measured":   measured,
            f"{lab.lower()}_last":       last_vals,
            f"{lab.lower()}_recency_s":  recency_s,
            f"{lab.lower()}_baseline":   np.full(n, base, np.float32),
            f"{lab.lower()}_delta_last": delta_last,
        })

    labs_block = pd.DataFrame(labs_dict, index=range(n))

    out = pd.concat(
        [base_block.reset_index(drop=True),
         static_block.reset_index(drop=True),
         vitals_block.reset_index(drop=True),
         labs_block.reset_index(drop=True)],
        axis=1, copy=False
    ).copy()

    return out

# =========================
# Labeling
# =========================
def find_pph_events_from_drugs(meas: pd.DataFrame, cfg: Dict) -> List[pd.Timestamp]:
    t0 = pd.to_datetime(meas["delivery_time"].iloc[0])
    grace = pd.to_timedelta(cfg["pph_prophylaxis_grace_min"], unit="min")
    cols = [c for c in cfg["pph_trigger_drugs"] if c in meas.columns]
    if cfg.get("include_fluids_as_label", False):
        cols += [c for c in cfg["fluid_drugs"] if c in meas.columns]
    if not cols:
        return []
    sub = meas[["ts"] + cols].copy()
    sub["any_trig"] = sub[cols].max(axis=1)
    trig_times = sub.loc[sub["any_trig"] == 1, "ts"].sort_values().tolist()
    events: List[pd.Timestamp] = []

    for ts in trig_times:
        if ts >= t0 + grace:
            events.append(ts)
            break

    n_req = int(cfg["pph_repeat_dose_threshold"])
    win = pd.to_timedelta(cfg["pph_repeat_dose_window_min"], unit="min")
    if len(trig_times) >= n_req:
        times = pd.Series(trig_times)
        for i in range(0, len(times) - n_req + 1):
            if times.iloc[i + n_req - 1] - times.iloc[i] <= win:
                events.append(times.iloc[i + n_req - 1])
                break
    return sorted({pd.Timestamp(e) for e in events})

def label_doc_binary(snaps: pd.DataFrame, doc_binary: int, cfg: Dict) -> pd.DataFrame:
    k = int(cfg["pph_future_horizon_min"])
    out = snaps.copy()
    out["pph_doc_flag"] = 1
    out["y_pph_doc_binary"] = int(doc_binary)
    if doc_binary == 1:
        out["y_pph_past"] = 1
        out[f"y_pph_future_{k}min"] = 1
        out["time_to_pph_min"] = 0.0
    else:
        out["y_pph_past"] = 0
        out[f"y_pph_future_{k}min"] = 0
        out["time_to_pph_min"] = np.nan
    return out

def label_from_events(snaps: pd.DataFrame, events: List[pd.Timestamp], cfg: Dict) -> pd.DataFrame:
    k = int(cfg["pph_future_horizon_min"])
    out = snaps.copy()
    out["pph_doc_flag"] = 0
    out["y_pph_doc_binary"] = 0
    out["y_pph_past"] = 0
    out[f"y_pph_future_{k}min"] = 0
    out["time_to_pph_min"] = np.nan
    if not events:
        return out
    ev = pd.Series(sorted(events))
    for i, t in out["snapshot_time"].items():
        out.at[i, "y_pph_past"] = int((ev <= t).any())
        in_future = ((ev > t) & (ev <= t + pd.to_timedelta(k, unit="min"))).any()
        out.at[i, f"y_pph_future_{k}min"] = int(in_future)
        next_ev = ev[ev > t]
        if not next_ev.empty:
            out.at[i, "time_to_pph_min"] = (next_ev.iloc[0] - t).total_seconds()/60.0
    return out

# =========================
# Worker (dual path: indices for threads; pyarrow slice for processes)
# =========================
def _load_group_frame(key, idx: Optional[np.ndarray]) -> pd.DataFrame:
    id_col = _CFG["id_col"]
    if _RT_MEM is not None and idx is not None:
        # Threading path: slice in RAM by indices
        grp = _RT_MEM.iloc[idx]
        return grp
    # Process path: read just the tiny slice via filters
    if not _HAVE_PA or _RT_DS_PATH is None:
        raise RuntimeError("Process mode requires pyarrow. Set backend='threading' or install pyarrow.")
    m_id, ep_idx = key
    dataset = ds.dataset(_RT_DS_PATH, format="parquet")
    filt = (ds.field(id_col) == m_id) & (ds.field("episode_idx") == ep_idx)
    tbl = dataset.to_table(filter=filt)
    return tbl.to_pandas()

def _one_worker_indices(key, idx: Optional[np.ndarray]) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    m_id, ep_idx = key
    grp = _load_group_frame(key, idx)
    if grp.empty:
        return None
    srow = _STATIC_IDX.loc[m_id] if m_id in _STATIC_IDX.index else pd.Series(dtype=object)

    feats = build_features_for_mother_fast(m_id, srow, grp, _CFG)
    if feats.empty:
        return None

    snaps = feats[[_CFG["id_col"], "hashed_mother_id", "snapshot_time"]].copy()
    feats["episode_idx"] = ep_idx

    doc_bin = _DOC_FLAG_MAP.get(m_id, np.nan)
    if pd.notna(doc_bin):
        labs = label_doc_binary(snaps, int(doc_bin), _CFG)
    else:
        events = find_pph_events_from_drugs(grp, _CFG)
        labs = label_from_events(snaps, events, _CFG)
    labs["episode_idx"] = ep_idx

    f_doc = feats if (m_id in _PPH_DOC_IDS) else None
    l_doc = labs  if (m_id in _PPH_DOC_IDS) else None
    return feats, labs, f_doc, l_doc

# =========================
# Main builder
# =========================
def build_dataset_with_doc_override(cfg: Dict):
    # ---------- Static (cached) ----------
    mothers = parse_mothers_static(cfg)
    fetus   = parse_fetus_agg(cfg)
    static_merged = mothers.merge(fetus, on=cfg["id_col"], how="left").fillna(cfg["fetus_defaults"])
    static_idx = static_merged.set_index(cfg["id_col"])

    # ---------- Realtime ----------
    rt = load_realtime_multi(cfg)

    # ---------- Episode split (cached) ----------
    print("start episode split")
    rt = add_episode_ids(rt, cfg)
    print("finished episode split")

    # ---------- Doc table ----------
    pph_doc = pd.read_csv(cfg["pph_doc_csv"], low_memory=False)
    if cfg["pph_doc_id_col"] not in pph_doc.columns:
        raise ValueError("Documented PPH table missing id column specified in CFG['pph_doc_id_col']")
    pph_doc = pph_doc.rename(columns={cfg["pph_doc_id_col"]: cfg["id_col"]})

    he_col = cfg.get("pph_doc_hebrew_col")
    if he_col is None:
        for col in pph_doc.columns:
            if str(col).strip().lower() in {"pph", "pph_flag", "pph_yes_no", "pph_hebrew"}:
                he_col = col
                break
    if he_col and he_col in pph_doc.columns:
        val = pph_doc[he_col].astype(str).str.strip()
        pph_doc["_doc_binary_from_hebrew"] = np.where(val.eq("כן"), 1,
                                              np.where(val.eq("לא"), 0, np.nan)).astype("float32")
    else:
        pph_doc["_doc_binary_from_hebrew"] = np.nan

    doc_flag_map = pph_doc.set_index(cfg["id_col"])["_doc_binary_from_hebrew"].to_dict()
    pph_doc_ids = set(pph_doc[cfg["id_col"]].dropna().astype(str))

    # ---------- Optional doc-only pre-slice ----------
    if cfg.get("pph_doc_only", False):
        rt = rt[rt[cfg["id_col"]].astype(str).isin(pph_doc_ids)].copy()

    # ---------- Prepare group keys + indices (cheap; no subframe materialization) ----------
    group_keys = [cfg["id_col"], "episode_idx"]
    # Group indices mapping: { (mother_id, episode_idx) -> ndarray of row positions }
    idx_map = {}
    for key, grp_idx in rt.groupby(group_keys, sort=False).indices.items():
        # .indices gives row labels; with RangeIndex it's positions already
        if isinstance(grp_idx, (list, tuple)):
            grp_idx = np.asarray(grp_idx, dtype=np.int64)
        else:
            grp_idx = np.fromiter(grp_idx, dtype=np.int64) if not isinstance(grp_idx, np.ndarray) else grp_idx.astype(np.int64, copy=False)
        idx_map[key] = grp_idx

    # ---------- If process backend: write one parquet dataset for worker-side slicing ----------
    rt_ds_path = None
    be = cfg.get("parallel_backend", "loky").lower()
    if be in ("loky", "process", "processes"):
        if not _HAVE_PA:
            raise RuntimeError("pyarrow is required for process mode. Install 'pyarrow' or use backend='threading'.")
        rt_ds_dir = cfg.get("rt_parquet_dataset_dir") or os.path.join(tempfile.gettempdir(), "pph_rt_dataset")
        os.makedirs(rt_ds_dir, exist_ok=True)
        rt_ds_path = os.path.join(rt_ds_dir, "rt.parquet")  # <- write a file inside the dir

        # Overwrite dataset (single directory of parquet files)
        # We write as a dataset (not partitioned) — predicate pushdown still works on columns
        # If rt is huge, you can add partitioning=['mother_id'] to speed filters further.
        rt.to_parquet(rt_ds_path, index=False, engine="pyarrow")  # pyarrow engine by default

    # ---------- Parallel execution ----------
    feat_parts_all: List[pd.DataFrame] = []
    lab_parts_all:  List[pd.DataFrame] = []
    feat_parts_doc: List[pd.DataFrame] = []
    lab_parts_doc:  List[pd.DataFrame] = []

    n_jobs = int(cfg.get("n_jobs", 1))

    # Initialize workers appropriately
    init_rt_for_workers = rt if be in ("threading", "threads", "thread") else None
    _init_worker(init_rt_for_workers, rt_ds_path, static_idx, doc_flag_map, pph_doc_ids, cfg)

    if n_jobs == 1 or n_jobs == 0:
        # Serial path uses the same worker function
        for key, grp_idx in idx_map.items():
            res = _one_worker_indices(key, grp_idx if init_rt_for_workers is not None else None)
            if res is None: continue
            f_all, l_all, f_doc, l_doc = res
            feat_parts_all.append(f_all); lab_parts_all.append(l_all)
            if f_doc is not None:
                feat_parts_doc.append(f_doc); lab_parts_doc.append(l_doc)
    else:
        from joblib import Parallel, delayed
        be_args = _joblib_backend_args(cfg)
        print(f"starting parallelism ({be}, n_jobs={n_jobs})")
        # NOTE: We pass only (key, index-array-or-None). The big rt is never sent to children in process mode.
        results = Parallel(n_jobs=n_jobs, batch_size='auto', **be_args)(
            delayed(_one_worker_indices)(key, idx_map[key] if init_rt_for_workers is not None else None)
            for key in idx_map.keys()
        )
        print("finished parallelism")

        for res in results:
            if res is None: continue
            f_all, l_all, f_doc, l_doc = res
            feat_parts_all.append(f_all); lab_parts_all.append(l_all)
            if f_doc is not None:
                feat_parts_doc.append(f_doc); lab_parts_doc.append(l_doc)

    if not feat_parts_all:
        raise RuntimeError("No features built. Check inputs and filters.")

    X_all = pd.concat(feat_parts_all, ignore_index=True)
    Y_all = pd.concat(lab_parts_all,  ignore_index=True)

    non_feat = {cfg["id_col"], "hashed_mother_id", "snapshot_time", "episode_idx"}
    feat_cols = [c for c in X_all.columns if c not in non_feat]

    os.makedirs(os.path.dirname(cfg["out_feature_cols_json"]), exist_ok=True)
    X_all.to_parquet(cfg["out_features_all_parquet"], index=False)
    Y_all.to_parquet(cfg["out_labels_all_parquet"], index=False)
    with open(cfg["out_feature_cols_json"], "w") as f:
        json.dump(feat_cols, f, indent=2)

    if feat_parts_doc:
        X_doc = pd.concat(feat_parts_doc, ignore_index=True)
        Y_doc = pd.concat(lab_parts_doc,  ignore_index=True)
    else:
        X_doc = X_all.iloc[0:0].copy()
        Y_doc = Y_all.iloc[0:0].copy()

    X_doc.to_parquet(cfg["out_features_doc_parquet"], index=False)
    Y_doc.to_parquet(cfg["out_labels_doc_parquet"], index=False)

    return X_all, Y_all, static_merged

# =========================
# Run
# =========================
if __name__ == "__main__":
    X_all, Y_all, static_df = build_dataset_with_doc_override(CFG)
    print(f"[OK] Features (ALL): {len(X_all):,} rows, "
          f"{X_all[CFG['id_col']].nunique():,} mothers, "
          f"{X_all[['episode_idx', CFG['id_col']]].drop_duplicates().shape[0]:,} episodes, "
          f"{len([c for c in X_all.columns if c not in {CFG['id_col'],'hashed_mother_id','snapshot_time','episode_idx'}])} columns")
    print(f"[OK] Labels   (ALL): {len(Y_all):,} rows")
    print("Saved:")
    print(" -", CFG["out_features_all_parquet"])
    print(" -", CFG["out_labels_all_parquet"])
    print(" -", CFG["out_features_doc_parquet"])
    print(" -", CFG["out_labels_doc_parquet"])
    print(" -", CFG["out_feature_cols_json"])
    print("Caches:")
    print(" -", CFG["static_cache_parquet"])
    print(" -", CFG["rt_with_episodes_cache"])
    if CFG.get("parallel_backend","loky") in ("loky","process","processes"):
        print(" - rt dataset dir:", CFG["rt_parquet_dataset_dir"])
