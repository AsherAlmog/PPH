# build_features_personalized_v2.py
# High-performance PPH feature builder with integrated labeling
# - Numba-accelerated per-window stats (fine & coarse)
# - Thread/process parallelism (kernels release the GIL)
# - Labeling: "pre-dose last frame" logic:
#     * For positive births (with ≥1 dose time):
#         - Keep negatives only before the first dose
#         - For each dose, keep exactly one positive frame: the last snapshot t strictly before the dose
#           satisfying 0 < (dose_time - t) <= label_pos_window_hours
#         - Drop all frames between doses and after the last dose
#     * For births with no doses: keep all frames as negatives
# - Min-sample gating; delta, delta_t, LS slope
# - Windowed derived hemodynamics: MAP, SI (NO PP)
# - Personalized deltas from static baselines (24h before birth)
# - Implausibles masked (set to NaN)
# - Static merge after stripping any measurement/lab columns
# - Robust, exact datetime parsing for dose/product timestamps

from __future__ import annotations

import json, os, tempfile, re
from contextlib import contextmanager as _cm
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: pyarrow for process backend I/O
try:
    import pyarrow.dataset as ds
    _HAVE_PA = True
except Exception:
    _HAVE_PA = False

# Optional: Numba for hot kernels
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict = {
    # -------- Paths --------
    "base_dir": r"D:\PPH",
    "realtime_pickle": r"D:\PPH\pph_wide_timeline.pkl",   # contains realtime (wide) with event_time_abs

    # You may provide either/both. If labels_doses_csv exists, we use it.
    # Otherwise we fall back to aligned_labels_csv with (event_time_abs,label) and derive doses from label>0 rows.
    "labels_doses_csv": r"D:\PPH\processed_drugs_new.csv",    # wide columns like blood_dose_*, blood_product_*
    "aligned_labels_csv": r"D:\PPH\pph_labels_aligned.csv",   # optional fallback: hashed_mother_id,pregnancy_index,event_time_abs,label

    # -------- Output --------
    "out_features_rt_only_parquet":         r"D:\PPH\features_rt_only_sub.parquet",
    "out_features_rt_plus_static_parquet":  r"D:\PPH\features_all_sub.parquet",
    "out_labels_all_parquet":               r"D:\PPH\labels_all_sub.parquet",
    "out_feature_cols_rt_only_json":        r"D:\PPH\feature_columns_rt_only_sub.json",
    "out_feature_cols_rt_plus_static_json": r"D:\PPH\feature_columns_sub.json",

    # -------- Feature grid & windows --------
    "fine_step_hours": 1.0,       # snapshot grid step (keep this at 1h to match your plan)
    "fine_window_hours": 3.0,
    "coarse_window_hours": 12.0,

    # -------- Labeling (pre-dose last-frame) --------
    # A snapshot time t is positive for a dose at T_dose iff:
    #   t < T_dose and (T_dose - t) <= label_pos_window_hours
    # With 1-hour snapshots, this marks exactly the last snapshot before each dose.
    "label_pos_window_hours": 1.0,

    # -------- Configure signals --------
    "measurements": ["sistol", "diastol", "BP - Mean", "saturation", "heat", "pulse"],
    "labs": ["HGB", "HCT", "PLT", "FIBRINOGEN", "WBC", "SODIUM_BLOOD", "CREATININE_BLOOD", "URIC_ACID_BLOOD"],

    # Baseline aliases (static)
    "measurement_baseline_alias": {
        "heat": ["temp", "temperature"],
        "sistol": ["systolic", "sbp"],
        "diastol": ["diastolic", "dbp"],
        "pulse": ["hr", "heart_rate"],
        "saturation": ["sat", "spo2"],
        "BP - Mean": ["map", "bp_mean"]
    },

    # Plausibility ranges (mask implausibles to NaN)
    "plausible_ranges": {
        "sistol": (40, 250),
        "diastol": (20, 150),
        "pulse": (30, 220),
        "saturation": (50, 100),
        "heat": (34, 41),
        "_MAP": (40, 170),
        "_SI": (0.2, 3.5),
        "HGB": (3, 20),
        "HCT": (10, 60),
        "PLT": (20_000, 1_000_000),
        "FIBRINOGEN": (100, 800),
        "WBC": (1, 100),
        "SODIUM_BLOOD": (110, 170),
        "CREATININE_BLOOD": (0.2, 10),
        "URIC_ACID_BLOOD": (1, 15),
    },

    # -------- Pregnancy / episodes --------
    "episode_gap_days": 183,

    # -------- Subsample BEFORE workers (optional) --------
    "subsample_per_label": True,
    "subsample_n_per_class": 10000,
    "subsample_seed": 111,

    # -------- Parallelization --------
    "n_jobs": -1,
    "parallel_backend": "threads",  # 'threads' or 'loky' (processes)

    # Cache for process backend
    "rt_parquet_dataset_dir": r"D:\PPH\.cache_builder\rt_dataset",

    # -------- Timing --------
    "print_timing": True,
}

def _subsample_births_by_dose(rt: pd.DataFrame,
                              dose_times: pd.DataFrame,
                              cfg: Dict) -> pd.DataFrame:
    """
    Balanced subsample of births (mother, episode):
      - 'Positive' birth: has ≥1 dose_time in dose_times
      - 'Negative' birth: has no doses
    Keeps up to cfg['subsample_n_per_class'] from each class (or all if fewer).
    """
    if rt.empty:
        return rt
    n_per = int(cfg.get("subsample_n_per_class", 2500))
    seed  = int(cfg.get("subsample_seed", 111))

    # Normalize types
    R = rt[["hashed_mother_id", "episode_idx"]].drop_duplicates().copy()
    R["hashed_mother_id"] = R["hashed_mother_id"].astype(str)
    R["episode_idx"] = pd.to_numeric(R["episode_idx"], errors="coerce").astype("Int64")

    D = dose_times[["hashed_mother_id", "episode_idx"]].drop_duplicates().copy() if (dose_times is not None and not dose_times.empty) else pd.DataFrame(columns=["hashed_mother_id","episode_idx"])
    if not D.empty:
        D["hashed_mother_id"] = D["hashed_mother_id"].astype(str)
        D["episode_idx"] = pd.to_numeric(D["episode_idx"], errors="coerce").astype("Int64")

    births = R.merge(D.assign(has_dose=1), on=["hashed_mother_id","episode_idx"], how="left")
    births["has_dose"] = births["has_dose"].fillna(0).astype("int8")

    pos_births = births[births["has_dose"] == 1]
    neg_births = births[births["has_dose"] == 0]

    # Sample up to n_per from each class
    if len(pos_births) > n_per:
        pos_births = pos_births.sample(n=n_per, random_state=seed)
    if len(neg_births) > n_per:
        neg_births = neg_births.sample(n=n_per, random_state=seed)

    keep_births = pd.concat([pos_births, neg_births], ignore_index=True)

    # Filter realtime rows to the kept births
    keep = rt.merge(keep_births[["hashed_mother_id","episode_idx"]],
                    on=["hashed_mother_id","episode_idx"], how="inner")
    print(f"[SUBSAMPLE] pos_births_kept={pos_births.shape[0]:,}  neg_births_kept={neg_births.shape[0]:,}  rows_after_filter={len(keep):,}")
    return keep


# =============================================================================
# Timing helper
# =============================================================================
_TIMINGS: Dict[str, float] = {}
from time import perf_counter

@_cm
def timed(name: str):
    t0 = perf_counter()
    try: yield
    finally:
        _TIMINGS[name] = _TIMINGS.get(name, 0.0) + (perf_counter() - t0)
        if CFG.get("print_timing", True):
            print(f"[TIMER] {name}: {_TIMINGS[name]:.3f}s")


# =============================================================================
# Strict datetime parsing for second-level resolution
# =============================================================================
_SEC_FMT = "%Y-%m-%d %H:%M:%S"

def _normalize_to_seconds_str(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.strftime(_SEC_FMT)
    st = s.astype("string")
    return st.str.extract(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?$', expand=False)

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(_normalize_to_seconds_str(s), format=_SEC_FMT)


# =============================================================================
# Robust exact parsing utilities for labels_doses_csv
# =============================================================================
LABELS_FMT_WITH_FRACT  = "%Y-%m-%d %H:%M:%S.%f"
LABELS_FMT_NO_FRACT    = "%Y-%m-%d %H:%M:%S"
LABELS_RE_WITH_FRACT   = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$")
LABELS_RE_NO_FRACT     = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

def _nonempty_mask(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")

def _parse_labels_time_exact(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m1 = s.str.match(LABELS_RE_WITH_FRACT)
    m2 = s.str.match(LABELS_RE_NO_FRACT)
    if m1.any():
        out.loc[m1] = pd.to_datetime(s.loc[m1], format=LABELS_FMT_WITH_FRACT)  # no coerce
    if m2.any():
        out.loc[m2] = pd.to_datetime(s.loc[m2], format=LABELS_FMT_NO_FRACT)    # no coerce
    return out


# =============================================================================
# Globals for workers (for joblib)
# =============================================================================
_CFG: Optional[Dict] = None
_RT_MEM: Optional[pd.DataFrame] = None
_RT_DS_PATH: Optional[str] = None

# Dose times store (tidy): columns [hashed_mother_id, episode_idx, dose_time]
_DOSE_TIMES: Optional[pd.DataFrame] = None

_STATIC_FULL: Optional[pd.DataFrame] = None  # unstripped static for baselines


# =============================================================================
# Fast per-window kernels (Numba)
# =============================================================================
def _ensure_contig_int64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.astype(np.int64, copy=False))

def _ensure_contig_f64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.astype(np.float64, copy=False))

def _window_stats_for_grid_fast(times_ns: np.ndarray,
                                values: np.ndarray,
                                grid_ns: np.ndarray,
                                win_ns: int,
                                min_samples: int) -> Dict[str, np.ndarray]:
    t = _ensure_contig_int64(times_ns)
    x = _ensure_contig_f64(values)
    g = _ensure_contig_int64(grid_ns)
    if t.size == 0 or g.size == 0:
        n = g.size
        return {k: np.full(n, np.nan, np.float32) for k in (
            "last","min","max","mean","std","median","slope_linfit","delta","delta_t"
        )}
    left_idx = np.searchsorted(t, g - int(win_ns), side="right")
    right_idx = np.searchsorted(t, g,              side="right")
    if _HAVE_NUMBA:
        return _numba_window_stats_kernel(
            t, x, g, left_idx.astype(np.int64), right_idx.astype(np.int64), int(min_samples)
        )
    else:
        return _python_window_stats_kernel(t, x, g, left_idx, right_idx, int(min_samples))

def _python_window_stats_kernel(t, x, g, left_idx, right_idx, min_samples):
    n = g.shape[0]
    last  = np.full(n, np.nan, np.float32)
    vmin  = np.full(n, np.nan, np.float32)
    vmax  = np.full(n, np.nan, np.float32)
    mean  = np.full(n, np.nan, np.float32)
    std   = np.full(n, np.nan, np.float32)
    med   = np.full(n, np.nan, np.float32)
    slope = np.full(n, np.nan, np.float32)
    delt  = np.full(n, np.nan, np.float32)
    dtsec = np.full(n, np.nan, np.float32)
    for i in range(n):
        L = int(left_idx[i]); R = int(right_idx[i])
        k = R - L
        if k <= 0: continue
        xv = x[L:R]
        tv = t[L:R].astype(np.float64)
        x_last = xv[-1]; x_first = xv[0]
        last[i] = np.float32(x_last)
        vmin[i] = np.float32(np.min(xv))
        vmax[i] = np.float32(np.max(xv))
        if k >= max(2, min_samples):
            mu = float(xv.mean())
            mean[i] = np.float32(mu)
            std[i] = np.float32(xv.std(ddof=0))
            med[i] = np.float32(np.median(xv))
            delt[i] = np.float32(x_last - x_first)
            dtsec[i] = np.float32((tv[-1] - tv[0]) / 1e9)
            t0 = tv[0]
            tsec = (tv - t0) / 1e9
            tm = tsec.mean()
            den = np.sum((tsec - tm)**2)
            if den > 0.0:
                slope[i] = np.float32(np.sum((tsec - tm)*(xv - mu)) / den)
    return {"last": last, "min": vmin, "max": vmax, "mean": mean, "std": std,
            "median": med, "slope_linfit": slope, "delta": delt, "delta_t": dtsec}

if _HAVE_NUMBA:
    @njit(cache=True, fastmath=False, parallel=True)
    def _numba_window_stats_kernel(t, x, g, left_idx, right_idx, min_samples):
        n = g.shape[0]
        last  = np.full(n, np.nan, np.float32)
        vmin  = np.full(n, np.nan, np.float32)
        vmax  = np.full(n, np.nan, np.float32)
        mean  = np.full(n, np.nan, np.float32)
        std   = np.full(n, np.nan, np.float32)
        med   = np.full(n, np.nan, np.float32)
        slope = np.full(n, np.nan, np.float32)
        delt  = np.full(n, np.nan, np.float32)
        dtsec = np.full(n, np.nan, np.float32)
        for i in prange(n):
            L = left_idx[i]; R = right_idx[i]
            k = R - L
            if k <= 0: continue
            xv = x[L:R]
            x_last = xv[k-1]
            last[i] = np.float32(x_last)
            mn = xv[0]; mx = xv[0]
            for j in range(1, k):
                v = xv[j]
                if v < mn: mn = v
                if v > mx: mx = v
            vmin[i] = np.float32(mn); vmax[i] = np.float32(mx)
            if k >= 2 and k >= min_samples:
                s = 0.0
                for j in range(k): s += xv[j]
                mu = s / k
                mean[i] = np.float32(mu)
                ss = 0.0
                for j in range(k):
                    d = xv[j] - mu; ss += d*d
                std[i] = np.float32((ss / k) ** 0.5)
                tmp = np.empty(k, np.float64)
                for j in range(k): tmp[j] = xv[j]
                tmp.sort()
                med[i] = np.float32(tmp[k//2]) if (k % 2)==1 else np.float32(0.5*(tmp[k//2-1]+tmp[k//2]))
                x_first = xv[0]
                delt[i] = np.float32(x_last - x_first)
                t0 = float(t[L]); t1 = float(t[R-1])
                dtsec[i] = np.float32((t1 - t0)/1e9)
                tsum = 0.0
                for j in range(k): tsum += (float(t[L+j]) - t0)
                tmean = tsum / k
                num = 0.0; den = 0.0
                for j in range(k):
                    ts = (float(t[L+j]) - t0) - tmean
                    den += ts*ts
                    num += ts*(xv[j]-mu)
                if den > 0.0:
                    slope[i] = np.float32(num/den)
        return {"last": last, "min": vmin, "max": vmax, "mean": mean, "std": std,
                "median": med, "slope_linfit": slope, "delta": delt, "delta_t": dtsec}


# =============================================================================
# Loaders & utilities
# =============================================================================
def _joblib_backend_args(cfg: Dict):
    be = cfg.get("parallel_backend", "threading").lower()
    if be in ("loky", "process", "processes"):
        return dict(backend="loky", prefer="processes")
    return dict(backend="threading", prefer="threads")

def _ensure_plain_columns(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if df is None: return df
    if isinstance(df.index, pd.MultiIndex):
        if any(k in df.index.names for k in key_cols):
            df = df.reset_index()
    elif df.index.name in key_cols:
        df = df.reset_index()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

def _load_static_table(cfg: Dict) -> pd.DataFrame:
    path = os.path.join(cfg["base_dir"], "static_merged.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    st = pd.read_parquet(path)
    if "hashed_mother_id" not in st.columns:
        raise ValueError("static_merged.parquet must contain 'hashed_mother_id'.")
    st["hashed_mother_id"] = st["hashed_mother_id"].astype(str)
    if "pregnancy_index" in st.columns:
        st["pregnancy_index"] = pd.to_numeric(st["pregnancy_index"], errors="coerce").astype("Int64")
        st = st.drop_duplicates(subset=["hashed_mother_id","pregnancy_index"], keep="last")
    else:
        st = st.drop_duplicates(subset=["hashed_mother_id"], keep="last")
    return st

def _load_realtime(cfg: Dict) -> pd.DataFrame:
    pkl = cfg["realtime_pickle"]
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Realtime pickle not found: {pkl}")
    rt = pd.read_pickle(pkl)
    assert "hashed_mother_id" in rt.columns
    assert "event_time_abs" in rt.columns
    rt["hashed_mother_id"] = rt["hashed_mother_id"].astype(str)
    rt["event_time_abs"] = _parse_datetime_series(rt["event_time_abs"])
    rt = rt.dropna(subset=["event_time_abs"]).copy()

    if "episode_idx" not in rt.columns:
        gap = pd.to_timedelta(int(cfg.get("episode_gap_days", 183)), unit="D")
        rt = rt.sort_values(["hashed_mother_id", "event_time_abs"])
        dt_ = rt.groupby("hashed_mother_id")["event_time_abs"].diff()
        new_ep = (dt_.isna()) | (dt_ > gap)
        rt["episode_idx"] = new_ep.groupby(rt["hashed_mother_id"]).cumsum().astype("int32")
    else:
        rt["episode_idx"] = pd.to_numeric(rt["episode_idx"], errors="coerce").fillna(0).astype("int32")
    return rt

def _add_pregnancy_index(rt: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    ep_bounds = (rt.groupby(["hashed_mother_id", "episode_idx"], as_index=False)["event_time_abs"].agg(ep_start="min"))
    ep_bounds = ep_bounds.sort_values(["hashed_mother_id", "ep_start"])
    ep_bounds["pregnancy_index"] = ep_bounds.groupby("hashed_mother_id").cumcount() + 1
    rt = rt.merge(ep_bounds[["hashed_mother_id", "episode_idx", "pregnancy_index"]],
                  on=["hashed_mother_id", "episode_idx"], how="left")
    rt["pregnancy_index"] = rt["pregnancy_index"].astype("int32")
    return rt


# =============================================================================
# Plausibility masking
# =============================================================================
def _mask_implausibles_inplace(df: pd.DataFrame, cfg: Dict):
    pr = cfg.get("plausible_ranges", {})
    for col, (vmin, vmax) in pr.items():
        if col.startswith("_"):
            continue
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df.loc[:, col] = s.mask((s < vmin) | (s > vmax))


# =============================================================================
# Feature engineering for a (mother, episode)
# =============================================================================
def _features_for_group(frame: pd.DataFrame,
                        cfg: Dict,
                        measurements: List[str],
                        labs: List[str],
                        static_row: Optional[pd.Series]
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    _mask_implausibles_inplace(frame, cfg)

    frame = frame.sort_values("event_time_abs")
    t_start = frame["event_time_abs"].iloc[0]
    t_end   = frame["event_time_abs"].iloc[-1]

    step = pd.to_timedelta(float(cfg["fine_step_hours"]), unit="h")
    grid = pd.date_range(t_start, t_end, freq=step, inclusive="both")
    n = len(grid)
    if n == 0:
        return (pd.DataFrame(), pd.DataFrame(), [])

    base = pd.DataFrame({
        "hashed_mother_id": frame["hashed_mother_id"].iloc[0],
        "episode_idx": frame["episode_idx"].iloc[0],
        "pregnancy_index": frame["pregnancy_index"].iloc[0],
        "snapshot_time": grid
    })

    f = frame.set_index("event_time_abs", drop=True).sort_index()
    grid_ns = grid.astype("int64").to_numpy()
    fine_ns   = int(pd.to_timedelta(float(cfg["fine_window_hours"]), unit="h").value)
    coarse_ns = int(pd.to_timedelta(float(cfg["coarse_window_hours"]), unit="h").value)
    minN = int(cfg.get("min_samples_per_window", 3))

    feat: Dict[str, np.ndarray] = {}

    def _run_stats_for_series(series: pd.Series, name_prefix: str):
        base_keys = ("last","min","max","mean","std","median","slope_linfit","delta","delta_t")
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            for scope in ("fine","coarse"):
                keys = base_keys if scope=="fine" else tuple(k for k in base_keys if k!="last")
                for k in keys:
                    feat[f"{name_prefix}_{scope}_{k}"] = np.full(n, np.nan, np.float32)
            return
        times_ns = s.index.astype("int64").to_numpy()
        vals = s.astype("float32").to_numpy()
        stats_f = _window_stats_for_grid_fast(times_ns, vals, grid_ns, fine_ns, minN)
        for k, arr in stats_f.items():
            feat[f"{name_prefix}_fine_{k}"] = arr
        stats_c = _window_stats_for_grid_fast(times_ns, vals, grid_ns, coarse_ns, minN)
        for k, arr in stats_c.items():
            if k == "last": continue
            feat[f"{name_prefix}_coarse_{k}"] = arr

    # Measurements
    for v in measurements:
        if v in f.columns:
            _run_stats_for_series(f[v], v)
        else:
            for k in ("last","min","max","mean","std","median","slope_linfit","delta","delta_t"):
                feat[f"{v}_fine_{k}"] = np.full(n, np.nan, np.float32)
            for k in ("min","max","mean","std","median","slope_linfit","delta","delta_t"):
                feat[f"{v}_coarse_{k}"] = np.full(n, np.nan, np.float32)

    # Labs
    for lab in labs:
        if lab in f.columns:
            _run_stats_for_series(f[lab], lab)
        else:
            for k in ("last","min","max","mean","std","median","slope_linfit","delta","delta_t"):
                feat[f"{lab}_fine_{k}"] = np.full(n, np.nan, np.float32)
            for k in ("min","max","mean","std","median","slope_linfit","delta","delta_t"):
                feat[f"{lab}_coarse_{k}"] = np.full(n, np.nan, np.float32)

    # Derived hemodynamics: MAP, SI (NO PP)
    pr = cfg.get("plausible_ranges", {})
    def _mask_by_key(s: pd.Series, key: str) -> pd.Series:
        if key not in pr: return s
        lo, hi = pr[key]; return s.where((s >= lo) & (s <= hi))

    has_sys = "sistol" in f.columns
    has_dia = "diastol" in f.columns
    has_hr  = "pulse"   in f.columns

    sys = pd.to_numeric(f["sistol"], errors="coerce") if has_sys else None
    dia = pd.to_numeric(f["diastol"], errors="coerce") if has_dia else None
    hr  = pd.to_numeric(f["pulse"],   errors="coerce") if has_hr  else None

    if sys is not None: sys = _mask_by_key(sys, "sistol")
    if dia is not None: dia = _mask_by_key(dia, "diastol")
    if hr  is not None: hr  = _mask_by_key(hr,  "pulse")

    # MAP
    if (sys is not None) and (dia is not None):
        MAP = dia + (sys - dia)/3.0
        MAP = _mask_by_key(MAP, "_MAP")
        _run_stats_for_series(MAP, "MAP")

    # SI
    if (sys is not None) and (hr is not None):
        SI = hr.divide(sys, fill_value=np.nan).where(sys != 0)
        SI = _mask_by_key(SI, "_SI")
        _run_stats_for_series(SI, "SI")

        # For convenience (legacy)
        p_last = hr.reindex(f.index.union(grid)).sort_index().ffill().reindex(grid).to_numpy(dtype=np.float32)
        s_last = sys.reindex(f.index.union(grid)).sort_index().ffill().reindex(grid).to_numpy(dtype=np.float32)
        shock = np.divide(p_last, s_last, out=np.full_like(p_last, np.nan), where=(s_last != 0))
        feat["shock_index_fine_last"] = shock.astype(np.float32)

    # -------- Personalized deltas from STATIC baselines --------
    def _get_measurement_baseline(name: str) -> float:
        if static_row is None or static_row.empty: return np.nan
        aliases = [name] + CFG.get("measurement_baseline_alias", {}).get(name, [])
        for a in aliases:
            col = f"{a}_24h_last_before_birth"
            if col in static_row.index:
                try:
                    return float(pd.to_numeric(pd.Series([static_row[col]])).iloc[0])
                except Exception:
                    return np.nan
        return np.nan

    def _get_lab_baseline(name: str) -> float:
        if static_row is None or static_row.empty: return np.nan
        if name in static_row.index:
            try:
                return float(pd.to_numeric(pd.Series([static_row[name]])).iloc[0])
            except Exception:
                return np.nan
        return np.nan

    def _fine_last_array(name: str, series_if_needed: Optional[pd.Series]) -> Optional[np.ndarray]:
        key = f"{name}_fine_last"
        arr = feat.get(key, None)
        if arr is not None:
            return arr.astype(np.float32)
        if series_if_needed is None:
            return None
        s = pd.to_numeric(series_if_needed, errors="coerce")
        return s.reindex(f.index.union(grid)).sort_index().ffill().reindex(grid).to_numpy(dtype=np.float32)

    def _attach_personalized(name: str, baseline_value: float, last_vals: Optional[np.ndarray]):
        if last_vals is None:
            feat[f"{name}_fine_abs_from_prebirth_last"] = np.full(n, np.nan, np.float32)
            feat[f"{name}_fine_pct_from_prebirth_last"] = np.full(n, np.nan, np.float32)
            return
        if np.isnan(baseline_value):
            abs_delta = np.full(n, np.nan, np.float32)
            pct_delta = np.full(n, np.nan, np.float32)
        else:
            abs_delta = (last_vals - baseline_value).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_delta = ((last_vals - baseline_value) / baseline_value).astype(np.float32)
                pct_delta[~np.isfinite(pct_delta)] = np.nan
        feat[f"{name}_fine_abs_from_prebirth_last"] = abs_delta
        feat[f"{name}_fine_pct_from_prebirth_last"] = pct_delta

    # Measurements personalized
    for v in measurements:
        base_v = _get_measurement_baseline(v)
        series_v = pd.to_numeric(f[v], errors="coerce") if v in f.columns else None
        last_v = _fine_last_array(v, series_v)
        _attach_personalized(v, base_v, last_v)

    # Labs personalized
    for lab in labs:
        base_l = _get_lab_baseline(lab)
        series_l = pd.to_numeric(f[lab], errors="coerce") if lab in f.columns else None
        last_l = _fine_last_array(lab, series_l)
        _attach_personalized(lab, base_l, last_l)

    # Derived personalized baselines for MAP & SI if components exist in static
    base_sys = _get_measurement_baseline("sistol")
    base_dia = _get_measurement_baseline("diastol")
    if not np.isnan(base_sys) and not np.isnan(base_dia):
        base_map = base_dia + (base_sys - base_dia)/3.0
        map_last = feat.get("MAP_fine_last", None)
        if map_last is None and (sys is not None and dia is not None):
            map_series = dia + (sys - dia)/3.0
            map_last = _fine_last_array("MAP", map_series)
        _attach_personalized("MAP", base_map, map_last)

    base_hr  = _get_measurement_baseline("pulse")
    if not np.isnan(base_sys) and not np.isnan(base_hr) and base_sys != 0:
        base_si = base_hr / base_sys
        si_last = feat.get("SI_fine_last", None)
        if si_last is None and (sys is not None and hr is not None):
            si_series = hr.divide(sys, fill_value=np.nan).where(sys != 0)
            si_last = _fine_last_array("SI", si_series)
        _attach_personalized("SI", base_si, si_last)

    X = pd.concat([base, pd.DataFrame(feat)], axis=1)
    Y_base = base[["hashed_mother_id", "pregnancy_index", "snapshot_time"]].copy()

    non_feat = {"hashed_mother_id", "episode_idx", "pregnancy_index", "snapshot_time"}
    feat_cols = [c for c in X.columns if c not in non_feat]
    return X, Y_base, feat_cols


# =============================================================================
# Dose times loaders
# =============================================================================
def _melt_label_times_from_wide(df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy dose/product timeline from a wide labels_doses_csv:
      returns columns: [hashed_mother_id, episode_idx, dose_time]
    Non-matching datetime strings are dropped (never coerced).
    """
    # Normalize episode_idx if needed
    if "episode_idx" not in df_labels.columns:
        if "pregnancy_index" in df_labels.columns:
            df_labels = df_labels.copy()
            df_labels["episode_idx"] = df_labels["pregnancy_index"]
        else:
            raise ValueError("Labels must include 'episode_idx' or 'pregnancy_index'.")

    # column name detection (case-insensitive)
    lower_cols = {c.lower(): c for c in df_labels.columns}
    dose_cols = [orig for low, orig in lower_cols.items() if low.startswith("blood_dose_")]
    prod_cols = [orig for low, orig in lower_cols.items() if low.startswith("blood_product_")]
    time_cols = dose_cols + prod_cols

    if not time_cols:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    long = df_labels[["hashed_mother_id", "episode_idx"] + time_cols] \
        .melt(id_vars=["hashed_mother_id", "episode_idx"], var_name="kind", value_name="time_str")

    long = long.loc[_nonempty_mask(long["time_str"])].copy()
    if long.empty:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    dt = _parse_labels_time_exact(long["time_str"])
    long = long.loc[dt.notna()].copy()
    if long.empty:
        return pd.DataFrame(columns=["hashed_mother_id", "episode_idx", "dose_time"])

    long["dose_time"] = dt.loc[long.index]
    long = (long.drop(columns=["time_str", "kind"])
                .drop_duplicates(subset=["hashed_mother_id", "episode_idx", "dose_time"])
                .reset_index(drop=True))
    long["hashed_mother_id"] = long["hashed_mother_id"].astype(str)
    long["episode_idx"] = pd.to_numeric(long["episode_idx"], errors="coerce").astype("Int64")
    return long

def _load_dose_times(cfg: Dict) -> pd.DataFrame:
    """
    Returns tidy dose times dataframe: [hashed_mother_id(str), episode_idx(Int64), dose_time(datetime64[ns])]
    Priority: labels_doses_csv (wide) -> aligned_labels_csv (fallback where label>0 provides event times)
    """
    # Primary: wide dose/product CSV
    p_wide = cfg.get("labels_doses_csv", None)
    if p_wide and os.path.exists(p_wide):
        with timed("read_labels_doses_csv"):
            L = pd.read_csv(p_wide, low_memory=False)
        with timed("make_tidy_dose_times"):
            doses = _melt_label_times_from_wide(L)
        if not doses.empty:
            with timed("sort_dose_times"):
                doses = doses.sort_values(["hashed_mother_id","episode_idx","dose_time"], kind="mergesort").reset_index(drop=True)
            return doses

    # Fallback: aligned labels CSV with event_time_abs,label
    p_align = cfg.get("aligned_labels_csv", None)
    if p_align and os.path.exists(p_align):
        with timed("read_aligned_labels_csv"):
            A = pd.read_csv(p_align, low_memory=False)
        need = {"hashed_mother_id", "pregnancy_index", "event_time_abs", "label"}
        if need - set(A.columns):
            return pd.DataFrame(columns=["hashed_mother_id","episode_idx","dose_time"])
        A["hashed_mother_id"] = A["hashed_mother_id"].astype(str)
        A["pregnancy_index"]  = pd.to_numeric(A["pregnancy_index"], errors="coerce").astype("Int64")
        A["label"] = pd.to_numeric(A["label"], errors="coerce").fillna(0).astype(int)
        A["event_time_abs"] = _parse_datetime_series(A["event_time_abs"])
        A = A.dropna(subset=["event_time_abs"])
        A = A[A["label"] > 0]
        if A.empty:
            return pd.DataFrame(columns=["hashed_mother_id","episode_idx","dose_time"])
        A = A.rename(columns={"pregnancy_index":"episode_idx"})
        A = A[["hashed_mother_id","episode_idx","event_time_abs"]].rename(columns={"event_time_abs":"dose_time"})
        with timed("sort_dose_times"):
            A = A.sort_values(["hashed_mother_id","episode_idx","dose_time"], kind="mergesort").reset_index(drop=True)
        return A

    # No labels source found
    return pd.DataFrame(columns=["hashed_mother_id","episode_idx","dose_time"])


# =============================================================================
# Labeling: pre-dose last-frame per birth (drop between doses)
# =============================================================================
def _label_pre_dose_last_frame(snaps_sorted: pd.DataFrame,
                               dose_times_sorted: np.ndarray,
                               pos_window_ns: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    snaps_sorted: columns include snapshot_time (sorted asc), hashed_mother_id, pregnancy_index
    dose_times_sorted: np.ndarray[datetime64[ns]] sorted ascending (unique per birth)
    Returns:
      labs_df (with 'label' and 'trigger_dose_time'), and a boolean mask (rows to keep)
    Rules:
      - If no doses: keep all snaps as negatives.
      - Else:
          * Keep negatives only for t < first_dose
          * For each dose d: find idx = rightmost snapshot strictly before d.
              - If 0 < (d - t[idx]) <= pos_window: mark that idx as positive and keep it.
          * Drop all other snapshots (between doses and after last dose).
    """
    out = snaps_sorted.copy()
    ts = out["snapshot_time"].to_numpy(copy=False)
    ts_ns = ts.view("i8")  # int64 ns

    n = len(out)
    labels = np.zeros(n, dtype=np.int8)
    trigger_ns = np.full(n, np.iinfo("i8").min, dtype="i8")
    keep = np.zeros(n, dtype=bool)

    if dose_times_sorted.size == 0:
        # No doses: keep everything as negatives
        keep[:] = True
        out["label"] = labels.astype("int8")
        out["trigger_dose_time"] = pd.Series(trigger_ns.view("datetime64[ns]"))
        return out, keep

    ds = dose_times_sorted
    ds_ns = ds.view("i8")

    # Keep negatives strictly before the first dose
    first_d = ds_ns[0]
    neg_mask = ts_ns < first_d
    keep |= neg_mask  # negatives before first dose

    # For each dose: mark the last snapshot STRICTLY before dose as positive if within window
    win = np.int64(pos_window_ns)
    pos_indices = []
    for d in ds_ns:
        idx = np.searchsorted(ts_ns, d, side="right") - 1
        if idx >= 0 and ts_ns[idx] < d:
            gap = d - ts_ns[idx]
            if 0 < gap <= win:
                # mark positive (and prefer it over negative if it happens to be kept)
                labels[idx] = 1
                trigger_ns[idx] = d
                pos_indices.append(idx)

    if pos_indices:
        pos_indices = np.unique(np.asarray(pos_indices))
        keep[pos_indices] = True  # keep exactly these positive frames

    # All snapshots after first dose and between subsequent doses are dropped
    out["label"] = labels.astype("int8")
    trig = trigger_ns.view("datetime64[ns]")
    trig = pd.Series(trig)
    trig[trigger_ns == np.iinfo("i8").min] = pd.NaT
    out["trigger_dose_time"] = trig.values

    return out, keep


# =============================================================================
# Worker I/O
# =============================================================================
def _init_worker(rt_in_memory: Optional[pd.DataFrame],
                 rt_dataset_path: Optional[str],
                 cfg: Dict,
                 dose_times_tidy: Optional[pd.DataFrame],
                 static_full: Optional[pd.DataFrame]):
    global _RT_MEM, _RT_DS_PATH, _CFG, _DOSE_TIMES, _STATIC_FULL
    _RT_MEM = rt_in_memory
    _RT_DS_PATH = rt_dataset_path
    _CFG = cfg
    _DOSE_TIMES = dose_times_tidy
    _STATIC_FULL = static_full

def _load_group_frame(key) -> pd.DataFrame:
    id_col = "hashed_mother_id"
    if _RT_MEM is not None:
        mid, ep = key
        mask = (_RT_MEM[id_col] == mid) & (_RT_MEM["episode_idx"] == ep)
        return _RT_MEM.loc[mask]
    if not _HAVE_PA or _RT_DS_PATH is None:
        raise RuntimeError("Process mode requires pyarrow; install pyarrow or switch to threading.")
    dataset = ds.dataset(_RT_DS_PATH, format="parquet")
    filt = (ds.field(id_col) == key[0]) & (ds.field("episode_idx") == key[1])
    tbl = dataset.to_table(filter=filt)
    return tbl.to_pandas()

def _get_static_row(mid: str, preg: int) -> Optional[pd.Series]:
    if _STATIC_FULL is None or _STATIC_FULL.empty: return None
    if "pregnancy_index" in _STATIC_FULL.columns:
        hit = _STATIC_FULL[( _STATIC_FULL["hashed_mother_id"] == str(mid) ) &
                           ( _STATIC_FULL["pregnancy_index"] == int(preg) )]
    else:
        hit = _STATIC_FULL[( _STATIC_FULL["hashed_mother_id"] == str(mid) )]
    if hit.empty: return None
    return hit.iloc[0]

def _dose_times_for_group(mid: str, ep: int) -> np.ndarray:
    if _DOSE_TIMES is None or _DOSE_TIMES.empty:
        return np.array([], dtype="datetime64[ns]")
    D = _DOSE_TIMES
    sub = D[(D["hashed_mother_id"] == str(mid)) & (D["episode_idx"].astype("Int64") == int(ep))]
    if sub.empty:
        return np.array([], dtype="datetime64[ns]")
    arr = sub["dose_time"].to_numpy(copy=False)
    # ensure unique & sorted
    arr = np.unique(arr)
    return arr

def _one_worker_indices(key) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str]]]:
    mid, ep = key
    grp = _load_group_frame(key)
    if grp.empty:
        return None
    grp_sorted = grp.sort_values("event_time_abs")
    preg = int(grp_sorted["pregnancy_index"].iloc[0])

    static_row = _get_static_row(str(mid), preg)
    measurements = _CFG.get("measurements", [])
    labs = _CFG.get("labs", [])

    # Build features on the uniform grid
    X_grp, Y_base, feat_cols = _features_for_group(grp, _CFG, measurements, labs, static_row)
    if X_grp.empty:
        return None

    # Prepare snapshot frame
    snaps = Y_base.copy()
    snaps["episode_idx"] = int(ep)
    snaps["hashed_mother_id"] = str(mid)
    snaps = snaps.sort_values("snapshot_time").reset_index(drop=True)

    # Dose times for this group
    dose_arr = _dose_times_for_group(str(mid), int(ep))

    # Label using "pre-dose last-frame" semantics
    pos_win_ns = int(pd.to_timedelta(float(_CFG.get("label_pos_window_hours", 1.0)), unit="h").value)
    labs_df, keep_mask = _label_pre_dose_last_frame(snaps, dose_arr, pos_win_ns)

    # Filter feature rows to only the kept snapshots
    kept_times = set(labs_df.loc[keep_mask, "snapshot_time"].astype("int64").to_numpy())
    X_grp = X_grp[X_grp["snapshot_time"].astype("int64").isin(kept_times)].reset_index(drop=True)

    # Align labs_df rows to kept snapshots only
    labs_df = labs_df.loc[keep_mask].reset_index(drop=True)
    labs_df["episode_idx"] = int(ep)

    return X_grp, labs_df[["hashed_mother_id","pregnancy_index","episode_idx","snapshot_time","label","trigger_dose_time"]], feat_cols


# =============================================================================
# Main builder
# =============================================================================
def build_dataset(cfg: Dict):
    with timed("load_realtime"):
        rt = _load_realtime(cfg)

    with timed("add_pregnancy_index"):
        rt = _add_pregnancy_index(rt, cfg)

    with timed("load_static"):
        static_df_full = _load_static_table(cfg)

    with timed("load_dose_times"):
        dose_times = _load_dose_times(cfg)  # tidy: [hashed_mother_id, episode_idx, dose_time]

    with timed("pre_worker_subsample"):
        # For this labeling, subsample is typically not needed; leave disabled or adapt if desired
        if cfg.get("subsample_per_label", False):
            # simple pass-through here; you can wire custom subsampling if needed
            rt_filt = _subsample_births_by_dose(rt, dose_times, cfg)
        else:
            rt_filt = rt.copy()

    with timed("prepare_workers"):
        be = cfg.get("parallel_backend", "threading").lower()
        n_jobs = int(cfg.get("n_jobs", 1))
        keys = list(rt_filt[["hashed_mother_id", "episode_idx"]].drop_duplicates().itertuples(index=False, name=None))

        rt_ds_path = None
        init_rt_for_workers = None
        if be in ("threading", "threads", "thread") or n_jobs in (0, 1):
            init_rt_for_workers = rt_filt
        else:
            if not _HAVE_PA:
                raise RuntimeError("pyarrow required for process backend. Install or switch backend.")
            ds_dir = cfg.get("rt_parquet_dataset_dir") or os.path.join(tempfile.gettempdir(), "pph_rt_dataset")
            os.makedirs(ds_dir, exist_ok=True)
            rt_ds_path = os.path.join(ds_dir, "rt.parquet")
            rt_filt.to_parquet(rt_ds_path, index=False, engine="pyarrow")

        _init_worker(init_rt_for_workers, rt_ds_path, cfg, dose_times, static_df_full)

    feat_parts: List[pd.DataFrame] = []
    lab_parts:  List[pd.DataFrame] = []
    feat_cols_union: List[str] = []

    with timed("build_features_parallel"):
        n_jobs = int(cfg.get("n_jobs", 1))
        if n_jobs in (0, 1):
            for key in keys:
                res = _one_worker_indices(key)
                if res is None: continue
                Xg, Yg, fcols = res
                feat_parts.append(Xg); lab_parts.append(Yg)
                feat_cols_union.extend(fcols)
        else:
            from joblib import Parallel, delayed
            be_args = _joblib_backend_args(cfg)
            results = Parallel(n_jobs=n_jobs, batch_size="auto", **be_args)(
                delayed(_one_worker_indices)(key) for key in keys
            )
            for res in results:
                if res is None: continue
                Xg, Yg, fcols = res
                feat_parts.append(Xg); lab_parts.append(Yg)
                feat_cols_union.extend(fcols)

    if not feat_parts:
        raise RuntimeError("No features produced. Check inputs and filters.")

    with timed("concat_results"):
        X_rt_only = pd.concat(feat_parts, ignore_index=True)
        Y_all = pd.concat(lab_parts,  ignore_index=True)

    # -------- Prepare stripped static for merge --------
    with timed("merge_static_after"):
        static_df = static_df_full.copy()
        if not static_df.empty:
            tokens = set()
            for m in cfg.get("measurements", []):
                tokens.add(m.lower())
                for a in cfg.get("measurement_baseline_alias", {}).get(m, []):
                    tokens.add(a.lower())
            for l in cfg.get("labs", []):
                tokens.add(l.lower())

            def _contains_token(col: str) -> bool:
                name = str(col).lower()
                return any(tok in name for tok in tokens)

            drop_cols = [c for c in static_df.columns if _contains_token(c)]
            if drop_cols:
                static_df = static_df.drop(columns=drop_cols, errors="ignore")

            dup = (set(static_df.columns) & set(X_rt_only.columns)) - {"hashed_mother_id", "pregnancy_index"}
            if dup:
                static_df = static_df.rename(columns={c: f"static__{c}" for c in dup})

            if "pregnancy_index" in static_df.columns:
                static_df["pregnancy_index"] = pd.to_numeric(static_df["pregnancy_index"], errors="coerce").astype("Int64")

            if "pregnancy_index" in static_df.columns:
                X_rt_plus_static = X_rt_only.merge(
                    static_df, on=["hashed_mother_id","pregnancy_index"], how="left", validate="many_to_one"
                )
            else:
                X_rt_plus_static = X_rt_only.merge(
                    static_df.drop(columns=[c for c in ["pregnancy_index"] if c in static_df.columns], errors="ignore"),
                    on="hashed_mother_id", how="left", validate="many_to_one"
                )
        else:
            X_rt_plus_static = X_rt_only.copy()

    with timed("save_outputs"):
        key_cols = {"hashed_mother_id", "episode_idx", "pregnancy_index", "snapshot_time"}
        feat_cols_rt_only = sorted([c for c in X_rt_only.columns if c not in key_cols])
        feat_cols_rt_plus_static = sorted([c for c in X_rt_plus_static.columns if c not in key_cols])

        os.makedirs(os.path.dirname(cfg["out_feature_cols_rt_only_json"]), exist_ok=True)

        X_rt_only.to_parquet(cfg["out_features_rt_only_parquet"], index=False)
        X_rt_plus_static.to_parquet(cfg["out_features_rt_plus_static_parquet"], index=False)
        Y_all.to_parquet(cfg["out_labels_all_parquet"], index=False)

        with open(cfg["out_feature_cols_rt_only_json"], "w") as f:
            json.dump(feat_cols_rt_only, f, indent=2)
        with open(cfg["out_feature_cols_rt_plus_static_json"], "w") as f:
            json.dump(feat_cols_rt_plus_static, f, indent=2)

    return X_rt_only, X_rt_plus_static, Y_all


def main():
    X_rt_only, X_rt_plus_static, Y_all = build_dataset(CFG)
    print(
        f"[OK] RT-only   : {len(X_rt_only):,} rows | "
        f"mothers={X_rt_only['hashed_mother_id'].nunique():,} | "
        f"pregnancies={X_rt_only[['hashed_mother_id', 'pregnancy_index']].drop_duplicates().shape[0]:,} | "
        f"episodes={X_rt_only[['hashed_mother_id', 'episode_idx']].drop_duplicates().shape[0]:,}"
    )
    print(f"[OK] RT+Static : {len(X_rt_plus_static):,} rows")
    print(f"[OK] Labels    : {len(Y_all):,} rows")
    print("Saved:")
    print(" - RT-only features :", CFG['out_features_rt_only_parquet'])
    print(" - RT+Static        :", CFG['out_features_rt_plus_static_parquet'])
    print(" - Labels           :", CFG['out_labels_all_parquet'])
    print(" - RT-only cols     :", CFG['out_feature_cols_rt_only_json'])
    print(" - RT+Static cols   :", CFG['out_feature_cols_rt_plus_static_json'])
    print("\n[TIMING SUMMARY]")
    for k, v in sorted(_TIMINGS.items()):
        print(f"{k:>28s}: {v:.3f}s")


if __name__ == "__main__":
    main()
