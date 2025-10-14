# build_features_personalized.py
# High-performance PPH feature builder (personalized baselines from static; configurable signals)
# - Numba-accelerated per-window stats (fine & coarse)
# - Thread/process parallelism (kernels release the GIL)
# - Labeling via interval overlap: [t-W, t] ∩ [T-t1, T+t2] ≠ ∅
# - Exclusion around positives
# - Min-sample gating; delta, delta_t, LS slope
# - Windowed derived hemodynamics: MAP, SI (NO PP)
# - Personalized deltas from static baselines (24h before birth)
# - Implausibles masked (set to NaN)
# - Static merge after stripping any measurement/lab columns

from __future__ import annotations

import json, os, tempfile
from contextlib import contextmanager
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
    "realtime_pickle": r"D:\PPH\pph_wide_timeline.pkl",

    # Aligned labels CSV (preferred): columns hashed_mother_id,pregnancy_index,episode_idx?,event_time_abs,label
    "aligned_labels_csv": r"D:\PPH\pph_labels_aligned.csv",

    # -------- Output --------
    "out_features_rt_only_parquet":         r"D:\PPH\features_rt_only.parquet",
    "out_features_rt_plus_static_parquet":  r"D:\PPH\features_all_sub.parquet",
    "out_labels_all_parquet":               r"D:\PPH\labels_all_sub.parquet",
    "out_feature_cols_rt_only_json":        r"D:\PPH\feature_columns_rt_only.json",
    "out_feature_cols_rt_plus_static_json": r"D:\PPH\feature_columns.json",

    # -------- Feature windows --------
    "fine_step_hours": 1.0,
    "fine_window_hours": 3.0,
    "coarse_window_hours": 12.0,

    # -------- Labeling with interval overlap --------
    "label_overlap_t1_hours": 0.0,
    "label_overlap_t2_hours": 0.0,

    # Exclusion zone around positive snapshots (drop negatives near positives)
    "exclude_neg_near_pos_before_h": 12.0,
    "exclude_neg_near_pos_after_h":  12.0,

    # -------- Configure which signals to compute stats for --------
    # Measurements (vitals / realtime measurements)
    # Use exact names that appear in your realtime table.
    "measurements": ["sistol", "diastol", "BP - Mean", "saturation", "heat", "pulse"],

    # Labs (realtime lab columns in the realtime table)
    "labs": ["HGB", "HCT", "PLT", "FIBRINOGEN", "WBC", "SODIUM_BLOOD", "CREATININE_BLOOD", "URIC_ACID_BLOOD"],

    # Aliases to find baseline columns in static_merged for measurements.
    # Baseline columns in static are expected to look like "<alias>_24h_last_before_birth".
    # Example: measurement "heat" baseline might be in columns "temp_24h_last_before_birth" or "temperature_24h_last_before_birth".
    "measurement_baseline_alias": {
        "heat": ["temp", "temperature"],
        "sistol": ["systolic", "sbp"],
        "diastol": ["diastolic", "dbp"],
        "pulse": ["hr", "heart_rate"],
        "saturation": ["sat", "spo2"],
        "BP - Mean": ["map", "bp_mean"]
    },

    # -------- Plausibility ranges (implausibles masked to NaN) --------
    "plausible_ranges": {
        "sistol": (40, 250),
        "diastol": (20, 150),
        "pulse": (30, 220),
        "saturation": (50, 100),
        "heat": (34, 41),

        # Derived hemodynamics:
        "_MAP": (40, 170),
        "_SI": (0.2, 3.5),

        # Labs (adjust to your units):
        "HGB": (3, 20),           # g/dL
        "HCT": (10, 60),          # %
        "PLT": (20_000, 1_000_000),
        "FIBRINOGEN": (100, 800), # mg/dL
        "WBC": (1, 100),          # K/μL
        "SODIUM_BLOOD": (110, 170),
        "CREATININE_BLOOD": (0.2, 10),
        "URIC_ACID_BLOOD": (1, 15),
    },

    # -------- Pregnancy / episodes --------
    "episode_gap_days": 183,

    # -------- Subsample BEFORE workers (optional) --------
    "subsample_per_label": True,
    "subsample_n_per_class": 2500,
    "subsample_seed": 111,

    # -------- Parallelization --------
    "n_jobs": -1,
    "parallel_backend": "threads",

    # Cache for process backend
    "rt_parquet_dataset_dir": r"D:\PPH\.cache_builder\rt_dataset",

    # -------- Timing --------
    "print_timing": True,
}


# =============================================================================
# Timing helper
# =============================================================================
_TIMINGS: Dict[str, float] = {}
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timed(name: str):
    t0 = perf_counter()
    try: yield
    finally:
        _TIMINGS[name] = _TIMINGS.get(name, 0.0) + (perf_counter() - t0)
        if CFG.get("print_timing", True):
            print(f"[TIMER] {name}: {_TIMINGS[name]:.3f}s")


# =============================================================================
# Strict datetime parsing (NO errors='coerce')
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
# Globals for workers (for joblib)
# =============================================================================
_CFG: Optional[Dict] = None
_RT_MEM: Optional[pd.DataFrame] = None
_RT_DS_PATH: Optional[str] = None

_LABELS_EVENTS: Optional[pd.DataFrame] = None
_HAS_EVENT_TIME: bool = False

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
    # Allow pregnancy-specific rows if provided
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
# Labeling via interval overlap (fast)
# =============================================================================
def _label_snapshots_overlap(snaps_times: np.ndarray,
                             win_ns: int,
                             label_times: np.ndarray,
                             label_vals: np.ndarray,
                             t1_ns: int,
                             t2_ns: int) -> np.ndarray:
    snaps_times = np.ascontiguousarray(snaps_times.astype(np.int64))
    label_times = np.ascontiguousarray(label_times.astype(np.int64))
    label_vals  = np.ascontiguousarray(label_vals.astype(np.int32))
    if snaps_times.size == 0:
        return np.zeros(0, dtype=np.int8)
    if label_times.size == 0:
        return np.zeros(snaps_times.size, dtype=np.int8)
    Lstart = label_times - t1_ns
    Lend   = label_times + t2_ns
    order_start = np.argsort(Lstart, kind='mergesort')
    Lstart_sorted = Lstart[order_start]
    labels_sorted = label_vals[order_start]
    out = np.zeros(snaps_times.size, dtype=np.int8)
    for i in range(snaps_times.size):
        we = snaps_times[i]; ws = we - win_ns
        idx_r = np.searchsorted(Lstart_sorted, we, side='right')
        if idx_r == 0: continue
        mlabel = 0; j = idx_r - 1; limit = 128
        while j >= 0 and limit > 0:
            orig = order_start[j]
            if Lend[orig] < ws: break
            if labels_sorted[j] > mlabel:
                mlabel = labels_sorted[j]
            j -= 1; limit -= 1
        out[i] = np.int8(mlabel if mlabel > 0 else 0)
    return out


# =============================================================================
# Plausibility masking
# =============================================================================
def _mask_implausibles_inplace(df: pd.DataFrame, cfg: Dict):
    pr = cfg.get("plausible_ranges", {})
    for col, (vmin, vmax) in pr.items():
        if col.startswith("_"):  # derived handled at creation
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

        # Legacy point feature
        p_last = hr.reindex(f.index.union(grid)).sort_index().ffill().reindex(grid).to_numpy(dtype=np.float32)
        s_last = sys.reindex(f.index.union(grid)).sort_index().ffill().reindex(grid).to_numpy(dtype=np.float32)
        shock = np.divide(p_last, s_last, out=np.full_like(p_last, np.nan), where=(s_last != 0))
        feat["shock_index_fine_last"] = shock.astype(np.float32)

    # -------- Personalized deltas from STATIC baselines --------
    def _get_measurement_baseline(name: str) -> float:
        """Look for <alias>_24h_last_before_birth in static_row"""
        if static_row is None or static_row.empty:
            return np.nan
        aliases = [name]
        aliases += CFG.get("measurement_baseline_alias", {}).get(name, [])
        for a in aliases:
            col = f"{a}_24h_last_before_birth"
            if col in static_row.index:
                try:
                    val = float(pd.to_numeric(pd.Series([static_row[col]])).iloc[0])
                except Exception:
                    val = np.nan
                return val
        return np.nan

    def _get_lab_baseline(name: str) -> float:
        """Labs baseline taken directly by same-named column in static."""
        if static_row is None or static_row.empty:
            return np.nan
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
    # MAP baseline if systolic & diastolic baselines exist
    base_sys = _get_measurement_baseline("sistol")
    base_dia = _get_measurement_baseline("diastol")
    if not np.isnan(base_sys) and not np.isnan(base_dia):
        base_map = base_dia + (base_sys - base_dia)/3.0
        # For last MAP array, regenerate from current MAP fine_last if present
        map_last = feat.get("MAP_fine_last", None)
        if map_last is None and (sys is not None and dia is not None):
            map_series = dia + (sys - dia)/3.0
            map_last = _fine_last_array("MAP", map_series)
        _attach_personalized("MAP", base_map, map_last)

    # SI baseline if systolic & pulse baselines exist
    base_hr  = _get_measurement_baseline("pulse")
    if not np.isnan(base_sys) and not np.isnan(base_hr) and base_sys != 0:
        base_si = base_hr / base_sys
        si_last = feat.get("SI_fine_last", None)
        if si_last is None and (sys is not None and hr is not None):
            si_series = hr.divide(sys, fill_value=np.nan).where(sys != 0)
            si_last = _fine_last_array("SI", si_series)
        _attach_personalized("SI", base_si, si_last)

    # Finalize
    X = pd.concat([base, pd.DataFrame(feat)], axis=1)
    Y_base = base[["hashed_mother_id", "pregnancy_index", "snapshot_time"]].copy()

    non_feat = {"hashed_mother_id", "episode_idx", "pregnancy_index", "snapshot_time"}
    feat_cols = [c for c in X.columns if c not in non_feat]
    return X, Y_base, feat_cols


# =============================================================================
# Read labels (aligned-only, simpler)
# =============================================================================
def _read_aligned_labels(cfg: Dict) -> Tuple[pd.DataFrame, bool]:
    p = cfg.get("aligned_labels_csv")
    if not p or not os.path.exists(p): return pd.DataFrame(), False
    L = pd.read_csv(p, low_memory=False)
    need = {"hashed_mother_id", "pregnancy_index", "event_time_abs", "label"}
    if need - set(L.columns): return pd.DataFrame(), False
    L["hashed_mother_id"] = L["hashed_mother_id"].astype(str)
    if "episode_idx" in L.columns:
        L["episode_idx"] = pd.to_numeric(L["episode_idx"], errors="coerce").astype("Int64")
    L["label"] = pd.to_numeric(L["label"], errors="coerce").fillna(0).astype(int)
    L["event_time_abs"] = _parse_datetime_series(L["event_time_abs"])
    L = L.dropna(subset=["event_time_abs"])
    return L.rename(columns={"event_time_abs": "event_time"}), True

def _apply_external_labels_overlap(snaps: pd.DataFrame,
                                   cfg: Dict,
                                   Le: Optional[pd.DataFrame],
                                   has_event_time: bool) -> pd.DataFrame:
    out = snaps.copy()
    out["hashed_mother_id"] = out["hashed_mother_id"].astype(str)
    if not np.issubdtype(out["snapshot_time"].dtype, np.datetime64):
        out["snapshot_time"] = _parse_datetime_series(out["snapshot_time"])
    if not has_event_time or Le is None or Le.empty:
        out["label"] = 0
        return out

    Le = Le.copy()
    Le["hashed_mother_id"] = Le["hashed_mother_id"].astype(str)
    if not np.issubdtype(Le["event_time"].dtype, np.datetime64):
        Le["event_time"] = _parse_datetime_series(Le["event_time"])
    Le["label"] = pd.to_numeric(Le["label"], errors="coerce").fillna(0).astype(int)
    Le = Le.dropna(subset=["event_time"])

    t1_ns = int(pd.to_timedelta(float(cfg.get("label_overlap_t1_hours", 0.0)), unit="h").value)
    t2_ns = int(pd.to_timedelta(float(cfg.get("label_overlap_t2_hours", 0.0)), unit="h").value)
    fine_ns = int(pd.to_timedelta(float(cfg["fine_window_hours"]), unit="h").value)

    labels = np.zeros(len(out), dtype=np.int8)
    out_idx = out.reset_index().rename(columns={"index": "__row__"})

    for (mid, preg, ep), sub_idx in out_idx.groupby(["hashed_mother_id", "pregnancy_index", "episode_idx"], sort=False):
        snaps_i = out.loc[sub_idx["__row__"], ["snapshot_time"]].sort_values("snapshot_time")
        if "episode_idx" in Le.columns and pd.notna(ep):
            ev = Le[(Le["hashed_mother_id"] == str(mid)) &
                    (Le["pregnancy_index"] == preg) &
                    (Le["episode_idx"].astype("Int64") == int(ep))]
        else:
            ev = Le[(Le["hashed_mother_id"] == str(mid)) & (Le["pregnancy_index"] == preg)]
            # allow episode-bounded if both sides have times (optional)
        if ev.empty: continue

        ev_times = ev["event_time"].values.astype("datetime64[ns]")
        ev_labels = ev["label"].to_numpy(dtype=int)
        tvals = snaps_i["snapshot_time"].values.astype("datetime64[ns]")

        lab_here = _label_snapshots_overlap(
            tvals.astype("int64"), fine_ns,
            ev_times.astype("int64"), ev_labels.astype(np.int32),
            t1_ns, t2_ns
        )
        labels[sub_idx["__row__"].to_numpy()] = lab_here.astype(np.int8)

    out["label"] = labels.astype(int)
    return out

def _apply_exclusion_around_positives(snaps_with_labels: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    before_h = float(cfg.get("exclude_neg_near_pos_before_h", 0.0))
    after_h  = float(cfg.get("exclude_neg_near_pos_after_h", 0.0))
    if before_h <= 0 and after_h <= 0:
        return snaps_with_labels
    before_td = pd.to_timedelta(before_h, unit="h")
    after_td  = pd.to_timedelta(after_h, unit="h")
    def _per_group(g):
        g = g.sort_values("snapshot_time")
        pos_times = g.loc[g["label"] > 0, "snapshot_time"].to_numpy()
        if pos_times.size == 0: return g
        keep = g["label"] > 0
        tvals = g["snapshot_time"].to_numpy()
        for tpos in pos_times:
            keep |= (tvals < (tpos - before_td)) | (tvals > (tpos + after_td))
        return g[keep]
    parts = []
    for _, g in snaps_with_labels.groupby(["hashed_mother_id", "episode_idx"], sort=False):
        parts.append(_per_group(g))
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Subsample (PRE-workers) — optional
# =============================================================================
def _subsample_pregnancies_before_workers(rt: pd.DataFrame,
                                          cfg: Dict,
                                          Le: pd.DataFrame,
                                          has_event_time: bool) -> pd.DataFrame:
    if not cfg.get("subsample_per_label", False):
        return rt
    eb = (rt.groupby(["hashed_mother_id","episode_idx"], as_index=False)["event_time_abs"].agg(ep_start="min", ep_end="max"))
    eb = eb.merge(rt[["hashed_mother_id","episode_idx","pregnancy_index"]].drop_duplicates(),
                  on=["hashed_mother_id","episode_idx"], how="left", validate="one_to_one")
    eb["hashed_mother_id"] = eb["hashed_mother_id"].astype(str)
    eb["pregnancy_index"]  = pd.to_numeric(eb["pregnancy_index"], errors="coerce").astype("Int64")

    if has_event_time and Le is not None and not Le.empty:
        E = Le.copy()
        E["hashed_mother_id"] = E["hashed_mother_id"].astype(str)
        E["pregnancy_index"]  = pd.to_numeric(E["pregnancy_index"], errors="coerce").astype("Int64")
        if not np.issubdtype(E["event_time"].dtype, np.datetime64):
            E["event_time"] = _parse_datetime_series(E["event_time"])
        E["label"] = pd.to_numeric(E["label"], errors="coerce").fillna(0).astype(int)
        E = E.dropna(subset=["event_time"])
        J = E.merge(eb, on=["hashed_mother_id","pregnancy_index"], how="inner")
        J = J[(J["event_time"] >= J["ep_start"]) & (J["event_time"] <= J["ep_end"]) & (J["label"] > 0)]
        preg_flag = (J.groupby(["hashed_mother_id","pregnancy_index"], as_index=False)
                       .size().assign(pos_flag=1)[["hashed_mother_id","pregnancy_index","pos_flag"]])
    else:
        preg_flag = (rt[["hashed_mother_id","pregnancy_index"]].drop_duplicates().assign(pos_flag=0))

    pregs = (rt[["hashed_mother_id","pregnancy_index"]].drop_duplicates()
             .merge(preg_flag, on=["hashed_mother_id","pregnancy_index"], how="left")
             .fillna({"pos_flag":0}))
    rng   = np.random.RandomState(int(cfg.get("subsample_seed", 123)))
    n_per = int(cfg.get("subsample_n_per_class", 100))
    keep_list = []
    for flag in (0, 1):
        sub = pregs[pregs["pos_flag"] == flag]
        if len(sub) <= n_per:
            keep_list.append(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=n_per, replace=False)
            keep_list.append(sub.loc[idx])
    keep_pregs = pd.concat(keep_list, ignore_index=True).drop_duplicates()
    keep_eps = eb.merge(keep_pregs, on=["hashed_mother_id","pregnancy_index"], how="inner")[["hashed_mother_id","episode_idx"]].drop_duplicates()
    return rt.merge(keep_eps, on=["hashed_mother_id","episode_idx"], how="inner")


# =============================================================================
# Worker I/O
# =============================================================================
def _init_worker(rt_in_memory: Optional[pd.DataFrame],
                 rt_dataset_path: Optional[str],
                 cfg: Dict,
                 labels_events: Optional[pd.DataFrame],
                 has_event_time: bool,
                 static_full: Optional[pd.DataFrame]):
    global _RT_MEM, _RT_DS_PATH, _CFG, _LABELS_EVENTS, _HAS_EVENT_TIME, _STATIC_FULL
    _RT_MEM = rt_in_memory
    _RT_DS_PATH = rt_dataset_path
    _CFG = cfg
    _LABELS_EVENTS = labels_events
    _HAS_EVENT_TIME = has_event_time
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

def _one_worker_indices(key) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str]]]:
    mid, ep = key
    grp = _load_group_frame(key)
    if grp.empty:
        return None
    grp_sorted = grp.sort_values("event_time_abs")
    preg = grp_sorted["pregnancy_index"].iloc[0]

    static_row = _get_static_row(str(mid), int(preg))
    measurements = _CFG.get("measurements", [])
    labs = _CFG.get("labs", [])

    X_grp, Y_base, feat_cols = _features_for_group(grp, _CFG, measurements, labs, static_row)
    if X_grp.empty: return None

    snaps = Y_base.copy()
    snaps["episode_idx"] = int(ep)
    snaps["hashed_mother_id"] = str(mid)

    Le_ep = _LABELS_EVENTS
    if _HAS_EVENT_TIME and Le_ep is not None and not Le_ep.empty:
        if "episode_idx" in Le_ep.columns and pd.notna(ep):
            Le_ep = Le_ep[(Le_ep["hashed_mother_id"] == str(mid)) &
                          (Le_ep["pregnancy_index"] == preg) &
                          (Le_ep["episode_idx"] == int(ep))]
        else:
            Le_ep = Le_ep[(Le_ep["hashed_mother_id"] == str(mid)) &
                          (Le_ep["pregnancy_index"] == preg)]

    labs_df = _apply_external_labels_overlap(snaps, _CFG, Le_ep, _HAS_EVENT_TIME)
    labs_df = _apply_exclusion_around_positives(labs_df, _CFG)

    keep_times = set(labs_df["snapshot_time"].astype("int64").to_numpy())
    X_grp = X_grp[X_grp["snapshot_time"].astype("int64").isin(keep_times)].reset_index(drop=True)

    X_grp["episode_idx"] = int(ep)
    labs_df["episode_idx"] = int(ep)

    return X_grp, labs_df.reset_index(drop=True), feat_cols


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

    with timed("read_aligned_labels"):
        Le, has_event_time = _read_aligned_labels(cfg)

    with timed("pre_worker_subsample"):
        rt_filt = _subsample_pregnancies_before_workers(rt, cfg, Le, has_event_time)

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

        _init_worker(init_rt_for_workers, rt_ds_path, cfg, Le, has_event_time, static_df_full)

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

            # prevent collisions except hashed_mother_id and pregnancy_index
            dup = (set(static_df.columns) & set(X_rt_only.columns)) - {"hashed_mother_id", "pregnancy_index"}
            if dup:
                static_df = static_df.rename(columns={c: f"static__{c}" for c in dup})

            # ensure pregnancy_index type alignment if exists
            if "pregnancy_index" in static_df.columns:
                static_df["pregnancy_index"] = pd.to_numeric(static_df["pregnancy_index"], errors="coerce").astype("Int64")

            # Merge on mother + (optionally) pregnancy_index when present
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
