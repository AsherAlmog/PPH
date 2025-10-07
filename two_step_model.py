# train_two_tower.py
# Two-tower PPH trainer: Static (tabular) + Realtime (sequence) + Fusion
# - Auto-builds caches if missing (static_merged.parquet, rt_episodes.parquet)
# - GPU-accelerated where available (LightGBM device='gpu', PyTorch CUDA)

from __future__ import annotations
import os, json, math, random, gc
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------------- ML / DL ----------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Prefer LightGBM for static (fast & strong). Fallback to XGBoost if you prefer.
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
    from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict = {
    # ---------- Paths ----------
    "base_dir": r"D:\PPH",

    # Source raw files (used only if caches missing)
    "mothers_csv": r"D:\PPH\MF_Maternal_table_20250812.csv",
    "fetus_csv":   r"D:\PPH\MF_FETAL_TABL_20250812_132000.csv",
    "realtime_pickle": r"D:\PPH\pph_wide_timeline.pkl",   # unified wide timeline pickle
    "realtime_files": [],  # optional extra parts

    # Documented PPH (ground truth)
    "pph_doc_csv": r"D:\PPH\MF_mother_pph_20250812.csv",
    "pph_doc_id_col": "hashed_mother_id",
    "pph_doc_label_col": "PPH",        # column with Hebrew "כן"/"לא"
    "pph_yes_token": "כן",
    "pph_no_token":  "לא",

    # ---------- Caches (will be created if missing) ----------
    "cache_dir": r"D:\PPH\.cache_two_tower",
    "static_cache": r"D:\PPH\.cache_two_tower\static_merged.parquet",
    "rt_cache":     r"D:\PPH\.cache_two_tower\rt_episodes.parquet",

    # ---------- IDs & time ----------
    "id_col": "mother_id",
    "mothers_id_col": "hashed_mother_id",
    "fetus_id_col": "hashed_mother_id",
    "rt_time_col":  "event_time_abs",     # in the pickle
    "rt_birth_col": "birth_datetime",
    "episode_gap_days": 183,              # split labors if >= 6 months apart

    # ---------- Keep columns from mothers (static) ----------
    "mothers_keep_columns": [
        # Dates / admin
        "child_birth_date","department_admission","department_discharge",
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
        # 24h pre-birth vitals
        "gbs_vagina","gbs_urine",
        "temp_24h_max_before_birth","temp_24h_min_before_birth","temp_24h_last_before_birth",
        "systolic_pressure_24h_last_before_birth","systolic_pressure_24h_max_before_birth","systolic_pressure_24h_min_before_birth",
        "diastolic_pressure_24h_last_before_birth","diastolic_pressure_24h_max_before_birth","diastolic_pressure_24h_min_before_birth",
        "saturation_24h_last_before_birth","saturation_24h_max_before_birth","saturation_24h_min_before_birth",
        # Baseline labs
        "WBC","HGB","PLT","Fibrinogen","Creatinine -Blood",
        # Interventions
        "clexane_in_regular_drugs","aspirin_in_regular_drugs","antibiotic_during_delivery",
        # Anesthesia
        "anesthesia_local","anesthesia_epidural","anesthesia_general","anesthesia_spinal","no_anesthesia",
        # Outcomes
        "death_date",
    ],

    # ---------- Fetus keep columns (EXACT LIST) ----------
    "fetus_keep_cols": [
        "hashed_mother_id",
        "newborn_medical_record",
        "birth_medical_record",
        "birth_time",
        "birth_type",
        "pose",
        "apgar1",
        "apgar5",
        "weight",
        "stillborn",
        "pregnancy_weeks",
        "days_in_hospital",
        "PH -Cord, V",
        "PH -Cord, Ar",
        "Base Excess -Cord, V",
        "Base Excess -Cord, Ar",
        "death_date"
    ],

    # ---------- Realtime signals (canonical names) ----------
    "vitals_map": {  # source name (pickle) -> canonical
        "sistol":     "sbp",
        "diastol":    "dbp",
        "BP - Mean":  "map",
        "pulse":      "hr",
        "heat":       "temp",
        "saturation": "spo2",
    },
    "labs_list": (
        "HGB","PLT","WBC","FIBRINOGEN","CREATININE_BLOOD"
    ),

    # ---------- RT featureization ----------
    "snapshot_every": "5min",       # grid spacing for sequences
    "max_monitor_duration": "6h",
    "baseline_window": "30min",     # baseline for deltas
    "seq_len_cap": 80,              # cap sequence length (truncate tail)
    "rt_channels": ["sbp","dbp","map","hr","temp","spo2"],  # included vitals

    # ---------- Training ----------
    "val_group_frac": 0.2,  # holdout by (mother, episode)
    "random_state": 42,

    # Static tower (LightGBM or XGBoost fallback)
    "lgb_params": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "device_type": "gpu",   # 'gpu' if available else will fallback at runtime
    },

    # Realtime tower (GRU)
    "rt_hidden": 64,
    "rt_layers": 1,
    "rt_dropout": 0.1,
    "batch_size": 128,
    "epochs": 6,
    "lr": 1e-3,

    # Fusion
    "fusion_C": 1.0,

    # ---------- Outputs ----------
    "out_dir": r"D:\PPH\tt_out",
}

# =============================================================================
# Utils
# =============================================================================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _coerce_num(s: pd.Series): return pd.to_numeric(s, errors="coerce")

def _read_realtime_part(path: str, time_col: str, birth_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=0)
    elif ext == ".csv":
        df = pd.read_csv(path)
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

# =============================================================================
# Static builder (mothers + fetus)
# =============================================================================
def parse_mothers_static(cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["mothers_csv"], low_memory=False)
    df = df.rename(columns={cfg["mothers_id_col"]: cfg["id_col"]})
    keep = [cfg["id_col"]] + [c for c in cfg["mothers_keep_columns"] if c in df.columns]
    df = df[keep].copy()

    # numeric coercion for non-date text
    for c in df.columns:
        if c == cfg["id_col"]:
            continue
        if df[c].dtype == object and ("date" not in c.lower()):
            df[c] = _coerce_num(df[c])

    # BMI
    if "height" in df.columns:
        h_m = _coerce_num(df["height"]) / 100.0
        if "weight_before_birth" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_birth"]) / (h_m**2)).replace([np.inf, -np.inf], np.nan).astype("float32")
        elif "weight_before_pregnancy" in df.columns:
            df["bmi"] = (_coerce_num(df["weight_before_pregnancy"]) / (h_m**2)).replace([np.inf, -np.inf], np.nan).astype("float32")

    # Flags
    if "number_of_births_P" in df.columns:
        df["parity_primip"] = (df["number_of_births_P"].fillna(0) == 0).astype("int8")
    if "number_of_cesareans_CS" in df.columns:
        df["prior_csection"] = (df["number_of_cesareans_CS"].fillna(0) > 0).astype("int8")
    for c in ["anesthesia_local","anesthesia_epidural","anesthesia_general","anesthesia_spinal","no_anesthesia","vbac_now"]:
        if c in df.columns:
            df[c] = (_coerce_num(df[c]).fillna(0) > 0).astype("int8")

    df = df.drop_duplicates(subset=[cfg["id_col"]]).reset_index(drop=True)
    df["hashed_mother_id"] = df[cfg["id_col"]]
    return df

def parse_fetus_agg(cfg: Dict) -> pd.DataFrame:
    keep_cols = cfg["fetus_keep_cols"]
    df = pd.read_csv(cfg["fetus_csv"], low_memory=False)
    if cfg.get("fetus_id_col", "hashed_mother_id") != "hashed_mother_id" and cfg["fetus_id_col"] in df.columns:
        df = df.rename(columns={cfg["fetus_id_col"]: "hashed_mother_id"})

    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()
    if df.empty:
        return pd.DataFrame(columns=[cfg["id_col"], "hashed_mother_id"])

    for c in ("birth_time","death_date"):
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")

    num_map = {
        "apgar1":"float32","apgar5":"float32","weight":"float32",
        "pregnancy_weeks":"float32","days_in_hospital":"float32",
        "PH -Cord, V":"float32","PH -Cord, Ar":"float32",
        "Base Excess -Cord, V":"float32","Base Excess -Cord, Ar":"float32",
    }
    for c, dt in num_map.items():
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)

    if "stillborn" in df.columns:
        s = pd.to_numeric(df["stillborn"], errors="coerce").fillna(0)
        df["stillborn"] = (s > 0).astype("int8")

    def _mode(s: pd.Series):
        if s.dropna().empty: return np.nan
        return s.mode(dropna=True).iloc[0]

    g = df.groupby("hashed_mother_id", sort=False)
    agg = g.agg(
        fetus_count=("newborn_medical_record","count") if "newborn_medical_record" in df.columns
                     else ("birth_medical_record","count") if "birth_medical_record" in df.columns
                     else ("weight","count"),
        stillborn_any=("stillborn","max") if "stillborn" in df.columns else ("weight","count"),
        apgar1_min=("apgar1","min") if "apgar1" in df.columns else ("weight","count"),
        apgar1_max=("apgar1","max") if "apgar1" in df.columns else ("weight","count"),
        apgar1_mean=("apgar1","mean") if "apgar1" in df.columns else ("weight","count"),
        apgar5_min=("apgar5","min") if "apgar5" in df.columns else ("weight","count"),
        apgar5_max=("apgar5","max") if "apgar5" in df.columns else ("weight","count"),
        apgar5_mean=("apgar5","mean") if "apgar5" in df.columns else ("weight","count"),
        birthweight_min_g=("weight","min") if "weight" in df.columns else ("apgar1","count"),
        birthweight_max_g=("weight","max") if "weight" in df.columns else ("apgar1","count"),
        birthweight_mean_g=("weight","mean") if "weight" in df.columns else ("apgar1","count"),
        preg_weeks_min=("pregnancy_weeks","min") if "pregnancy_weeks" in df.columns else ("weight","count"),
        preg_weeks_max=("pregnancy_weeks","max") if "pregnancy_weeks" in df.columns else ("weight","count"),
        preg_weeks_mean=("pregnancy_weeks","mean") if "pregnancy_weeks" in df.columns else ("weight","count"),
        cord_ph_v_mean=("PH -Cord, V","mean") if "PH -Cord, V" in df.columns else ("weight","count"),
        cord_ph_a_mean=("PH -Cord, Ar","mean") if "PH -Cord, Ar" in df.columns else ("weight","count"),
        base_excess_v_mean=("Base Excess -Cord, V","mean") if "Base Excess -Cord, V" in df.columns else ("weight","count"),
        base_excess_a_mean=("Base Excess -Cord, Ar","mean") if "Base Excess -Cord, Ar" in df.columns else ("weight","count"),
        days_in_hosp_mean=("days_in_hospital","mean") if "days_in_hospital" in df.columns else ("weight","count"),
    ).reset_index()

    def _null_if_count(col):
        if col in agg.columns and agg[col].dtype.kind in ("i","u"):
            agg[col] = np.nan
    for c in ["stillborn_any","apgar1_min","apgar1_max","apgar1_mean",
              "apgar5_min","apgar5_max","apgar5_mean",
              "birthweight_min_g","birthweight_max_g","birthweight_mean_g",
              "preg_weeks_min","preg_weeks_max","preg_weeks_mean",
              "cord_ph_v_mean","cord_ph_a_mean","base_excess_v_mean","base_excess_a_mean",
              "days_in_hosp_mean"]:
        _null_if_count(c)

    modes = g.agg(
        mode_birth_type=("birth_type", _mode) if "birth_type" in df.columns else ("pose", _mode),
        mode_pose=("pose", _mode) if "pose" in df.columns else ("birth_type", _mode)
    ).reset_index()

    agg = agg.merge(modes, on="hashed_mother_id", how="left")
    agg["fetus_count"] = agg["fetus_count"].fillna(0).astype("int16")
    if "stillborn_any" in agg.columns:
        agg["stillborn_any"] = agg["stillborn_any"].fillna(0).astype("int8")
    agg["multiple_gestation"] = (agg["fetus_count"] >= 2).astype("int8")
    agg[cfg["id_col"]] = agg["hashed_mother_id"]

    ordered = [cfg["id_col"], "hashed_mother_id", "fetus_count", "multiple_gestation",
               "stillborn_any",
               "apgar1_min","apgar1_max","apgar1_mean",
               "apgar5_min","apgar5_max","apgar5_mean",
               "birthweight_min_g","birthweight_max_g","birthweight_mean_g",
               "preg_weeks_min","preg_weeks_max","preg_weeks_mean",
               "cord_ph_v_mean","cord_ph_a_mean","base_excess_v_mean","base_excess_a_mean",
               "days_in_hosp_mean","mode_birth_type","mode_pose"]
    ordered = [c for c in ordered if c in agg.columns]
    return agg[ordered].copy()

def build_or_load_static_cache(cfg: Dict) -> pd.DataFrame:
    p = cfg["static_cache"]
    if os.path.exists(p):
        return pd.read_parquet(p)

    moms = parse_mothers_static(cfg)
    fetus = parse_fetus_agg(cfg)
    static_merged = moms.merge(fetus, on=[cfg["id_col"],"hashed_mother_id"], how="left")
    _ensure_dir(os.path.dirname(p))
    static_merged.to_parquet(p, index=False)
    return static_merged

# =============================================================================
# Realtime standardization + episode split
# =============================================================================
def load_realtime_multi(cfg: Dict) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    pkl = cfg.get("realtime_pickle")
    if pkl and os.path.exists(pkl):
        parts.append(_read_realtime_part(pkl, cfg["rt_time_col"], cfg["rt_birth_col"]))
    for path in cfg.get("realtime_files", []):
        if os.path.exists(path):
            parts.append(_read_realtime_part(path, cfg["rt_time_col"], cfg["rt_birth_col"]))
    if not parts:
        raise RuntimeError("No realtime data found.")
    rt = pd.concat(parts, ignore_index=True)
    rt = rt.rename(columns={
        cfg["rt_time_col"]:  "ts",
        cfg["rt_birth_col"]: "delivery_time",
        "hashed_mother_id":  cfg["id_col"],
    })
    rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce")
    rt["delivery_time"] = pd.to_datetime(rt["delivery_time"], errors="coerce")

    # map vitals -> canonical; ensure columns exist
    for src, dst in cfg["vitals_map"].items():
        if src in rt.columns and src != dst:
            rt = rt.rename(columns={src: dst})
    for v in cfg["vitals_map"].values():
        if v not in rt.columns: rt[v] = np.nan

    for lab in cfg["labs_list"]:
        if lab not in rt.columns: rt[lab] = np.nan
        else: rt[lab] = _coerce_num(rt[lab])

    # postpartum only
    rt = rt[rt["ts"] >= rt["delivery_time"]].copy()
    rt = rt.sort_values([cfg["id_col"], "delivery_time", "ts"]).reset_index(drop=True)
    rt["hashed_mother_id"] = rt[cfg["id_col"]]
    return rt

def add_episode_ids(rt: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Episode = contiguous rows per mother where delivery_time is same OR subsequent
       delivery is within 183 days. If delivery_time changes and gap >= 183 days → new episode.
    """
    gap = pd.to_timedelta(cfg["episode_gap_days"], unit="D")
    # assume rows sorted by mother, delivery_time, ts
    groups = []
    ep_ids = np.empty(len(rt), dtype="int32")
    cur = -1
    last_m = None
    last_deliv = None
    for i,(m, d) in enumerate(rt[[cfg["id_col"],"delivery_time"]].itertuples(index=False, name=None)):
        if (m != last_m):
            cur += 1
            ep_ids[i] = cur
            last_m, last_deliv = m, d
            continue
        # same mother: new episode if delivery_time changed and gap >= 183d
        if pd.isna(last_deliv) or pd.isna(d):
            # if missing, keep in same episode unless mother switched above
            ep_ids[i] = cur
        else:
            if d != last_deliv and (d - last_deliv) >= gap:
                cur += 1
            ep_ids[i] = cur
            last_deliv = d
    rt = rt.copy()
    rt["episode_idx"] = ep_ids
    return rt

def build_or_load_rt_cache(cfg: Dict) -> pd.DataFrame:
    p = cfg["rt_cache"]
    if os.path.exists(p):
        return pd.read_parquet(p)
    rt = load_realtime_multi(cfg)
    rt = add_episode_ids(rt, cfg)
    _ensure_dir(os.path.dirname(p))
    rt.to_parquet(p, index=False)
    return rt

# =============================================================================
# Labels from documented PPH
# =============================================================================
def load_doc_labels(cfg: Dict) -> pd.DataFrame:
    doc = pd.read_csv(cfg["pph_doc_csv"], low_memory=False)
    if cfg["pph_doc_id_col"] != cfg["id_col"] and cfg["pph_doc_id_col"] in doc.columns:
        doc = doc.rename(columns={cfg["pph_doc_id_col"]: cfg["id_col"]})
    if cfg["pph_doc_label_col"] not in doc.columns:
        raise ValueError("Doc PPH label column not found.")

    ymap = {
        cfg["pph_yes_token"]: 1,
        cfg["pph_no_token"]:  0,
    }
    lab = doc[[cfg["id_col"], cfg["pph_doc_label_col"]]].copy()
    lab["y_doc"] = lab[cfg["pph_doc_label_col"]].map(ymap)
    lab = lab.dropna(subset=["y_doc"]).copy()
    lab["y_doc"] = lab["y_doc"].astype("int8")
    lab = lab.drop(columns=[cfg["pph_doc_label_col"]]).drop_duplicates(subset=[cfg["id_col"]])
    return lab

# =============================================================================
# Sequence building (per episode)
# =============================================================================
def build_sequences(rt: pd.DataFrame, cfg: Dict) -> Tuple[dict, pd.DataFrame]:
    """
    Returns:
      seqs: dict[(mother_id, episode_idx)] -> np.ndarray [T, 2*C]  (abs + delta)
      index_df: rows with (mother_id, episode_idx, T_used)
    """
    vitals = cfg["rt_channels"]
    baseline_td = pd.to_timedelta(cfg["baseline_window"])

    seqs = {}
    rows = []

    for (m, e), g in rt.groupby([cfg["id_col"], "episode_idx"], sort=False):
        if g.empty:
            continue

        t0 = g["delivery_time"].iloc[0]
        t_end = min(g["ts"].max(), t0 + pd.to_timedelta(cfg["max_monitor_duration"]))

        # Build a regular time grid from delivery to t_end
        grid = pd.date_range(t0, t_end, freq=cfg["snapshot_every"], inclusive="left")
        n = len(grid)
        if n == 0:
            continue

        frame = g.set_index("ts").sort_index()
        early = frame[(frame.index >= t0) & (frame.index <= t0 + baseline_td)]

        # baselines: mean in first baseline window; else first non-null value
        baselines = {}
        for v in vitals:
            if v in early.columns and not early[v].dropna().empty:
                baselines[v] = float(early[v].dropna().mean())
            elif v in frame.columns and not frame[v].dropna().empty:
                baselines[v] = float(frame[v].dropna().iloc[0])
            else:
                baselines[v] = np.nan

        # forward-fill each vital onto the grid
        block = {}
        for v in vitals:
            if v in frame.columns:
                s = frame[v].astype("float32")
                block[v] = s.reindex(s.index.union(grid)).ffill().reindex(grid).to_numpy(dtype="float32")
            else:
                block[v] = np.full(n, np.nan, dtype="float32")

        X_abs = np.stack([block[v] for v in vitals], axis=1)  # [T, C]
        base = np.array([baselines[v] for v in vitals], dtype="float32")[None, :]
        X_delta = X_abs - base
        X = np.concatenate([X_abs, X_delta], axis=1)          # [T, 2*C]

        # cap sequence length
        cap = min(n, int(cfg["seq_len_cap"]))
        X = X[:cap]
        seqs[(m, int(e))] = X
        rows.append((m, int(e), cap))

    index_df = pd.DataFrame(rows, columns=[cfg["id_col"], "episode_idx", "T_used"])
    return seqs, index_df


# =============================================================================
# Static table → features, labels
# =============================================================================
def align_static_with_labels(static_df: pd.DataFrame, ydoc: pd.DataFrame, rt_index: pd.DataFrame, cfg: Dict):
    # Keep only mothers present in sequences
    moms = set(rt_index[cfg["id_col"]].unique().tolist())
    S = static_df[static_df[cfg["id_col"]].isin(moms)].copy()
    S = S.merge(ydoc, on=cfg["id_col"], how="left")
    S = S.dropna(subset=["y_doc"]).copy()
    S["y_doc"] = S["y_doc"].astype("int8")

    # Simple numeric-only features (keep ids)
    non_feat = {cfg["id_col"], "hashed_mother_id"}
    feat_cols = [c for c in S.columns if c not in non_feat and pd.api.types.is_numeric_dtype(S[c])]
    Xs = S[feat_cols].replace([np.inf,-np.inf], np.nan)
    Xs = Xs.fillna(Xs.median(numeric_only=True))
    for c in Xs.columns: Xs[c] = Xs[c].astype("float32")
    ys = S["y_doc"].astype("int8")
    ids = S[[cfg["id_col"]]].copy()
    return Xs, ys, ids, feat_cols

# =============================================================================
# Datasets for realtime sequences
# =============================================================================
class RTDataset(Dataset):
    def __init__(self, keys, seqs: dict, ymap: dict):
        self.keys = keys  # list of (mother, episode)
        self.seqs = seqs
        self.ymap = ymap

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        x = self.seqs[k]  # [T, 2*C]
        y = self.ymap.get(k[0], None)  # label by mother (doc)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

def rt_collate(batch):
    # pad sequences to max T in batch
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    C = xs[0].shape[1]
    maxT = max(lengths)
    padded = torch.zeros(len(xs), maxT, C, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0], :] = x
    return padded, torch.stack(ys), torch.tensor(lengths, dtype=torch.int64)

# =============================================================================
# GRU model
# =============================================================================
class GRUHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=in_ch, hidden_size=hidden, num_layers=layers,
                          dropout=(dropout if layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, lengths):
        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hN = self.gru(packed)  # hN: [layers, B, H]
        h = hN[-1]                 # [B, H]
        logit = self.head(h).squeeze(1)
        return logit

# =============================================================================
# Group split
# =============================================================================
def make_group_masks(keys: List[Tuple[str,int]], frac: float, seed: int = 42):
    # group by mother only (so different episodes for same mom stay in same split)
    moms = [k[0] for k in keys]
    codes, uniques = pd.factorize(pd.Series(moms), sort=False)
    rng = np.random.RandomState(seed)
    n_groups = len(uniques)
    n_val = max(1, int(math.ceil(frac * n_groups)))
    val_groups = set(rng.choice(np.arange(n_groups), size=n_val, replace=False).tolist())
    is_val = np.array([c in val_groups for c in codes])
    is_tr  = ~is_val
    tr_idx = np.where(is_tr)[0].tolist()
    va_idx = np.where(is_val)[0].tolist()
    return tr_idx, va_idx

# =============================================================================
# Metrics & plots
# =============================================================================
def summarize_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> str:
    y_pred = (y_score >= thr).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    return "\n".join([
        f"AUC: {auc:.4f}",
        f"AP:  {ap:.4f}",
        f"P@{thr:.2f}: {p:.4f}",
        f"R@{thr:.2f}: {r:.4f}",
        f"F1@{thr:.2f}: {f1:.4f}",
        f"Spec@{thr:.2f}: {spec:.4f} | Sens@{thr:.2f}: {sens:.4f}",
        f"CM@{thr:.2f}: TN={tn} FP={fp} FN={fn} TP={tp}"
    ])

def plot_roc_pr(y, p, out_roc, out_pr):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.savefig(out_roc, dpi=150); plt.close()
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    plt.figure(figsize=(6,5)); plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.legend(); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.savefig(out_pr, dpi=150); plt.close()

# =============================================================================
# Main
# =============================================================================
def main(cfg: Dict = CFG):
    _ensure_dir(cfg["out_dir"])
    # 1) Caches (build if missing)
    static_df = build_or_load_static_cache(cfg)
    rt = build_or_load_rt_cache(cfg)

    # 2) Labels (doc wins; per mother)
    ydoc = load_doc_labels(cfg)

    # 3) Build per-episode sequences
    seqs, idx_df = build_sequences(rt, cfg)
    if idx_df.empty:
        raise RuntimeError("No sequences built.")

    # 4) Static tower dataset (episode alignment by mother)
    Xs, ys, ids, feat_cols = align_static_with_labels(static_df, ydoc, idx_df, cfg)
    # Map mother -> label for sequence supervision
    ymap = {row[cfg["id_col"]]: int(row["y_doc"]) for _, row in
            Xs.join(ids)[[cfg["id_col"]]].join(ys).join(ids).dropna().reset_index(drop=True).assign(y_doc=ys).iterrows()}  # robust

    # Keep only episodes of mothers with labels
    keys_all = [(m, e) for (m, e) in seqs.keys() if m in ymap]
    if not keys_all:
        raise RuntimeError("No labeled sequences after filtering by doc labels.")

    # 5) Split by mother (grouped)
    tr_idx, va_idx = make_group_masks(keys_all, cfg["val_group_frac"], cfg["random_state"])
    tr_keys = [keys_all[i] for i in tr_idx]
    va_keys = [keys_all[i] for i in va_idx]

    # 6) Static tower train/val split (by mother)
    moms_all = ids[cfg["id_col"]].astype(str).tolist()
    ids_series = ids[cfg["id_col"]].astype(str)
    tr_moms = set(k[0] for k in tr_keys)
    va_moms = set(k[0] for k in va_keys)
    tr_mask = ids_series.isin(tr_moms).values
    va_mask = ids_series.isin(va_moms).values

    Xs_tr, ys_tr = Xs.iloc[tr_mask], ys.iloc[tr_mask]
    Xs_va, ys_va = Xs.iloc[va_mask], ys.iloc[va_mask]

    # 7) Train Static tower
    if _HAS_LGB:
        params = dict(cfg["lgb_params"])
        # If no GPU, fallback
        if params.get("device_type","gpu") == "gpu":
            try:
                lgb.register_logger(None)
            except Exception:
                pass
        static_model = lgb.LGBMClassifier(**params)
        static_model.fit(Xs_tr, ys_tr, eval_set=[(Xs_va, ys_va)], verbose=False)
        s_val = static_model.predict_proba(Xs_va)[:,1]
        s_tr  = static_model.predict_proba(Xs_tr)[:,1]
    else:
        # XGBoost fallback
        static_model = XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="auc",
            tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
            random_state=cfg["random_state"]
        )
        static_model.fit(Xs_tr, ys_tr, eval_set=[(Xs_va, ys_va)], verbose=False)
        s_val = static_model.predict_proba(Xs_va)[:,1]
        s_tr  = static_model.predict_proba(Xs_tr)[:,1]

    print("\n[Static tower] Validation\n" + summarize_metrics(ys_va.values, s_val))
    plot_roc_pr(ys_va.values, s_val, os.path.join(cfg["out_dir"], "static_roc.png"),
                              os.path.join(cfg["out_dir"], "static_pr.png"))

    # 8) Train Realtime (sequence) tower
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = len(cfg["rt_channels"]) * 2  # abs + delta
    model = GRUHead(in_ch, cfg["rt_hidden"], cfg["rt_layers"], cfg["rt_dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()

    ds_tr = RTDataset(tr_keys, seqs, ymap)
    ds_va = RTDataset(va_keys, seqs, ymap)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, collate_fn=rt_collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False, collate_fn=rt_collate, num_workers=0)

    best_auc, best_state = -1, None
    for ep in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for xb, yb, lens in dl_tr:
            xb, yb, lens = xb.to(device), yb.to(device), lens.to(device)
            opt.zero_grad()
            logits = model(xb, lens)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        train_loss = total / len(ds_tr)

        # val
        model.eval()
        probs, ytrue = [], []
        with torch.no_grad():
            for xb, yb, lens in dl_va:
                xb, yb, lens = xb.to(device), yb.to(device), lens.to(device)
                logits = model(xb, lens)
                p = torch.sigmoid(logits)
                probs.append(p.detach().cpu().numpy())
                ytrue.append(yb.detach().cpu().numpy())
        p_va = np.concatenate(probs); y_va = np.concatenate(ytrue)
        auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
        print(f"[RT] epoch {ep+1}/{cfg['epochs']} loss={train_loss:.4f} AUC={auc:.4f}")
        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_state = auc, {k: v.cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    # final RT metrics/plots
    model.eval()
    probs, ytrue = [], []
    with torch.no_grad():
        for xb, yb, lens in dl_va:
            xb, yb, lens = xb.to(device), yb.to(device), lens.to(device)
            logits = model(xb, lens)
            probs.append(torch.sigmoid(logits).cpu().numpy()); ytrue.append(yb.cpu().numpy())
    r_val = np.concatenate(probs); y_va = np.concatenate(ytrue)
    print("\n[RT tower] Validation\n" + summarize_metrics(y_va, r_val))
    plot_roc_pr(y_va, r_val, os.path.join(cfg["out_dir"], "rt_roc.png"),
                            os.path.join(cfg["out_dir"], "rt_pr.png"))

    # 9) Fusion model (logistic regression on concatenated scores)
    # Align static/rt validation by mother
    va_moms_sorted = sorted(list(va_moms)) if 'va_moms' in locals() else sorted({k[0] for k in va_keys})
    # Build per-mother static score map
    S_va = ids.iloc[va_mask].reset_index(drop=True)
    s_map = {S_va[cfg["id_col"]].iloc[i]: float(s_val[i]) for i in range(len(S_va))}
    # Build per-mother rt average score over episodes (or max)
    rt_scores = {}
    for (m, e), x in zip(va_keys, r_val):  # r_val is per-batch concat in same order as dl_va iteration; map robustly:
        # safer: recompute r_val keyed
        pass
    # Safer recompute keyed r_val:
    probs, keys = [], []
    with torch.no_grad():
        for xb, yb, lens in DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False, collate_fn=rt_collate):
            logits = model(xb.to(device), lens.to(device))
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
    # match to ds_va.keys order
    for k, p in zip(ds_va.keys, probs):
        rt_scores.setdefault(k[0], []).append(float(p))
    r_map = {m: float(np.mean(v)) for m, v in rt_scores.items()}

    # Fusion dataset (validation)
    X_f_va, y_f_va = [], []
    for m in va_moms_sorted:
        if (m in s_map) and (m in r_map) and (m in ymap):
            X_f_va.append([s_map[m], r_map[m]])
            y_f_va.append(ymap[m])
    X_f_va = np.array(X_f_va, dtype=np.float32); y_f_va = np.array(y_f_va, dtype=np.int32)

    # Fusion dataset (train)
    tr_moms_sorted = sorted(list(tr_moms)) if 'tr_moms' in locals() else sorted({k[0] for k in tr_keys})
    S_tr = ids.iloc[tr_mask].reset_index(drop=True)
    s_map_tr = {S_tr[cfg["id_col"]].iloc[i]: float(s_tr[i]) for i in range(len(S_tr))}
    probs_tr = []
    with torch.no_grad():
        for xb, yb, lens in DataLoader(RTDataset(tr_keys, seqs, ymap), batch_size=cfg["batch_size"],
                                       shuffle=False, collate_fn=rt_collate):
            logits = model(xb.to(device), lens.to(device))
            p = torch.sigmoid(logits).cpu().numpy()
            probs_tr.extend(p.tolist())
    rt_scores_tr = {}
    for k, p in zip(tr_keys, probs_tr):
        rt_scores_tr.setdefault(k[0], []).append(float(p))
    r_map_tr = {m: float(np.mean(v)) for m, v in rt_scores_tr.items()}

    X_f_tr, y_f_tr = [], []
    for m in tr_moms_sorted:
        if (m in s_map_tr) and (m in r_map_tr) and (m in ymap):
            X_f_tr.append([s_map_tr[m], r_map_tr[m]])
            y_f_tr.append(ymap[m])
    X_f_tr = np.array(X_f_tr, dtype=np.float32); y_f_tr = np.array(y_f_tr, dtype=np.int32)

    fusion = LogisticRegression(C=cfg["fusion_C"], max_iter=200)
    fusion.fit(X_f_tr, y_f_tr)
    p_f_va = fusion.predict_proba(X_f_va)[:,1]
    print("\n[Fusion] Validation\n" + summarize_metrics(y_f_va, p_f_va))
    plot_roc_pr(y_f_va, p_f_va, os.path.join(cfg["out_dir"], "fusion_roc.png"),
                              os.path.join(cfg["out_dir"], "fusion_pr.png"))

    # 10) Save artifacts
    # Static tower
    if _HAS_LGB:
        static_model.booster_.save_model(os.path.join(cfg["out_dir"], "static_model.txt"))
        with open(os.path.join(cfg["out_dir"], "static_features.json"), "w") as f:
            json.dump(feat_cols, f, indent=2)
    else:
        static_model.save_model(os.path.join(cfg["out_dir"], "static_model.json"))

    # RT tower
    torch.save({"state_dict": model.state_dict(),
                "in_ch": in_ch, "hidden": cfg["rt_hidden"],
                "layers": cfg["rt_layers"], "dropout": cfg["rt_dropout"]},
               os.path.join(cfg["out_dir"], "rt_gru.pt"))

    # Fusion
    with open(os.path.join(cfg["out_dir"], "fusion_lr.pkl"), "wb") as f:
        import pickle; pickle.dump(fusion, f)

    print("\n[OK] Saved models & plots in:", cfg["out_dir"])

if __name__ == "__main__":
    main()
