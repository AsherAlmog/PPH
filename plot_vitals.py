# plot_vitals.py
from __future__ import annotations
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
CFG: Dict = {
    "features_path": r"D:\PPH\features_all.parquet",
    "labels_path":   r"D:\PPH\out_labels\labels_vector.parquet",

    "n_per_class": 25,
    "random_state": 123,

    # feature columns produced by your builder
    "hgb_col": "HGB_fine_last",
    "hct_col": "HCT_fine_last",

    # axes cosmetics (None = auto)
    "hgb_ylim": None,  # e.g., (5, 17)
    "hct_ylim": None,  # e.g., (15, 55)
    "xlim_minutes": None,  # e.g., (-120, 720)

    "out_dir": r"D:\PPH\episode_plots",
    "dpi": 150,
}

EP_KEYS = ("episode_idx", "pregnancy_index")

# ===================== UTILITIES =====================

def _parquet_columns(path: str) -> List[str]:
    """Return column names without loading the table."""
    try:
        import pyarrow.parquet as pq
        return [str(n) for n in pq.ParquetFile(path).schema.names]
    except Exception:
        # Fallback (may load data on small files)
        return list(pd.read_parquet(path, engine="pyarrow", columns=None, nthreads=1).columns)  # type: ignore

def _birth_keys(df: pd.DataFrame) -> List[str]:
    have = [c for c in EP_KEYS if c in df.columns]
    if not have:
        raise ValueError("Need either 'episode_idx' OR 'pregnancy_index' in dataframe.")
    return ["hashed_mother_id", have[0]]

def _coerce_idcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hashed_mother_id" in df.columns:
        df["hashed_mother_id"] = df["hashed_mother_id"].astype("string")
    for c in EP_KEYS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
    if "snapshot_time" in df.columns and not np.issubdtype(df["snapshot_time"].dtype, np.datetime64):
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    return df

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))

def _pick_n(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed)
    idx = _rng(seed).choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[np.sort(idx)]

# ===================== READERS =====================

def _read_labels(path: str) -> pd.DataFrame:
    cols_exist = set(_parquet_columns(path))
    use = [c for c in ["hashed_mother_id", "episode_idx", "pregnancy_index", "snapshot_time", "label"] if c in cols_exist]
    must = {"hashed_mother_id", "snapshot_time", "label"}
    if not must.issubset(use):
        raise ValueError("labels parquet must include at least: hashed_mother_id, snapshot_time, label, and one of episode_idx/pregnancy_index.")
    L = pd.read_parquet(path, columns=use)
    L = _coerce_idcols(L)
    L["label"] = pd.to_numeric(L["label"], errors="coerce").fillna(0).astype("int8")
    return L.dropna(subset=["snapshot_time"])

def _first_bleed_per_birth(L: pd.DataFrame) -> pd.DataFrame:
    k = _birth_keys(L)
    pos = (L[L["label"] > 0]
           .sort_values("snapshot_time")
           .drop_duplicates(k, keep="first")
           .rename(columns={"snapshot_time": "first_bleed_time"}))
    births = L.drop_duplicates(k)[k]
    births = births.merge(pos[k + ["first_bleed_time"]], on=k, how="left")
    births["bleed_flag"] = births["first_bleed_time"].notna().astype("int8")
    return births

def _read_features(path: str, hgb_col: str, hct_col: str) -> pd.DataFrame:
    cols_exist = set(_parquet_columns(path))
    needed = ["hashed_mother_id", "episode_idx", "pregnancy_index", "snapshot_time"]
    for c in (hgb_col, hct_col):
        if c and c in cols_exist:
            needed.append(c)
    cols = [c for c in needed if c in cols_exist]
    if "snapshot_time" not in cols or "hashed_mother_id" not in cols:
        raise ValueError("features parquet must include at least: hashed_mother_id, snapshot_time, and one of episode_idx/pregnancy_index.")
    X = pd.read_parquet(path, columns=cols)
    X = _coerce_idcols(X)
    for c in (hgb_col, hct_col):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X.dropna(subset=["snapshot_time"])

# ===================== TIME HELPERS =====================

def _episode_start(df: pd.DataFrame, k: List[str]) -> pd.DataFrame:
    return (df.groupby(k, observed=True, sort=False)["snapshot_time"]
              .min().rename("episode_start").reset_index())

def _attach_mins_postpartum(df: pd.DataFrame, ep0: pd.DataFrame, k: List[str]) -> pd.DataFrame:
    out = df.merge(ep0, on=k, how="left")
    out["mins_postpartum"] = (out["snapshot_time"] - out["episode_start"]).dt.total_seconds() / 60.0
    return out

# ===================== PLOTTING =====================

def _plot_one_birth(
    birth_df: pd.DataFrame,
    first_bleed_time: Optional[pd.Timestamp],
    hgb_col: str,
    hct_col: str,
    out_path: str,
    cfg: Dict
):
    birth_df = birth_df.sort_values("snapshot_time")
    tmin = birth_df["mins_postpartum"].to_numpy()

    hgb = birth_df[hgb_col].to_numpy(dtype=np.float32) if hgb_col in birth_df.columns else None
    hct = birth_df[hct_col].to_numpy(dtype=np.float32) if hct_col in birth_df.columns else None

    # If neither exists, skip plotting this birth
    if hgb is None and hct is None:
        print(f"[WARN] Skipping (no {hgb_col}/{hct_col}) for birth: {birth_df[['hashed_mother_id']].iloc[0].to_dict()}")
        return

    ncols = (hgb is not None) + (hct is not None)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6.5*ncols, 4.5), sharex=True)
    if ncols == 1:
        axes = [axes]
    ax_idx = 0

    title_bits = []
    if "hashed_mother_id" in birth_df.columns:
        title_bits.append(f"hashed_mother_id={birth_df['hashed_mother_id'].iloc[0]}")
    for key in ("episode_idx", "pregnancy_index"):
        if key in birth_df.columns:
            title_bits.append(f"{key}={birth_df[key].iloc[0]}")
    plt.suptitle(" | ".join(title_bits), fontsize=11)

    # HGB absolute
    if hgb is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(tmin, hgb, marker="o", linewidth=1)
        ax.set_ylabel("HGB (g/dL)")
        if cfg.get("hgb_ylim"): ax.set_ylim(*cfg["hgb_ylim"])
        ax.set_xlabel("Minutes postpartum")
        if cfg.get("xlim_minutes"): ax.set_xlim(cfg["xlim_minutes"])
        if first_bleed_time is not None and "episode_start" in birth_df.columns:
            xbleed = (first_bleed_time - birth_df["episode_start"].iloc[0]).total_seconds() / 60.0
            ax.axvline(xbleed, color="red", linestyle="--", linewidth=1)

    # HCT absolute
    if hct is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(tmin, hct, marker="o", linewidth=1)
        ax.set_ylabel("HCT (%)")
        if cfg.get("hct_ylim"): ax.set_ylim(*cfg["hct_ylim"])
        ax.set_xlabel("Minutes postpartum")
        if cfg.get("xlim_minutes"): ax.set_xlim(cfg["xlim_minutes"])
        if first_bleed_time is not None and "episode_start" in birth_df.columns:
            xbleed = (first_bleed_time - birth_df["episode_start"].iloc[0]).total_seconds() / 60.0
            ax.axvline(xbleed, color="red", linestyle="--", linewidth=1)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=cfg.get("dpi", 150))
    plt.close(fig)

# ===================== MAIN =====================

def main(cfg: Dict = CFG):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Labels → births with earliest bleed time
    L = _read_labels(cfg["labels_path"])
    births = _first_bleed_per_birth(L)
    k = _birth_keys(births)

    pos_births = births[births["bleed_flag"] == 1]
    neg_births = births[births["bleed_flag"] == 0]
    if pos_births.empty or neg_births.empty:
        raise RuntimeError("Need both positive and negative births in labels to sample from.")

    pos_sample = _pick_n(pos_births, cfg["n_per_class"], cfg["random_state"])
    neg_sample = _pick_n(neg_births, cfg["n_per_class"], cfg["random_state"])
    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)

    # Features (only selected births)
    X = _read_features(cfg["features_path"], cfg["hgb_col"], cfg["hct_col"])
    X = X.merge(sample[k], on=k, how="inner")

    # Episode start & mins postpartum
    ep0 = _episode_start(X, k)
    X = _attach_mins_postpartum(X, ep0, k)

    # Attach first bleed time
    X = X.merge(sample[k + ["first_bleed_time", "bleed_flag"]], on=k, how="left")

    # Plot each birth
    n_plotted = 0
    for keys, df_b in X.groupby(k, observed=True, sort=False):
        mid = df_b["hashed_mother_id"].iloc[0]
        epi_val = df_b[k[1]].iloc[0]
        out_name = f"{str(mid)}__{k[1]}={str(epi_val)}.png"
        first_bleed_time = df_b["first_bleed_time"].iloc[0] if "first_bleed_time" in df_b.columns else None

        _plot_one_birth(
            birth_df=df_b,
            first_bleed_time=first_bleed_time,
            hgb_col=cfg["hgb_col"],
            hct_col=cfg["hct_col"],
            out_path=os.path.join(cfg["out_dir"], out_name),
            cfg=cfg
        )
        n_plotted += 1

    print(f"[OK] Plots written to: {cfg['out_dir']}")
    print(f"Sampled births: {len(pos_sample)} positives, {len(neg_sample)} negatives.")
    print(f"Births plotted: {n_plotted}")

if __name__ == "__main__":
    main()
