# build_labels_from_drugs.py
# Efficient PPH labels builder using ExecutionDate as the relevant time column.
# Outputs:
#   1) D:/PPH/processed/labels_by_pregnancy.csv
#      One row per (hashed_mother_id, pregnancy_index) with counts and label
#   2) D:/PPH/processed/positive_execution_dates.csv
#      One row per positive (label>0) with all ExecutionDate timestamps of blood-related events

from __future__ import annotations
import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = r"D:/PPH"
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Adjust this filename if your CSV is named differently (kept close to your example)
DRUGS_CSV = os.path.join(RAW_DIR, "MF_mother_drugs_20250812.csv")

# 180-day gap to split episodes/pregnancies
GAP_DAYS = 180


# =============================================================================
# HELPERS
# =============================================================================
def normalize_drug_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw DrugName values to two buckets:
      - 'blood dose'     (e.g., RBC, PRBC, 'red blood cells')
      - 'blood product'  (e.g., plasma, platelets)
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Build a lowercase working column for robust matching
    name = df["DrugName"].astype(str).str.lower()

    # Flags (vectorized; avoids per-row Python)
    is_product = (
        name.str.contains("plasma", na=False)
        | name.str.contains("platelet", na=False)
    )
    is_dose = (
        name.str.contains("red blood cells", na=False)
        | name.str.contains("rbc", na=False)
        | name.str.contains("prbc", na=False)
    )

    df["blood dose"] = is_dose.astype("int8")
    df["blood product"] = is_product.astype("int8")
    return df


def split_into_pregnancies(df: pd.DataFrame, gap_days: int = GAP_DAYS) -> pd.DataFrame:
    """
    For each mother, split their events into episodes separated by > gap_days
    based on ExecutionDate. Assign 'pregnancy_index' (1-based).
    """
    df = df.sort_values(["hashed_mother_id", "ExecutionDate"]).copy()
    # Mark new period when gap > threshold
    new_period = (
        df.groupby("hashed_mother_id")["ExecutionDate"].diff() > pd.Timedelta(days=gap_days)
    ).astype("int8")

    # Cumulative sum within mother -> period_group
    df["period_group"] = df.groupby("hashed_mother_id")["ExecutionDate"].apply(
        lambda s: (s.diff() > pd.Timedelta(days=gap_days)).cumsum()
    ).astype("int32")

    # Convert to pregnancy_index per mother (stable ordering by time)
    df["pregnancy_index"] = (
        df.groupby("hashed_mother_id")["period_group"].rank(method="dense").astype("int32")
    )

    # A tighter integer pregnancy index (1..K per mother) via factorize per mother:
    # (Factorize guarantees consecutive ints starting at 0, then add 1)
    df["pregnancy_index"] = (
        df.groupby("hashed_mother_id")["period_group"].transform(
            lambda x: pd.factorize(x, sort=True)[0] + 1
        ).astype("int32")
    )
    return df.drop(columns=["period_group"])


def aggregate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate counts per (mother, pregnancy_index) and compute label.
    Label logic (same as your original rule-set):
      label = 0 if blood dose == 0
      label = 1 if blood dose in {1,2}  (unless special case below)
      label = 2 if blood dose > 2 OR (blood dose == 2 and blood product > 0)
    """
    agg = (
        df.groupby(["hashed_mother_id", "pregnancy_index"], as_index=False)
        .agg({
            "blood dose": "sum",
            "blood product": "sum",
        })
    )
    # Convert to small ints
    agg["blood dose"] = agg["blood dose"].astype("int32")
    agg["blood product"] = agg["blood product"].astype("int32")

    conditions = [
        (agg["blood dose"] == 0),
        (agg["blood dose"].between(1, 2)),
        ((agg["blood dose"] > 2) | ((agg["blood dose"] == 2) & (agg["blood product"] > 0))),
    ]
    choices = [0, 1, 2]
    agg["label"] = np.select(conditions, choices, default=0).astype("int8")

    # Order columns
    agg = agg[["hashed_mother_id", "pregnancy_index", "blood dose", "blood product", "label"]]
    return agg


def collect_positive_execution_dates(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    For label>0 episodes, collect the exact ExecutionDate timestamps of any
    blood-related event (dose OR product). Also provide the first positive time.
    """
    # Any blood-related event row
    any_blood = df[(df["blood dose"] == 1) | (df["blood product"] == 1)]

    times = (
        any_blood.groupby(["hashed_mother_id", "pregnancy_index"])["ExecutionDate"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .reset_index(name="execution_dates")
    )

    # Merge with labels, keep only positives
    pos = labels_df.merge(times, on=["hashed_mother_id", "pregnancy_index"], how="left")
    pos = pos[pos["label"] > 0].copy()

    # First positive timestamp per episode
    pos["first_positive_execution"] = pos["execution_dates"].apply(
        lambda lst: (pd.Timestamp(lst[0]) if isinstance(lst, list) and len(lst) else pd.NaT)
    )

    # Serialize the list to ISO strings for CSV readability
    pos["execution_dates_iso"] = pos["execution_dates"].apply(
        lambda lst: ";".join(pd.Series(lst, dtype="datetime64[ns]").astype("datetime64[ns]").astype(str))
        if isinstance(lst, list) and len(lst) else ""
    )
    pos = pos.drop(columns=["execution_dates"])

    # Reorder
    pos = pos[
        ["hashed_mother_id", "pregnancy_index", "label",
         "first_positive_execution", "execution_dates_iso"]
    ]
    return pos


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ---------------------------
    # Load
    # ---------------------------
    usecols = ["hashed_mother_id", "DrugName", "ExecutionDate"]
    df = pd.read_csv(DRUGS_CSV, usecols=usecols)

    # ---------------------------
    # Parse time & clean
    # ---------------------------
    df["ExecutionDate"] = pd.to_datetime(df["ExecutionDate"], errors="coerce")
    df = df.dropna(subset=["hashed_mother_id", "ExecutionDate"]).copy()
    df["hashed_mother_id"] = df["hashed_mother_id"].astype(str)

    # ---------------------------
    # Normalize names -> flags
    # ---------------------------
    df = normalize_drug_names(df)

    # Quick short-circuit: keep only rows that could matter
    df = df[(df["blood dose"] == 1) | (df["blood product"] == 1)].copy()
    if df.empty:
        # Still write empty outputs to keep pipeline predictable
        labels_out = os.path.join(OUT_DIR, "labels_by_pregnancy.csv")
        pos_out = os.path.join(OUT_DIR, "positive_execution_dates.csv")
        pd.DataFrame(columns=["hashed_mother_id", "pregnancy_index", "blood dose", "blood product", "label"])\
            .to_csv(labels_out, index=False)
        pd.DataFrame(columns=["hashed_mother_id", "pregnancy_index", "label",
                              "first_positive_execution", "execution_dates_iso"])\
            .to_csv(pos_out, index=False)
        print("No blood-related drug rows found. Wrote empty outputs.")
        return

    # ---------------------------
    # Split into pregnancies by 180d gaps (per mother)
    # ---------------------------
    df = split_into_pregnancies(df, gap_days=GAP_DAYS)

    # ---------------------------
    # Aggregate + label per (mother, pregnancy_index)
    # ---------------------------
    labels_df = aggregate_labels(df)

    # ---------------------------
    # Collect exact ExecutionDate values for positives
    # ---------------------------
    positive_times = collect_positive_execution_dates(df, labels_df)

    # ---------------------------
    # Save
    # ---------------------------
    labels_out = os.path.join(OUT_DIR, "labels_by_pregnancy.csv")
    pos_out = os.path.join(OUT_DIR, "positive_execution_dates.csv")
    labels_df.to_csv(labels_out, index=False)
    positive_times.to_csv(pos_out, index=False)

    # ---------------------------
    # Print quick stats
    # ---------------------------
    print("Saved:", labels_out)
    print("Saved:", pos_out)
    counts = labels_df["label"].value_counts().sort_index()
    print("\nLabel counts (0/1/2):")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
