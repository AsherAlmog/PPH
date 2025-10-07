import pandas as pd
import os


# Step 1: Utility functions for file handling
def read_csv_file(path: str, n_rows: int = None) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, nrows=n_rows, encoding="utf-8")


def write_excel_file(df: pd.DataFrame, path: str):
    """Writes a pandas DataFrame to an Excel file."""
    df.to_excel(path, index=False)


# Step 2: Data Cleaning and Preprocessing
def clean_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by converting time and numeric fields,
    then pivots measurements into their respective columns.
    """
    # Convert time and numeric fields
    df['parameter_time'] = pd.to_datetime(df['parameter_time'])
    df['ResultNumeric'] = pd.to_numeric(df['ResultNumeric'], errors='coerce')

    # Pivot to create meaningful measurement columns
    df_pivot = df.pivot_table(
        index=['hashed_mother_id', 'parameter_time'],
        columns='Parameter_Name',
        values='ResultNumeric',
        aggfunc='first'
    ).reset_index()

    # Handle specific measurement columns (e.g., systolic and diastolic pressures)
    df_pivot['diastol'] = df_pivot['לחץ דיאסטולי'].fillna(df_pivot['לחץ דם. דיאסטולי'])
    df_pivot['sistol'] = df_pivot['לחץ סיסטולי'].fillna(df_pivot['לחץ דם. סיסטולי'])
    df_pivot.drop(columns=['לחץ דיאסטולי', 'לחץ דם. דיאסטולי', 'לחץ סיסטולי', 'לחץ דם. סיסטולי'], inplace=True)

    return df_pivot


# Step 3: Aligning Parameters by Absolute Time Difference
def align_by_abs_time_diff(
    df: pd.DataFrame,
    id_field: str,
    time_field: str,
    source_params: list,
    target_params: list
) -> pd.DataFrame:
    """
    Aligns source parameters to target parameters based on the absolute time difference,
    ensuring no self-matching of rows.
    """
    # Separate data into source and target fields
    source_grid = df[source_params + [id_field, time_field]].copy()
    target_grid = df[target_params + [id_field, time_field]].copy()

    aligned_data = []

    # Loop over each unique ID
    for unique_id in df[id_field].unique():
        source_rows = source_grid[source_grid[id_field] == unique_id]
        target_rows = target_grid[target_grid[id_field] == unique_id]

        # Align each source row to the closest target row
        for source_idx, source_row in source_rows.iterrows():
            # Compute absolute time differences
            time_diffs = abs(target_rows[time_field] - source_row[time_field])
            time_diffs = time_diffs.mask(time_diffs.index == source_idx)  # Exclude self-match

            # Skip if no valid target rows available
            if time_diffs.isna().all():
                continue

            # Find the closest target row
            closest_row_idx = time_diffs.idxmin()

            # Combine source and matched target rows
            aligned_row = {
                **source_row.to_dict(),
                **{col: target_rows.loc[closest_row_idx, col] for col in target_params},
            }
            aligned_data.append(aligned_row)

    # Create a new DataFrame from aligned rows
    aligned_df = pd.DataFrame(aligned_data)

    # Drop rows without valid target parameter values
    aligned_df = aligned_df.dropna(subset=target_params, how='all').reset_index(drop=True)

    # Reorder columns: ID first, time second
    column_order = [id_field, time_field] + [col for col in aligned_df.columns if col not in [id_field, time_field]]
    return aligned_df[column_order]


# Step 4: Main Program
if __name__ == '__main__':
    # Define input/output file paths
    base_path = r"D:\PPH"
    input_file_path = os.path.join(base_path, "measurements.csv")
    output_file_path = os.path.join(base_path, "pph_wide_timeline_part01_small.xlsx")

    # Load raw data
    df = pd.read_csv(input_file_path)
    write_excel_file(df, output_file_path)

    # Clean and preprocess data
    df_pivot = clean_and_process_data(df)

    # Define measurement groups
    primary_measurements = ['sistol', 'diastol', 'BP - Mean']
    secondary_measurements = ['דופק', 'חום', 'סטורציה']

    # Align primary and secondary measurements
    aligned_patient_data = align_by_abs_time_diff(
        df_pivot,
        id_field="hashed_mother_id",
        time_field="parameter_time",
        source_params=primary_measurements,
        target_params=secondary_measurements
    )

    # Write aligned data to an Excel file
    write_excel_file(aligned_patient_data, output_file_path)

    print(f"Aligned data has been written to: {output_file_path}")

    # "meas": os.path.join(base_dir, "measurements.csv"),
    # "labs": os.path.join(base_dir, "MF_mother_labs_20250812.csv"),
    # "drugs": os.path.join(base_dir, "MF_mother_drugs_20250812.csv"),
    # "births": os.path.join(base_dir, "MF_FETAL_TABL_20250812_132000.csv"),