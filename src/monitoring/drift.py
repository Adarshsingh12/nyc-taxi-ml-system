from pathlib import Path
import pandas as pd
import numpy as np


FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_PATH = Path("monitoring/drift_report.csv")

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_7d_mean",
    "rolling_14d_mean",
]


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature data not found")

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10
) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = expected.quantile(quantiles).values

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    psi = np.sum(
        (expected_pct - actual_pct)
        * np.log((expected_pct + 1e-6) / (actual_pct + 1e-6))
    )

    return psi


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame
) -> pd.DataFrame:
    records = []

    for col in FEATURE_COLUMNS:
        psi = population_stability_index(
            reference_df[col],
            current_df[col]
        )

        mean_ref = reference_df[col].mean()
        mean_cur = current_df[col].mean()

        records.append({
            "feature": col,
            "psi": psi,
            "mean_reference": mean_ref,
            "mean_current": mean_cur,
            "mean_shift": mean_cur - mean_ref
        })

    return pd.DataFrame(records)


def run_drift_detection(
    reference_end_date: str,
    current_start_date: str
):
    df = load_features()

    reference_df = df[df["date"] <= reference_end_date]
    current_df = df[df["date"] >= current_start_date]

    if reference_df.empty or current_df.empty:
        raise ValueError("Insufficient data for drift detection")

    drift_df = compute_drift(reference_df, current_df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_csv(REPORT_PATH, index=False)

    print(f"Drift report saved to {REPORT_PATH}")
    print(drift_df)


if __name__ == "__main__":
    # Example dates — adjust as needed
    run_drift_detection(
        reference_end_date="2022-04-30",
        current_start_date="2022-05-01"
    )
