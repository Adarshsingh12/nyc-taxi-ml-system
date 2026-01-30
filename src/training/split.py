from pathlib import Path
import pandas as pd


FEATURES_PATH = Path("data/processed/features.parquet")


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Feature data not found. Run feature engineering first."
        )

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15
):
    if train_frac + val_frac >= 1.0:
        raise ValueError("Train + validation fraction must be < 1.0")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def run_split():
    df = load_features()
    train_df, val_df, test_df = time_based_split(df)

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print(
        f"Train dates: {train_df.date.min()} → {train_df.date.max()}"
    )
    print(
        f"Validation dates: {val_df.date.min()} → {val_df.date.max()}"
    )
    print(
        f"Test dates: {test_df.date.min()} → {test_df.date.max()}"
    )


if __name__ == "__main__":
    run_split()
