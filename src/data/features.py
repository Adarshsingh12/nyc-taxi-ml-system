from pathlib import Path
import pandas as pd


PROCESSED_DATA_PATH = Path("data/processed/daily_demand.parquet")
FEATURES_OUTPUT_PATH = Path("data/processed/features.parquet")


def load_daily_demand() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing first."
        )

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pickup_zone", "date"])

    df["lag_1"] = (
        df.groupby("pickup_zone")["trip_count"]
        .shift(1)
    )

    df["lag_7"] = (
        df.groupby("pickup_zone")["trip_count"]
        .shift(7)
    )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pickup_zone", "date"])

    shifted = (
        df.groupby("pickup_zone")["trip_count"]
        .shift(1)
    )

    df["rolling_7d_mean"] = (
        shifted
        .groupby(df["pickup_zone"])
        .rolling(window=7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_14d_mean"] = (
        shifted
        .groupby(df["pickup_zone"])
        .rolling(window=14)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def build_features() -> pd.DataFrame:
    df = load_daily_demand()

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop rows with insufficient history
    df = df.dropna().reset_index(drop=True)

    return df


def save_features(df: pd.DataFrame) -> None:
    FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_OUTPUT_PATH, index=False)


def run_feature_engineering() -> None:
    print("Building features...")
    features_df = build_features()

    print(f"Saving features to {FEATURES_OUTPUT_PATH}")
    save_features(features_df)


if __name__ == "__main__":
    run_feature_engineering()
