from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw/yellow")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "daily_demand.parquet"


def load_raw_data() -> pd.DataFrame:
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            "No raw data found. Run ingestion first."
        )

    dfs = []
    for file in files:
        df = pd.read_parquet(
            file,
            columns=["tpep_pickup_datetime", "PULocationID"]
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def aggregate_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["date"] = df["pickup_datetime"].dt.date

    daily = (
        df
        .groupby(["date", "PULocationID"])
        .size()
        .reset_index(name="trip_count")
        .rename(columns={"PULocationID": "pickup_zone"})
        .sort_values(["date", "pickup_zone"])
        .reset_index(drop=True)
    )

    return daily


def save_processed_data(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)


def run_preprocessing() -> None:
    print("Loading raw data...")
    raw_df = load_raw_data()

    print("Aggregating daily demand...")
    daily_demand = aggregate_daily_demand(raw_df)

    print(f"Saving processed data to {OUTPUT_FILE}")
    save_processed_data(daily_demand)


if __name__ == "__main__":
    run_preprocessing()
