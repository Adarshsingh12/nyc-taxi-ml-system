import os
import requests
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"[SKIP] {output_path.name} already exists")
        return

    print(f"[DOWNLOAD] {output_path.name}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)


def run_ingestion(config_path: str = "configs/data.yaml") -> None:
    config = load_config(config_path)

    base_url = config["base_url"]
    months = config["months"]
    raw_dir = Path(config["raw_data_dir"])

    for month in months:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{base_url}/{filename}"
        output_path = raw_dir / filename

        download_file(url, output_path)


if __name__ == "__main__":
    run_ingestion()
