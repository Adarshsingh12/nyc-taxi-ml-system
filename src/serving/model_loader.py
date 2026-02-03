import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "nyc-taxi-demand"


def load_best_model(metric: str = "rmse"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError("MLflow experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"]
    )

    for run in runs:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded model from run {run_id}")
            return model
        except Exception:
            # No model logged for this run → skip
            continue

    raise ValueError("No valid model run found")
