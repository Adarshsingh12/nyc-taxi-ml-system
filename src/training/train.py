from pathlib import Path
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.training.split import load_features, time_based_split


TARGET_COL = "trip_count"
NON_FEATURE_COLS = ["date", "pickup_zone", TARGET_COL]


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df[TARGET_COL]
    return X, y


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def run_training():
    df = load_features()
    train_df, val_df, _ = time_based_split(df)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    mlflow.set_experiment("nyc-taxi-demand")

    # ------------------
    # Baseline
    # ------------------
    with mlflow.start_run(run_name="baseline_lag1"):
        y_pred = val_df["lag_1"]
        rmse, mae = evaluate(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

    # ------------------
    # Linear Regression
    # ------------------
    with mlflow.start_run(run_name="linear_regression"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse, mae = evaluate(y_val, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

    # ------------------
    # XGBoost
    # ------------------
    with mlflow.start_run(run_name="xgboost"):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse, mae = evaluate(y_val, preds)

        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1
        })

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model"
        )


if __name__ == "__main__":
    run_training()
