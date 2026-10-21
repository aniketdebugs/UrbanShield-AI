"""Train model using a real, quality-checked merged dataset."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "indore_heat.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "air_temp_c", "humidity", "wind_speed", "ndvi", "ndbi",
    "green_fraction", "built_fraction", "water_fraction",
    "latitude", "longitude"
]
TARGET = "lst_c"

def split_groups(df, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(splitter.split(df, groups=df["cell_id"]))
    return df.iloc[a].copy(), df.iloc[b].copy()

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Export real Landsat data, build and validate "
            "data/processed/indore_heat.csv first. See README.md."
        )
    df = pd.read_csv(DATA, parse_dates=["date"])
    required = FEATURES + [TARGET, "cell_id", "date"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=required).copy()
    if df["cell_id"].nunique() < 10 or len(df) < 100:
        raise ValueError("Need adequate data coverage; aim for >=100 rows and >=10 cells.")
    train, test = split_groups(df, 0.25, 42)
    fit, calibration = split_groups(train, 0.25, 43)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
            l2_regularization=1.0, random_state=42
        )
    }
    # Choose model using calibration partition as a validation set; final
    # results are reported on untouched grouped test cells.
    scores = {}
    fitted = {}
    for name, model in models.items():
        model.fit(fit[FEATURES], fit[TARGET])
        pred = model.predict(calibration[FEATURES])
        scores[name] = {
            "MAE": float(mean_absolute_error(calibration[TARGET], pred)),
            "RMSE": float(np.sqrt(mean_squared_error(calibration[TARGET], pred))),
            "R2": float(r2_score(calibration[TARGET], pred))
        }
        fitted[name] = model

    best_name = min(scores, key=lambda name: scores[name]["MAE"])
    model = fitted[best_name]
    cal_pred = model.predict(calibration[FEATURES])
    residuals = np.abs(calibration[TARGET].to_numpy() - cal_pred)
    alpha = 0.10
    rank = min(len(residuals), int(np.ceil((len(residuals)+1)*(1-alpha))))
    q = float(np.sort(residuals)[rank-1])

    pred = model.predict(test[FEATURES])
    y = test[TARGET].to_numpy()
    metrics = {
        "data_mode": "REAL_DATA",
        "selected_model": best_name,
        "model_comparison": scores,
        "test_MAE": float(mean_absolute_error(y, pred)),
        "test_RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "test_R2": float(r2_score(y, pred)),
        "nominal_interval_coverage": 0.90,
        "test_interval_coverage": float(np.mean((y >= pred-q) & (y <= pred+q))),
        "mean_interval_width": float(2*q),
        "conformal_q": q,
        "feature_columns": FEATURES,
        "target": TARGET,
        "n_rows": int(len(df)),
        "n_cells": int(df["cell_id"].nunique())
    }
    joblib.dump(model, MODEL_DIR / "heat_model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
