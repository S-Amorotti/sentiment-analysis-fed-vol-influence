# src/models/train_rf.py
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_ml_experiment(X, y, param_grid=None, random_state=42, n_jobs=-1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=random_state)

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    importances = best_model.feature_importances_

    return {
        "best_params": grid_search.best_params_,
        "mse": mse,
        "r2": r2,
        "model": best_model,
        "importances": importances,
        "X_train_cols": X.columns,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_rf_model(cfg: Config):
    cfg.ensure_dirs()
    logger.info(f"Loading ML data from {cfg.ml_data_csv}")
    df = pd.read_csv(cfg.ml_data_csv)

    market_features = ["Close", "High", "Low", "Open", "Volume"]
    vader_features = ["vader_compound", "vader_positive", "vader_negative"]
    finbert_features = [
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "net_finbert_sentiment",
    ]
    rolling_features = [
        "DailyVol_rolling_mean_3",
        "DailyVol_rolling_std_3",
        "Close_rolling_mean_3",
        "Close_rolling_std_3",
    ]

    target = "Daily Volatility"

    df_clean = df.dropna(subset=market_features + vader_features + finbert_features + [target])

    X_combined_rolling = df_clean[vader_features + finbert_features + market_features + rolling_features]
    y = df_clean[target]

    logger.info(f"Training RF model on {X_combined_rolling.shape[0]} samples, {X_combined_rolling.shape[1]} features.")
    results = run_ml_experiment(
        X_combined_rolling, y, random_state=cfg.random_state, n_jobs=cfg.n_jobs
    )

    logger.info(f"Best Params: {results['best_params']}")
    logger.info(f"MSE: {results['mse']:.6f}")
    logger.info(f"R²: {results['r2']:.4f}")

    model_obj = {
        "model": results["model"],
        "feature_cols": list(X_combined_rolling.columns),
        "scaler": None,         # If you want to add it later
        "metadata": {
            "best_params": results["best_params"],
            "mse": results["mse"],
            "r2": results["r2"],
        }
    }
    with open(cfg.model_path, "wb") as f:
        pickle.dump(model_obj, f)
    logger.info(f"Saved model & metadata to {cfg.model_path}")

    return results


if __name__ == "__main__":
    cfg = Config()
    train_rf_model(cfg)
