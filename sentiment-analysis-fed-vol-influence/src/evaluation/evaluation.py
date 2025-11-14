# src/evaluation/evaluation.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_and_plot(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger.info("Loading ML dataset and trained model.")

    # Load features
    df = pd.read_csv(cfg.features_csv)
    logger.info(f"Loaded dataset with shape {df.shape} from {cfg.features_csv}")

    # Load model object
    with open(cfg.model_path, "rb") as f:
        obj = pickle.load(f)

    model = obj["model"]
    feature_cols = obj["feature_cols"]

    # Check existence
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        logger.error(f"Missing feature columns in dataset: {missing_feats}")
        raise ValueError("Feature mismatch between model and dataset")

    X = df[feature_cols]

    # CASE 1: Your older notebook predicted 'Daily Volatility'
    if "Daily Volatility" in df.columns:
        y = df["Daily Volatility"]
        y_label = "Daily Volatility"

    # CASE 2: Your old modular code predicted 'ret_after'
    elif "ret_after" in df.columns:
        y = df["ret_after"]
        y_label = "ret_after"

    else:
        raise ValueError("No target column ('Daily Volatility' or 'ret_after') found in dataset.")

    # Predict
    y_pred = model.predict(X)

    # Scatter plot: predicted vs actual
    plt.figure(figsize=(7, 5))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.xlabel(f"Actual {y_label}")
    plt.ylabel(f"Predicted {y_label}")
    plt.title(f"Predicted vs Actual {y_label}")
    plt.tight_layout()
    out_path = cfg.figures_dir / "pred_vs_actual.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved scatter plot to {out_path}")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(8, 6))
        plt.barh(feature_cols, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances (Random Forest)")
        plt.tight_layout()
        out_path = cfg.figures_dir / "feature_importances.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved feature importance plot to {out_path}")
    else:
        logger.info("Model has no feature_importances_, skipping FI plot.")


if __name__ == "__main__":
    cfg = Config()
    evaluate_and_plot(cfg)
