# src/features/build_ml_dataset.py
import pandas as pd
from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_ml_dataset(cfg: Config):
    cfg.ensure_dirs()
    logger.info(f"Loading sentiment data from {cfg.sentiment_csv}")
    sentiment_data = pd.read_csv(cfg.sentiment_csv, parse_dates=["Date"])

    logger.info(f"Loading market data from {cfg.market_csv}")
    market_data = pd.read_csv(cfg.market_csv, parse_dates=["Date"])

    analysis = pd.merge(sentiment_data, market_data, how="left", on="Date")
    analysis["is_trading_day"] = analysis["Close"].notna()
    ml_data = analysis[analysis["is_trading_day"]].copy()

    ml_data = ml_data.sort_values("Date")
    ml_data["DailyVol_rolling_mean_3"] = ml_data["Daily Volatility"].rolling(window=3).mean()
    ml_data["DailyVol_rolling_std_3"] = ml_data["Daily Volatility"].rolling(window=3).std()
    ml_data["Close_rolling_mean_3"] = ml_data["Close"].rolling(window=3).mean()
    ml_data["Close_rolling_std_3"] = ml_data["Close"].rolling(window=3).std()

    ml_data = ml_data.dropna()
    ml_data.to_csv(cfg.ml_data_csv, index=False)
    logger.info(f"ML dataset saved to {cfg.ml_data_csv} with shape {ml_data.shape}")


if __name__ == "__main__":
    cfg = Config()
    build_ml_dataset(cfg)
