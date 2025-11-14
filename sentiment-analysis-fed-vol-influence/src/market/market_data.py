# src/market/market_data.py
import pandas as pd
import yfinance as yf

from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_market_data(cfg: Config) -> pd.DataFrame:
    logger.info(f"Downloading {cfg.index_ticker} from {cfg.market_start_date} to {cfg.market_end_date}")
    sp500 = yf.download(cfg.index_ticker, start=cfg.market_start_date, end=cfg.market_end_date)

    if sp500.empty:
        logger.error("No market data returned from Yahoo Finance.")
        return pd.DataFrame()

    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = [col[0] for col in sp500.columns]

    sp500.reset_index(inplace=True)

    required_cols = {"High", "Low", "Open"}
    if required_cols.issubset(sp500.columns):
        sp500["Daily Volatility"] = (sp500["High"] - sp500["Low"]) / sp500["Open"]
    else:
        logger.error(f"Could not compute Daily Volatility, missing {required_cols - set(sp500.columns)}")

    return sp500


def run_market_pipeline(cfg: Config):
    cfg.ensure_dirs()
    df = get_market_data(cfg)
    if df.empty:
        return
    df.to_csv(cfg.market_csv, index=False)
    logger.info(f"Market data saved to {cfg.market_csv}")


if __name__ == "__main__":
    cfg = Config()
    run_market_pipeline(cfg)
