# main.py
import argparse

from src.config import Config
from src.utils.logging_utils import get_logger
from src.data.scrape_and_clean import run_scrape_and_clean
from src.sentiment.sentiment_pipeline import run_sentiment
from src.market.market_data import run_market_pipeline
from src.features.build_ml_dataset import build_ml_dataset
from src.models.train_rf import train_rf_model
from src.evaluation.evaluation import evaluate_and_plot


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: scrape → clean → sentiment → market → features → ML"
    )
    parser.add_argument("--scrape", action="store_true", help="Scrape & clean Fed PDFs.")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis.")
    parser.add_argument("--market", action="store_true", help="Download market data.")
    parser.add_argument("--features", action="store_true", help="Build ML dataset.")
    parser.add_argument("--train", action="store_true", help="Train RF model.")
    parser.add_argument("--evaluate", action="store_true", help="Run model evaluation plots.")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # If nothing specified: run full pipeline
    if not any([args.scrape, args.sentiment, args.market, args.features, args.train, args.evaluate]):
        logger.info("No step specified, running full pipeline.")
        args.scrape = args.sentiment = args.market = args.features = args.train = args.evaluate = True

    if args.scrape:
        logger.info("Step 1: Scrape + clean Fed documents.")
        run_scrape_and_clean(cfg)

    if args.sentiment:
        logger.info("Step 2: Run sentiment analysis.")
        run_sentiment(cfg)

    if args.market:
        logger.info("Step 3: Download market data.")
        run_market_pipeline(cfg)

    if args.features:
        logger.info("Step 4: Build ML dataset.")
        build_ml_dataset(cfg)

    if args.train:
        logger.info("Step 5: Train Random Forest model.")
        train_rf_model(cfg)

    if args.evaluate:
        logger.info("Step 6: Evaluate the model & generate plots.")
        evaluate_and_plot(cfg)

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
