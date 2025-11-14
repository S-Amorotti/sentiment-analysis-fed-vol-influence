# src/sentiment/sentiment_pipeline.py
import glob
import os
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.config import Config
from src.data.scrape_and_clean import extract_date_from_filename
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def init_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def init_finbert():
    finbert_model_name = "yiyanghkust/finbert-tone"
    logger.info(f"Loading FinBERT model: {finbert_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
    tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def get_vader_scores(text, analyzer):
    scores = analyzer.polarity_scores(text)
    return scores["compound"], scores["pos"], scores["neg"]


def get_finbert_scores(text, pipe):
    snippet = text[:512] if isinstance(text, str) else ""
    if not snippet.strip():
        return 0.0, 0.0, 0.0
    result = pipe(snippet)[0]
    label = result["label"]
    score = result["score"]

    pos = neg = neu = 0.0
    if label == "Positive":
        pos = score
    elif label == "Negative":
        neg = score
    elif label == "Neutral":
        neu = score
    return pos, neg, neu


def run_sentiment(cfg: Config):
    cfg.ensure_dirs()
    cleaned_files = glob.glob(os.path.join(cfg.cleaned_dir.as_posix(), "*", "*.txt"))
    logger.info(f"Found {len(cleaned_files)} cleaned text files.")

    vader = init_vader()
    finbert_pipe = init_finbert()

    sentiment_records = []

    for f in cleaned_files:
        doc_date = extract_date_from_filename(f)
        if doc_date is None:
            logger.warning(f"No date found in filename, skipping: {f}")
            continue

        with open(f, "r", encoding="utf-8") as file:
            text = file.read()

        vader_compound, vader_pos, vader_neg = get_vader_scores(text, vader)
        finb_pos, finb_neg, finb_neu = get_finbert_scores(text, finbert_pipe)

        sentiment_records.append(
            {
                "Date": doc_date,
                "vader_compound": vader_compound,
                "vader_positive": vader_pos,
                "vader_negative": vader_neg,
                "finbert_positive": finb_pos,
                "finbert_negative": finb_neg,
                "finbert_neutral": finb_neu,
            }
        )

    if not sentiment_records:
        logger.error("No sentiment records generated. Check cleaned files and date extraction.")
        return

    df = pd.DataFrame(sentiment_records)
    df = df.groupby("Date", as_index=False).mean()
    df["Date"] = pd.to_datetime(df["Date"])
    df["net_finbert_sentiment"] = df["finbert_positive"] - df["finbert_negative"]

    cfg.intermediate_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.sentiment_csv, index=False)
    logger.info(f"Saved sentiment data to {cfg.sentiment_csv}")


if __name__ == "__main__":
    cfg = Config()
    run_sentiment(cfg)
