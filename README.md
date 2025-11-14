# FED NLP Trading – Monetary Policy Text to Market Reaction

This project builds an end-to-end machine learning pipeline that:

1. Scrapes Federal Reserve FOMC-related PDF documents.
2. Extracts and cleans the text.
3. Computes sentiment using:
   - VADER
   - FinBERT (`yiyanghkust/finbert-tone`)
4. Merges sentiment with simple market reaction features from `yfinance`.
5. Trains a Random Forest regression model to predict short-term returns around policy events.
6. Produces basic evaluation plots.

## Project Structure

```text
.
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py
    ├── utils/
    ├── data/
    ├── preprocessing/
    ├── sentiment/
    ├── features/
    ├── models/
    └── evaluation/
```

## Installation

```
git clone <your-repo-url>
cd fed-nlp-trading

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```
python main.py
```

Or step by step:

```
python main.py --scrape --preprocess
python main.py --sentiment
python main.py --features
python main.py --train
python main.py --evaluate
```