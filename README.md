# Central Bank Sentiment Analysis and Market Volatility Prediction
Using FinBERT, VADER and Market Features to Predict S&P 500 Volatility

This project builds a complete and reproducible machine learning pipeline that:

- Scrapes Federal Reserve FOMC PDF documents
- Cleans and preprocesses minutes and statements
- Extracts sentiment using VADER and FinBERT
- Downloads S&P 500 market data
- Builds a feature-rich machine learning dataset
- Trains a Random Forest model
- Evaluates model performance and produces plots

The project is fully modular and every step can be run independently or as a complete pipeline.

## Project Overview

Central bank communications, especially FOMC minutes and monetary policy statements, have a measurable effect on financial markets.
This project analyzes the tone of these communications and evaluates how sentiment relates to market volatility measured on the S&P 500.

We use the following components:

- VADER sentiment analysis
- FinBERT transformer model for financial sentiment
- Rolling market statistics
- Random Forest Regression to predict volatility

## Project Structure
```
sentiment-analysis-fed-vol-influence/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── config.py
    ├── utils/
    │   └── logging_utils.py
    ├── data/
    │   └── scrape_and_clean.py
    ├── sentiment/
    │   └── sentiment_pipeline.py
    ├── market/
    │   └── market_data.py
    ├── features/
    │   └── build_ml_dataset.py
    ├── models/
    │   └── train_rf.py
    └── evaluation/
        └── evaluation.py
│
└── data/
    ├── fed_documents/
    │   ├── raw/
    │   └── cleaned/
    ├── intermediate/
    ├── figures/
    └── models/
```

Each module can be executed independently using:
```
python -m src.<module>
```
Installation:
```
git clone <repo-url>
cd sentiment-analysis-fed-vol-influence

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Usage
Run the entire pipeline

```
python main.py
```

Run individual steps
Scrape and clean FOMC PDFs
```
python main.py --scrape
```
Sentiment analysis (VADER + FinBERT)
```
python main.py --sentiment
```
Download S&P 500 market data
```
python main.py --market
```
Build machine learning dataset
```
python main.py --features
```
Train Random Forest model
```
python main.py --train
```
Evaluate model and generate plots
```
python main.py --evaluate
```

Plots are saved in data/figures/.

## Methodology
###1. Scraping Federal Reserve PDFs

Documents are pulled from:
- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm


The scraper identifies:
- Monetary policy statements
- FOMC minutes
- Excludes projection tables

Downloaded files are stored in:
- data/fed_documents/raw/

### 2. Text Extraction and Cleaning

Using pdfplumber, the pipeline extracts raw text.
Cleaning includes:
- Removing headers and footers
- Removing page numbers
- Removing "For release at..." text
- Normalizing whitespace

Cleaned text is stored in:
- data/fed_documents/cleaned/statements/
- data/fed_documents/cleaned/minutes/

Files are renamed with their date prepended:
- YYYY-MM-DD_filename.txt

### 3. Sentiment Extraction

Two sentiment systems are used:
#### VADER

Outputs:
- compound
- positive
- negative

#### FinBERT

Outputs:
- finbert_positive
- finbert_negative
- finbert_neutral
- net_finbert_sentiment (positive minus negative)

Result stored at:
- data/intermediate/sentiment_data.csv

### 4. Market Data

The pipeline downloads S&P 500 data using yfinance.

Daily volatility is computed as:
- (High - Low) / Open


Market data is saved to:
- data/intermediate/market_data_from_2019.csv

### 5. Feature Engineering

The dataset merges:
- Sentiment features
- Market features
- Rolling windows (3-day mean and std)

Final ML dataset:
- data/intermediate/ml_data.csv

### 6. Machine Learning

A RandomForestRegressor is trained with hyperparameter tuning using GridSearchCV.

Input features include:
- All FinBERT sentiment scores
- All VADER scores
- OHLCV market data
- Rolling volatility and price features

Target variable:
Daily Volatility

The trained model is saved to:
- models/rf_volatility_model.pkl

### 7. Evaluation

Using the evaluation module, the following plots are produced:

#### Predicted vs Actual volatility
Saved to:
- figures/pred_vs_actual.png

#### Feature importances
Saved to:
- figures/feature_importances.png


These plots help assess model performance and interpret feature influence.
