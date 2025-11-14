# src/config.py
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class Config:
    # Base dirs
    base_dir: Path = Path(".")
    data_dir: Path = base_dir / "data"

    # Fed docs
    raw_dir: Path = data_dir / "fed_documents" / "raw"
    cleaned_dir: Path = data_dir / "fed_documents" / "cleaned"

    # Intermediate
    intermediate_dir: Path = data_dir / "intermediate"
    sentiment_csv: Path = intermediate_dir / "sentiment_data.csv"
    market_csv: Path = intermediate_dir / "market_data_from_2019.csv"
    ml_data_csv: Path = intermediate_dir / "ml_data.csv"
    evaluation_csv: Path = intermediate_dir / "evaluation_ml_data.csv"
    features_csv: Path = ml_data_csv   # alias for compatibility

    # Figures / models
    figures_dir: Path = base_dir / "figures"
    models_dir: Path = base_dir / "models"
    model_path: Path = models_dir / "rf_volatility_model.pkl"

    # Scraping
    fed_calendar_url: str = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    # Market
    index_ticker: str = "^GSPC"
    market_start_date: str = "2019-01-01"
    market_end_date: str = datetime.today().strftime("%Y-%m-%d")

    # ML
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1

    def ensure_dirs(self):
        for d in [
            self.data_dir,
            self.raw_dir,
            self.cleaned_dir,
            self.intermediate_dir,
            self.figures_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
