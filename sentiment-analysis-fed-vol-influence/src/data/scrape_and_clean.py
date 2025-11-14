# src/data/scrape_and_clean.py
import os
import re
from datetime import datetime
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_document_links(base_url: str):
    logger.info(f"Fetching FOMC PDF links from {base_url}")
    response = requests.get(base_url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch page: {base_url} (status={response.status_code})")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if (
            "monetarypolicy/files" in href
            and href.endswith(".pdf")
            and ("monetary" in href or "fomcminutes" in href or "projtabl" in href)
        ):
            full_url = urljoin(base_url, href)
            links.append(full_url)

    logger.info(f"Found {len(links)} candidate PDFs.")
    return links


def download_documents(links, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for link in links:
        try:
            file_name = os.path.basename(link)
            file_path = raw_dir / file_name
            if file_path.exists():
                logger.info(f"File already exists, skipping: {file_name}")
                continue

            resp = requests.get(link)
            if resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded: {file_name}")
            else:
                logger.warning(f"Failed to download {link} (status={resp.status_code})")
        except Exception as e:
            logger.exception(f"Error downloading {link}: {e}")


def extract_date_from_filename(filename: str):
    base = os.path.basename(filename)
    match = re.search(r"(\d{8})", base)
    if match:
        date_str = match.group(1)  # 'YYYYMMDD'
        try:
            return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            return None
    return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
        return text
    except Exception as e:
        logger.exception(f"Error extracting text from {pdf_path}: {e}")
        return ""


def remove_useless_lines(lines):
    cleaned = []
    for line in lines:
        line = line.replace("_", " ")
        line = " ".join(line.split())
        if re.match(r"^\s*Page\s*\d+\s*$", line, re.IGNORECASE):
            continue
        if len(line.strip()) < 3:
            continue
        cleaned.append(line)
    return cleaned


def clean_statements_text(text: str) -> str:
    lines = text.split("\n")
    lines = remove_useless_lines(lines)

    release_pattern = re.compile(
        r"For release at.*\d{4},?\s*-?\s*\d+\s*-", re.IGNORECASE
    )
    filtered = []
    for line in lines:
        if release_pattern.search(line):
            continue
        filtered.append(line)
    final_text = " ".join(filtered)
    return final_text.strip()


def clean_minutes_text(text: str) -> str:
    lines = text.split("\n")
    lines = remove_useless_lines(lines)

    financial_terms = ["federal reserve", "federal open market committee", "monetary policy"]
    start_index = 0
    found_start = False
    for i, line in enumerate(lines):
        check_line = line.lower()
        if any(term in check_line for term in financial_terms):
            start_index = i
            found_start = True
            break

    if found_start:
        lines = lines[start_index:]

    final_text = " ".join(lines)
    return final_text.strip()


def clean_text_by_type(text: str, doc_type: str) -> str:
    if doc_type == "minutes":
        return clean_minutes_text(text)
    elif doc_type == "statements":
        return clean_statements_text(text)
    return text


def save_cleaned_text(text: str, file_name: str, doc_type: str, cleaned_dir: Path):
    type_dir = cleaned_dir / doc_type
    type_dir.mkdir(parents=True, exist_ok=True)
    doc_date = extract_date_from_filename(file_name)
    if doc_date:
        out_name = f"{doc_date}_{file_name.replace('.pdf','')}.txt"
    else:
        out_name = file_name.replace(".pdf", ".txt")
    output_path = type_dir / out_name
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved cleaned text to {output_path}")


def process_documents(raw_dir: Path, cleaned_dir: Path):
    logger.info(f"Processing raw PDFs from {raw_dir}")
    for file_name in os.listdir(raw_dir):
        file_path = raw_dir / file_name

        if "projtabl" in file_name.lower():
            logger.info(f"Skipping projtabl document: {file_name}")
            continue

        doc_type = "unknown"
        if "minutes" in file_name.lower():
            doc_type = "minutes"
        elif "monetary" in file_name.lower():
            doc_type = "statements"

        if doc_type in ["minutes", "statements"]:
            text = extract_text_from_pdf(file_path)
            cleaned_text = clean_text_by_type(text, doc_type)
            save_cleaned_text(cleaned_text, file_name, doc_type, cleaned_dir)
        else:
            logger.info(f"Unknown document type (skipping): {file_name}")


def run_scrape_and_clean(cfg: Config):
    cfg.ensure_dirs()
    links = get_document_links(cfg.fed_calendar_url)
    download_documents(links, cfg.raw_dir)
    process_documents(cfg.raw_dir, cfg.cleaned_dir)


if __name__ == "__main__":
    cfg = Config()
    run_scrape_and_clean(cfg)
