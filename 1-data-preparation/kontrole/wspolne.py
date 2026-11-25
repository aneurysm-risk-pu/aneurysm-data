"""
Wspólne ścieżki i funkcje dla kontroli etapu 1 (zadania #43–#46, Sprint 6).

Droga danych (sprawdzona na plikach 25.11.2026):
  data/raw/{kor,neuro}_merged_aggregated_1W_mean.csv          dane surowe
  -> data/interim/cleaned/{kor,neuro}_cleaning.csv            krok 1: scripts-lk/clean_full.py
  -> data/imputation-inputs/{kor,neuro}_shortend.csv          krok 2: scripts-lf/aneurysm_data_analysis (1).ipynb
  -> data/imputation-inputs/aneurysm_concatted.csv            krok 3: scripts-lf/aneurysm_data_merge.ipynb
  -> data/processed/aneurysm_sgkf_input.csv                   krok 4: 3-sgkf-split/przygotowanie/przygotuj_dane.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DANE = BASE / "data"
WYNIKI = Path(__file__).resolve().parent / "results"
KOHORTY = ("kor", "neuro")
ETYKIETA = {"kor": 0, "neuro": 1}
LIMIT_HISTORII = {"kor": 11, "neuro": 38}    # krok 2, komórka 122 notebooka


def surowe(k: str) -> pd.DataFrame:
    return pd.read_csv(DANE / "raw" / f"{k}_merged_aggregated_1W_mean.csv", dtype=str)


def po_kroku1(k: str) -> pd.DataFrame:
    return pd.read_csv(DANE / "interim" / "cleaned" / f"{k}_cleaning.csv", low_memory=False)


def po_kroku2(k: str) -> pd.DataFrame:
    return pd.read_csv(DANE / "imputation-inputs" / f"{k}_shortend.csv")


def po_kroku3() -> pd.DataFrame:
    return pd.read_csv(DANE / "imputation-inputs" / "aneurysm_concatted.csv")


def po_kroku4() -> pd.DataFrame:
    return pd.read_csv(DANE / "processed" / "aneurysm_sgkf_input.csv")


def cechy_lab(surowy: pd.DataFrame) -> list[str]:
    """40 kolumn wyników badań: te, które w danych surowych mają kolumnę `-unit`."""
    return [c for c in surowy.columns if f"{c}-unit" in surowy.columns]


def liczba(v) -> float:
    """Komórka danych surowych -> liczba, tak jak w kroku 1 (clean_full.py):
    {'values': [...]} -> średnia; jeśli któryś wpis nie jest liczbą (np. '<0.4'), cała komórka -> brak."""
    if not isinstance(v, str):
        return np.nan
    if v.startswith("{"):
        try:
            w = [float(x) for x in ast.literal_eval(v)["values"]]
            return float(np.mean(w)) if w else np.nan
        except (ValueError, TypeError, KeyError):
            return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def zapisz(nazwa: str, wynik: dict) -> None:
    WYNIKI.mkdir(exist_ok=True)
    sciezka = WYNIKI / f"{nazwa}.json"
    sciezka.write_text(json.dumps(wynik, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nZapisano -> {sciezka.relative_to(BASE)}")
