"""
Wspólne funkcje dla kontroli preprocessingu (zadania #43–#46, Sprint 6).

Kontrole sprawdzają preprocessing Liwii: 1-data-preparation/scripts-lf/final_preprocessing.py
(dane surowe -> data/processed/{kor,neuro}_preprocessed_final.csv -> concatted_preprocessed.csv).

`przebieg()` powtarza jej kroki w tej samej kolejności, korzystając z jej funkcji,
i zwraca stan danych po każdym kroku. Na końcu sprawdza, że wynik jest identyczny
z zapisanymi plikami — dzięki temu liczby z kontroli dotyczą dokładnie tych danych.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DANE = BASE / "data"
WYNIKI = Path(__file__).resolve().parent / "results"
SKRYPT_LIWII = BASE / "1-data-preparation" / "scripts-lf" / "final_preprocessing.py"
KOHORTY = ("kor", "neuro")


def surowe(k: str) -> pd.DataFrame:
    return pd.read_csv(DANE / "raw" / f"{k}_merged_aggregated_1W_mean.csv", encoding="utf-8-sig", low_memory=False)


def wynik_liwii(k: str) -> pd.DataFrame:
    return pd.read_csv(DANE / "processed" / f"{k}_preprocessed_final.csv", encoding="utf-8-sig")


def cechy_lab(df: pd.DataFrame) -> list[str]:
    """Kolumny wyników badań obecne w df (z 40 badań danych surowych: te, które mają kolumnę `-unit`)."""
    lab = [c for c in pd.read_csv(DANE / "raw" / "kor_merged_aggregated_1W_mean.csv", nrows=0,
                                  encoding="utf-8-sig").columns if not c.endswith(("-unit", "-norm"))]
    return [c for c in lab if c in df.columns and c not in
            ("custom_id", "patient_id", "examination_date", "patient_age", "patient_sex",
             "diagnosis_id", "descriptive_result", "examination_type")]


def _funkcje_liwii() -> dict:
    """Funkcje z final_preprocessing.py. Adnotacje `list[str] | None` wymagają Pythona 3.10,
    więc odkładamy ich ocenę (plik Liwii zostaje bez zmian)."""
    kod = "from __future__ import annotations\n" + SKRYPT_LIWII.read_text(encoding="utf-8")
    ns = {"__name__": "final_preprocessing", "__file__": str(SKRYPT_LIWII)}
    exec(compile(kod, str(SKRYPT_LIWII), "exec"), ns)
    return ns


def przebieg() -> list[tuple[str, dict]]:
    """[(nazwa kroku, {"kor": df, "neuro": df}), ...] w kolejności run_preprocessing()."""
    f = _funkcje_liwii()
    d = {k: surowe(k) for k in KOHORTY}
    kroki = [("0 dane surowe", dict(d))]
    with contextlib.redirect_stdout(io.StringIO()):                  # funkcje Liwii dużo wypisują
        def krok(nazwa, fn):
            for k in KOHORTY:
                d[k] = fn(d[k], k.upper())
            kroki.append((nazwa, dict(d)))
        krok("1 duplikaty custom_id", f["deduplicate_within_cohort"])
        krok("2 rozpakowanie słowników", f["unpack_all_dict_columns"])
        krok("3 bez jednostek, norm i nazw badań", f["process_and_drop_units_and_norms"])
        krok("4 liczby", f["convert_numeric_features"])
        krok("5 bez diagnosis_id i opisów", lambda x, _: x.drop(columns=["diagnosis_id", "descriptive_result"]))
        wspolni = set(d["kor"]["patient_id"]) & set(d["neuro"]["patient_id"])
        krok("6 bez pacjentów obecnych w obu grupach", lambda x, _: x[~x["patient_id"].isin(wspolni)].copy())
        krok("7 wiek 18–100 i płeć", f["clean_demographics"])
        krok("8 walidacja wartości", f["validate_clinical_values"])
        krok("9 ponad 60% braków (kolumny, potem rekordy)",
             lambda x, k: f["drop_high_missingness"](x, threshold=0.60, cohort_name=k))
        wspolne_kol = [c for c in d["kor"].columns if c in set(d["neuro"].columns)]
        krok("10 tylko kolumny wspólne", lambda x, _: x[wspolne_kol].copy())
    for k in KOHORTY:   # kontrola: odtworzony wynik = pliki Liwii
        zapis = wynik_liwii(k)
        assert d[k].shape == zapis.shape and list(d[k].columns) == list(zapis.columns), k
        num = zapis.select_dtypes("number").columns
        assert np.allclose(d[k][num].astype(float).to_numpy(), zapis[num].to_numpy(), equal_nan=True), k
    return kroki


def zapisz(nazwa: str, wynik: dict) -> None:
    WYNIKI.mkdir(exist_ok=True)
    sciezka = WYNIKI / f"{nazwa}.json"
    sciezka.write_text(json.dumps(wynik, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nZapisano -> {sciezka.relative_to(BASE)}")
