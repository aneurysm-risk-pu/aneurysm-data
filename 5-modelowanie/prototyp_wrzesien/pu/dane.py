"""
Warstwa danych: wczytanie, etykieta pacjentowa, filtry kohorty, agregacja.

Wszystko, co zalezy od decyzji etapu 0, przechodzi przez konfiguracje.
Funkcje sa czyste — dostaja DataFrame, zwracaja nowy, niczego nie zapisuja.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import META_KOLUMNY, ZAKRESY_FIZJOLOGICZNE, KonfiguracjaPU

KOL_PACJENT = "patient_id"
KOL_DATA = "examination_date"
KOL_ETYKIETA = "label"


def wczytaj(cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Wczytuje dane PRZED imputacja — imputacja dzieje sie wewnatrz foldu."""
    df = pd.read_csv(cfg.sciezka_danych)
    df[KOL_DATA] = pd.to_datetime(df[KOL_DATA])
    return df


def kolumny_cech(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_KOLUMNY]


# ---------------------------------------------------------------------------
# 0.5 — wartosci skrajne
# ---------------------------------------------------------------------------

def zastosuj_zakresy(df: pd.DataFrame, cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Wartosci poza zakresem przezycia zamienia na brak danych (opcjonalnie).

    Traktujemy je jak brak, a nie jak zero czy wartosc brzegowa, bo imputacja
    wewnatrz foldu potrafi je odtworzyc z reszty profilu. Domyslnie wylaczone,
    dopoki nie zapadnie decyzja 0.5.
    """
    if cfg.wartosci_skrajne == "zostaw":
        return df
    d = df.copy()
    for kol, (lo, hi) in ZAKRESY_FIZJOLOGICZNE.items():
        if kol in d.columns:
            d.loc[(d[kol] < lo) | (d[kol] > hi), kol] = np.nan
    return d


# ---------------------------------------------------------------------------
# 0.1 — etykieta pacjentowa
# ---------------------------------------------------------------------------

def etykieta_pacjentowa(df: pd.DataFrame, cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Jedna etykieta na pacjenta plus flagi kontrolne.

    Zwraca tabele: patient_id, true_label, mieszany, n_rekordow.
    `true_label` to prawda o pacjencie — model jej nie widzi, kiedy dziala
    ukrywanie z punktu 2.
    """
    g = df.groupby(KOL_PACJENT)[KOL_ETYKIETA]
    pac = pd.DataFrame({
        "true_label": g.max().astype(int),
        "mieszany": g.nunique().gt(1),
        "n_rekordow": df.groupby(KOL_PACJENT).size(),
    }).reset_index()

    if cfg.mieszani == "bez_mieszanych":
        pac = pac[~pac["mieszany"]].copy()
    return pac.sort_values(KOL_PACJENT).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 0.2 — kontrola czasu i pochodzenia
# ---------------------------------------------------------------------------

def wspolne_okno_dat(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Przeciecie zakresow dat obu kohort."""
    kor = df.loc[df[KOL_ETYKIETA] == 0, KOL_DATA]
    neuro = df.loc[df[KOL_ETYKIETA] == 1, KOL_DATA]
    return max(kor.min(), neuro.min()), min(kor.max(), neuro.max())


def zastosuj_kontrole_czasu(df: pd.DataFrame, cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Wariant A z ETAP0_USTALENIA.md: ograniczenie do wspolnego okna dat.

    Uwaga: usuwa okolo 90% rekordow NEURO, wiec sluzy przede wszystkim jako
    analiza wrazliwosci, nie jako scenariusz glowny.
    """
    if cfg.kontrola_czasu == "brak":
        return df
    od, do = wspolne_okno_dat(df)
    return df[(df[KOL_DATA] >= od) & (df[KOL_DATA] <= do)].copy()


# ---------------------------------------------------------------------------
# 0.3 — agregacja rekordow do pacjenta
# ---------------------------------------------------------------------------

def agreguj_do_pacjenta(df: pd.DataFrame, feat: list[str],
                        cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Jeden wiersz na pacjenta wedlug reguly z decyzji 0.3.

    Maksimum i ostatni pomiar zostaly swiadomie odrzucone — patrz punkt 0.3
    w ETAP0_USTALENIA.md. Maksimum koreluje z liczba badan pacjenta, a liczba
    badan wynika z hospitalizacji, nie ze stanu zdrowia.
    """
    fn = {"mediana": "median", "srednia": "mean"}[cfg.regula_agregacji]
    return df.groupby(KOL_PACJENT)[feat].agg(fn).sort_index()


def wagi_rekordowe(df: pd.DataFrame) -> pd.Series:
    """Wagi odwrotne do liczby rekordow pacjenta.

    Potrzebne tylko przy `jednostka_treningu='rekord'`, zeby czesciej badani
    pacjenci nie wazyli wiecej w funkcji straty. Suma wag kazdego pacjenta
    wynosi 1, niezaleznie od liczby jego rekordow.
    """
    n = df.groupby(KOL_PACJENT)[KOL_PACJENT].transform("size")
    return 1.0 / n


# ---------------------------------------------------------------------------

def przygotuj(cfg: KonfiguracjaPU) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Pelna sciezka wejsciowa: dane rekordowe, tabela pacjentow, lista cech."""
    df = wczytaj(cfg)
    df = zastosuj_kontrole_czasu(df, cfg)
    df = zastosuj_zakresy(df, cfg)
    pac = etykieta_pacjentowa(df, cfg)
    df = df[df[KOL_PACJENT].isin(set(pac[KOL_PACJENT]))].copy()
    return df, pac, kolumny_cech(df)
