"""
Krok 1: wczytanie danych i układy foldów.

Wersja tymczasowa zadania #38 (Adrian). Gdy będzie gotowa wersja zespołu,
podmieniamy ją, zachowując to samo zachowanie:
  - fold 4 jest zbiorem testowym (decyzja D1) i nie występuje w układach roboczych,
  - układ końcowy (foldy 0–3 -> fold 4) trzeba włączyć świadomie: koncowy=True.
"""

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "3-sgkf-split" / "sgkf"))
from podzial import (DOMYSLNA, KOL_DATA, KOL_PACJENT, kolumny_cech,  # noqa: E402
                     tabela_pacjentow, wczytaj_podzial, zbuduj_podzial)

WEJSCIE = BASE / "data" / "processed" / "aneurysm_sgkf_input.csv"
FOLDY_ROBOCZE = (0, 1, 2, 3)
FOLD_TESTOWY = 4


def wczytaj(sciezka: Path = WEJSCIE) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(rekordy, tabela pacjentów z kolumną `fold`, lista 38 cech).

    Dla domyślnego wejścia używamy zapisanego przydziału (`pacjent_fold.csv`,
    sprawdzany odciskiem danych). Dla innych danych, np. syntetycznych w trybie
    `brak`, przydział liczymy w pamięci tą samą metodą.
    """
    df = pd.read_csv(sciezka)
    df[KOL_DATA] = pd.to_datetime(df[KOL_DATA])
    pac = tabela_pacjentow(df, DOMYSLNA)
    if Path(sciezka).resolve() == WEJSCIE.resolve():
        przydzial, _ = wczytaj_podzial()
    else:
        przydzial = zbuduj_podzial(df, pac, DOMYSLNA)
    pac["fold"] = pac[KOL_PACJENT].map(przydzial)
    assert pac["fold"].notna().all(), "pacjent bez foldu"
    return df, pac, kolumny_cech(df)


def uklady(pac: pd.DataFrame, koncowy: bool = False) -> list[tuple[str, pd.Index, pd.Index]]:
    """Lista (nazwa, pacjenci treningu, pacjenci oceny)."""
    fold = pac.set_index(KOL_PACJENT)["fold"]
    if koncowy:
        return [("koncowy 0-3 -> 4", fold.index[fold.isin(FOLDY_ROBOCZE)], fold.index[fold == FOLD_TESTOWY])]
    return [(f"walidacja {v}", fold.index[fold.isin(set(FOLDY_ROBOCZE) - {v})], fold.index[fold == v])
            for v in FOLDY_ROBOCZE]
