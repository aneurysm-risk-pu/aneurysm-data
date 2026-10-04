"""
przygotuj_dane.py
=================

CZĘŚĆ A etapu 3 — przygotowanie danych do SGKF.

Wejście:  data/imputation-inputs/aneurysm_concatted.csv
          (78 197 rekordów, 38 cech, dane PRZED imputacją)
Wyjście:  data/processed/aneurysm_sgkf_input.csv
          — jedyne wejście części B (3-sgkf-split/sgkf/)
          przygotowanie/results/usuniete_rekordy.csv     spis usuniętych rekordów
          przygotowanie/results/usuniete_wartosci.csv    spis wartości zamienionych na brak
          przygotowanie/results/podsumowanie.json        liczby do dokumentacji

Decyzje z 04.10.2026 (uzasadnienie: 1026_REALIZACJA_DO_SGKF.md, sekcja 6;
liczby: analiza_decyzji.py):

  1. Pacjenci obecni w obu kohortach (63 osoby, 264 rekordy). Pacjent nie może
     być jednocześnie pozytywny i nieoznaczony. Zostają jako pozytywni, a ich
     rekordy KOR są usuwane (145 rekordów). Kto to był, zapisuje spis
     results/usuniete_rekordy.csv — dane wynikowe nie mają dodatkowej kolumny.

  2. Wartości niemożliwe zamieniane na brak danych (MICE uzupełnia je później
     wewnątrz foldu z reszty profilu). Usuwamy tylko wartości, które nie mogą
     być prawdziwym wynikiem żywego pacjenta albo przeczą innemu pomiarowi
     z tego samego rekordu. Skrajne, ale możliwe w ciężkich stanach zostają
     (np. K 9–15 mmol/l, Na 80–100 mmol/l, WBC > 200 G/l przy białaczce).

       KREA > 50 mg/dl             poza zakresem mg/dl — błąd jednostki (µmol/l)
       KREA > 10 mg/dl przy        sprzeczność: eGFR liczy się z kreatyniny, a przy
         eGFR-MDRD >= 30           KREA > 10 mg/dl eGFR musi być skrajnie niskie
       K > 15 mmol/l               niezgodne z życiem
       Na < 80 mmol/l              niezgodne z życiem

Rozjazd czasowy kohort i asymetria braków NIE są tu korygowane — decyzja
z 04.10.2026: opisujemy je jako ograniczenia (dane są, jakie są).

Uruchomienie:
    python 3-sgkf-split/przygotowanie/przygotuj_dane.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
ZRODLO = BASE_DIR / "data" / "imputation-inputs" / "aneurysm_concatted.csv"
WYJSCIE = BASE_DIR / "data" / "processed" / "aneurysm_sgkf_input.csv"
WYNIKI = Path(__file__).resolve().parent / "results"

KOL_PACJENT = "patient_id"
KOL_ETYKIETA = "label"

# (cecha, opis reguły, maska na DataFrame)
REGULY_WARTOSCI = [
    ("KREA", "KREA > 50 mg/dl (poza zakresem mg/dl, błąd jednostki)",
     lambda d: d["KREA"] > 50),
    ("KREA", "KREA > 10 mg/dl przy eGFR-MDRD >= 30 (sprzeczność z eGFR)",
     lambda d: (d["KREA"] > 10) & (d["KREA"] <= 50) & (d["eGFR-MDRD"] >= 30)),
    ("K", "K > 15 mmol/l (niezgodne z życiem)",
     lambda d: d["K"] > 15),
    ("Na", "Na < 80 mmol/l (niezgodne z życiem)",
     lambda d: d["Na"] < 80),
]


def pacjenci_mieszani(df: pd.DataFrame) -> set:
    """Pacjenci, którzy mają rekordy w obu kohortach."""
    e = df.groupby(KOL_PACJENT)[KOL_ETYKIETA].nunique()
    return set(e[e > 1].index)


def usun_rekordy_kor_mieszanych(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usuwa rekordy KOR pacjentów obecnych też w NEURO.

    Zwraca (dane, spis usuniętych rekordów — z niego wiadomo, którzy to pacjenci).
    """
    mieszani = pacjenci_mieszani(df)
    do_usuniecia = df[KOL_PACJENT].isin(mieszani) & (df[KOL_ETYKIETA] == 0)
    kolumny = [c for c in (KOL_PACJENT, "custom_id", "examination_date", KOL_ETYKIETA) if c in df.columns]
    spis = df.loc[do_usuniecia, kolumny].copy()
    spis["powod"] = "rekord KOR pacjenta obecnego także w NEURO"
    return df[~do_usuniecia].copy(), spis


def usun_wartosci_niemozliwe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zamienia wartości niemożliwe na NaN. Zwraca (dane, spis zmienionych komórek)."""
    d = df.copy()
    wpisy = []
    for cecha, opis, maska_fn in REGULY_WARTOSCI:
        if cecha not in d.columns:
            continue
        maska = maska_fn(d).fillna(False).to_numpy()
        if maska.any():
            w = d.loc[maska, [c for c in (KOL_PACJENT, "custom_id", KOL_ETYKIETA) if c in d.columns]].copy()
            w["cecha"] = cecha
            w["wartosc"] = d.loc[maska, cecha].to_numpy()
            w["regula"] = opis
            wpisy.append(w)
            d.loc[maska, cecha] = np.nan
    spis = pd.concat(wpisy, ignore_index=True) if wpisy else pd.DataFrame(
        columns=[KOL_PACJENT, "custom_id", KOL_ETYKIETA, "cecha", "wartosc", "regula"])
    return d, spis


def przygotuj(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pełne przygotowanie: (dane, spis usuniętych rekordów, spis usuniętych wartości)."""
    d, rekordy = usun_rekordy_kor_mieszanych(df)
    d, wartosci = usun_wartosci_niemozliwe(d)
    return d.reset_index(drop=True), rekordy.reset_index(drop=True), wartosci


def podsumowanie(zrodlo: pd.DataFrame, wynik: pd.DataFrame,
                 rekordy: pd.DataFrame, wartosci: pd.DataFrame) -> dict:
    reguly = {}
    for _, opis, _ in REGULY_WARTOSCI:
        g = wartosci[wartosci["regula"] == opis]
        reguly[opis] = {"wartosci": int(len(g)), "pacjenci": int(g[KOL_PACJENT].nunique()),
                        "kor": int((g[KOL_ETYKIETA] == 0).sum()), "neuro": int((g[KOL_ETYKIETA] == 1).sum())}
    return {
        "plik_zrodlowy": str(ZRODLO.relative_to(BASE_DIR)),
        "plik_wynikowy": str(WYJSCIE.relative_to(BASE_DIR)),
        "rekordy_przed": int(len(zrodlo)), "rekordy_po": int(len(wynik)),
        "pacjenci_przed": int(zrodlo[KOL_PACJENT].nunique()), "pacjenci_po": int(wynik[KOL_PACJENT].nunique()),
        "usuniete_rekordy_kor_mieszanych": int(len(rekordy)),
        "pacjenci_mieszani": int(rekordy[KOL_PACJENT].nunique()),
        "wartosci_na_brak": reguly,
        "wartosci_na_brak_lacznie": int(len(wartosci)),
    }


def main() -> None:
    zrodlo = pd.read_csv(ZRODLO)
    wynik, rekordy, wartosci = przygotuj(zrodlo)

    WYJSCIE.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator ustawiony jawnie: ten sam plik na Windowsie i macOS
    wynik.to_csv(WYJSCIE, index=False, lineterminator="\n")

    WYNIKI.mkdir(parents=True, exist_ok=True)
    rekordy.to_csv(WYNIKI / "usuniete_rekordy.csv", index=False, lineterminator="\n")
    wartosci.to_csv(WYNIKI / "usuniete_wartosci.csv", index=False, lineterminator="\n")
    pods = podsumowanie(zrodlo, wynik, rekordy, wartosci)
    (WYNIKI / "podsumowanie.json").write_text(json.dumps(pods, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")

    print(f"Rekordy: {pods['rekordy_przed']:,} -> {pods['rekordy_po']:,} "
          f"(usunięto {pods['usuniete_rekordy_kor_mieszanych']} rekordów KOR "
          f"{pods['pacjenci_mieszani']} pacjentów obecnych w obu kohortach)")
    print(f"Pacjenci: {pods['pacjenci_przed']:,} -> {pods['pacjenci_po']:,}")
    print(f"Wartości zamienione na brak: {pods['wartosci_na_brak_lacznie']}")
    for opis, s in pods["wartosci_na_brak"].items():
        print(f"  {s['wartosci']:4d}  (KOR {s['kor']}, NEURO {s['neuro']}, pacjentów {s['pacjenci']})  {opis}")
    print(f"\nWynik -> {WYJSCIE.relative_to(BASE_DIR)}")
    print(f"Spis  -> {WYNIKI.relative_to(BASE_DIR)}/usuniete_rekordy.csv, usuniete_wartosci.csv, podsumowanie.json")


if __name__ == "__main__":
    main()
