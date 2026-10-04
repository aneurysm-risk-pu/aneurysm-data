"""
Testy części A etapu 3 — przygotowania danych do SGKF (przygotuj_dane.py).

Reguły sprawdzamy na danych syntetycznych, gdzie znamy prawidłową odpowiedź.
Ostatni test porównuje plik zapisany w repozytorium z wynikiem kodu na danych
źródłowych — pilnuje, żeby data/processed/aneurysm_sgkf_input.csv nie
rozjechał się z regułami.

Uruchomienie:
    python 3-sgkf-split/przygotowanie/tests/test_przygotowanie.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from przygotuj_dane import (KOL_MIESZANY, KOL_PACJENT, WYJSCIE, ZRODLO,  # noqa: E402
                            przygotuj)


def dane_testowe() -> pd.DataFrame:
    """Pacjent 1: KOR + NEURO (mieszany). Pacjenci 2–8: wartości graniczne."""
    return pd.DataFrame({
        KOL_PACJENT: [1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        "custom_id": ["1-a", "1-b", "1-c", "2-a", "3-a", "4-a", "5-a", "6-a", "7-a", "8-a"],
        "label":     [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        "KREA":      [0.9, 0.9, 0.9, 60.0, 20.0, 20.0, 0.9, 0.9, 12.0, 0.9],
        "eGFR-MDRD": [60.0, 60.0, 60.0, 60.0, 60.0, 8.0, 60.0, 60.0, np.nan, 60.0],
        "K":         [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 16.0, 12.0, 4.0, 4.0],
        "Na":        [140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 75.0],
        "WBC":       [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 230.0],
    })


def test_pacjent_mieszany_zostaje_pozytywny_bez_rekordow_kor():
    wynik, rekordy, _ = przygotuj(dane_testowe())
    p1 = wynik[wynik[KOL_PACJENT] == 1]
    assert len(rekordy) == 2 and set(rekordy["custom_id"]) == {"1-a", "1-b"}
    assert (p1["label"] == 1).all() and len(p1) == 1
    assert (p1[KOL_MIESZANY] == 1).all()
    assert (wynik.loc[wynik[KOL_PACJENT] != 1, KOL_MIESZANY] == 0).all()
    assert (wynik.groupby(KOL_PACJENT)["label"].nunique() == 1).all(), "pacjent nadal w dwóch kohortach"


def test_przygotowanie_nie_usuwa_pacjentow():
    zrodlo = dane_testowe()
    wynik, _, _ = przygotuj(zrodlo)
    assert set(wynik[KOL_PACJENT]) == set(zrodlo[KOL_PACJENT])


def test_wartosci_niemozliwe_na_brak_a_skrajne_mozliwe_zostaja():
    wynik, _, wartosci = przygotuj(dane_testowe())
    w = wynik.set_index("custom_id")
    assert np.isnan(w.loc["2-a", "KREA"]), "KREA > 50 musi zniknąć"
    assert np.isnan(w.loc["3-a", "KREA"]), "KREA 20 przy eGFR 60 to sprzeczność"
    assert w.loc["4-a", "KREA"] == 20.0, "KREA 20 przy eGFR 8 jest wiarygodne (niewydolność nerek)"
    assert w.loc["7-a", "KREA"] == 12.0, "bez eGFR nie ma podstaw do uznania KREA 12 za sprzeczne"
    assert np.isnan(w.loc["5-a", "K"]) and w.loc["6-a", "K"] == 12.0, "K 16 znika, K 12 zostaje"
    assert np.isnan(w.loc["8-a", "Na"]), "Na 75 znika"
    assert w.loc["8-a", "WBC"] == 230.0, "WBC 230 (np. białaczka) zostaje"
    assert len(wartosci) == 4


def test_reguly_zmieniaja_tylko_wskazane_komorki():
    zrodlo = dane_testowe()
    wynik, rekordy, wartosci = przygotuj(zrodlo)
    pozostale = zrodlo[~zrodlo["custom_id"].isin(rekordy["custom_id"])].set_index("custom_id")
    w = wynik.set_index("custom_id")[pozostale.columns.drop(KOL_PACJENT)]
    zmienione = set(zip(wartosci["custom_id"], wartosci["cecha"]))
    for cid in pozostale.index:
        for kol in w.columns:
            if (cid, kol) in zmienione:
                continue
            a, b = pozostale.loc[cid, kol], w.loc[cid, kol]
            assert (pd.isna(a) and pd.isna(b)) or a == b, f"zmieniona komórka spoza reguł: {cid} {kol}"


def test_wynik_nie_zalezy_od_kolejnosci_wierszy():
    zrodlo = dane_testowe()
    a = przygotuj(zrodlo)[0].sort_values("custom_id").reset_index(drop=True)
    b = przygotuj(zrodlo.iloc[::-1].reset_index(drop=True))[0].sort_values("custom_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_plik_w_repozytorium_zgodny_z_kodem():
    if not WYJSCIE.exists():
        print("    [pominięty] brak data/processed/aneurysm_sgkf_input.csv — uruchom przygotuj_dane.py")
        return
    oczekiwany, _, _ = przygotuj(pd.read_csv(ZRODLO))
    zapisany = pd.read_csv(WYJSCIE)
    assert list(zapisany.columns) == list(oczekiwany.columns), "inne kolumny niż daje kod"
    pd.testing.assert_frame_equal(zapisany, oczekiwany.reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    lokalne = dict(globals())
    testy = [(n, f) for n, f in lokalne.items() if n.startswith("test_")]
    bledy = 0
    for nazwa, fn in testy:
        try:
            fn()
            print(f"  OK   {nazwa}")
        except AssertionError as e:
            bledy += 1
            print(f"  BLAD {nazwa}: {e}")
    print(f"\n{len(testy) - bledy}/{len(testy)} testow przeszlo")
    sys.exit(1 if bledy else 0)
