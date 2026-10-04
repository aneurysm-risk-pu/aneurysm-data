"""
Testy czesci B etapu 3 — zamrozonego podzialu na foldy (sgkf/podzial.py).

Wlasnosci sprawdzamy na danych syntetycznych, gdzie znamy prawidlowa odpowiedz.
Ostatni test korzysta z zapisanego artefaktu i prawdziwych danych — jest
pomijany, jesli podzial nie zostal jeszcze zbudowany.

Uruchomienie:
    python 3-sgkf-split/sgkf/tests/test_podzial.py
"""

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from podzial import (KOL_PACJENT, WYNIKI, KonfiguracjaPodzialu,  # noqa: E402
                     diagnostyka, indeksy_foldow, kolumny_cech, podzial_wewnetrzny,
                     rozstepy, tabela_pacjentow, wczytaj_podzial, zbuduj_podzial)


# ---------------------------------------------------------------------------
# Dane syntetyczne
# ---------------------------------------------------------------------------

def dane_testowe(n_pac=2000, udzial_poz=0.1, n_mieszanych=10, seed=0) -> pd.DataFrame:
    """Uklad jak w prawdziwych danych: KOR w 2020-2021, NEURO glownie wczesniej,
    NEURO badani czesciej, kilku pacjentow w obu kohortach."""
    rng = np.random.default_rng(seed)
    wiersze = []
    for p in range(1, n_pac + 1):
        poz = p > n_pac * (1 - udzial_poz)
        n = rng.integers(1, 9) if poz else rng.integers(1, 4)
        if poz:
            start = pd.Timestamp("2020-06-01") if rng.random() < 0.15 else pd.Timestamp("2014-01-01")
        else:
            start = pd.Timestamp("2020-01-15")
        for k in range(n):
            wiersze.append({KOL_PACJENT: p, "label": int(poz),
                            "examination_date": start + pd.Timedelta(days=int(7 * k)),
                            "c0": rng.normal(10, 2)})
    df = pd.DataFrame(wiersze)
    # pacjenci mieszani: pozytywny dostaje dodatkowy rekord KOR
    mieszani = df.loc[df["label"] == 1, KOL_PACJENT].unique()[:n_mieszanych]
    extra = df[df[KOL_PACJENT].isin(mieszani)].groupby(KOL_PACJENT).head(1).copy()
    extra["label"] = 0
    extra["examination_date"] = pd.Timestamp("2021-03-01")
    return pd.concat([df, extra], ignore_index=True)


def _cfg(**kw) -> KonfiguracjaPodzialu:
    return KonfiguracjaPodzialu(**{**asdict(KonfiguracjaPodzialu(min_warstwa=10)), **kw})


# ---------------------------------------------------------------------------
# Warunki konieczne
# ---------------------------------------------------------------------------

def test_kazdy_pacjent_dokladnie_w_jednym_foldzie():
    df = dane_testowe()
    pac = tabela_pacjentow(df, _cfg())
    przydzial = zbuduj_podzial(pac, _cfg())
    assert przydzial.index.is_unique
    assert set(przydzial.index) == set(df[KOL_PACJENT])
    assert set(przydzial.unique()) == set(range(5))


def test_indeksy_rekordowe_nie_dziela_pacjenta_i_pokrywaja_wszystko():
    df = dane_testowe()
    cfg = _cfg()
    przydzial = zbuduj_podzial(tabela_pacjentow(df, cfg), cfg)
    testowe = []
    for _, tr, te in indeksy_foldow(df, przydzial):
        assert not set(df[KOL_PACJENT].iloc[tr]) & set(df[KOL_PACJENT].iloc[te])
        assert len(tr) + len(te) == len(df)
        testowe.extend(te)
    assert sorted(testowe) == list(range(len(df))), "rekord pominiety albo powtorzony w testach"


def test_wariant_35_cech_rozni_sie_tylko_trzema_kolumnami():
    df = pd.DataFrame(columns=["patient_id", "custom_id", "examination_date", "label",
                               "HGB", "CRP", "MONO", "%MONO", "K"])
    assert not {"patient_id", "custom_id", "examination_date", "label"} & set(kolumny_cech(df, "38")), \
        "kolumna meta nie moze byc cecha"
    assert set(kolumny_cech(df, "38")) - set(kolumny_cech(df, "35")) == {"CRP", "MONO", "%MONO"}


# ---------------------------------------------------------------------------
# Odtwarzalnosc
# ---------------------------------------------------------------------------

def test_podzial_nie_zalezy_od_kolejnosci_wierszy():
    df = dane_testowe()
    cfg = _cfg()
    a = zbuduj_podzial(tabela_pacjentow(df, cfg), cfg)
    b = zbuduj_podzial(tabela_pacjentow(df.sample(frac=1, random_state=3), cfg), cfg)
    assert a.sort_index().equals(b.sort_index()), "kolejnosc wierszy zmienila podzial"


def test_ziarno_zmienia_podzial_a_to_samo_ziarno_go_powtarza():
    df = dane_testowe()
    pac = tabela_pacjentow(df, _cfg())
    a = zbuduj_podzial(pac, _cfg(seed=42))
    assert a.equals(zbuduj_podzial(pac, _cfg(seed=42)))
    assert not a.equals(zbuduj_podzial(pac, _cfg(seed=7)))


# ---------------------------------------------------------------------------
# Balans — po to jest stratyfikacja wielokluczowa
# ---------------------------------------------------------------------------

def test_stratyfikacja_wielokluczowa_wyrownuje_czynniki_zaklocajace():
    df = dane_testowe(n_pac=4000)
    gorzej, lepiej = [], []
    for s in range(5):
        cfg1 = _cfg(warstwy=("etykieta",), seed=s)
        pac1 = tabela_pacjentow(df, cfg1)
        gorzej.append(rozstepy(diagnostyka(df, pac1, zbuduj_podzial(pac1, cfg1))))
        cfg3 = _cfg(seed=s)
        pac3 = tabela_pacjentow(df, cfg3)
        lepiej.append(rozstepy(diagnostyka(df, pac3, zbuduj_podzial(pac3, cfg3))))
    for k in ("poz_w_oknie_pp", "poz_5plus_rek_pp"):
        assert max(w[k] for w in lepiej) < max(w[k] for w in gorzej), k
    assert max(w["udzial_poz_pacjenci_pp"] for w in lepiej) < 0.5


def test_male_warstwy_sa_scalane():
    df = dane_testowe(n_pac=600)
    pac = tabela_pacjentow(df, _cfg(min_warstwa=50))
    assert pac["warstwa"].value_counts().min() >= 5, "warstwa mniejsza niz liczba foldow"
    assert pac["warstwa"].str.endswith("_inne").any()


def test_foldy_wewnetrzne_tylko_z_treningu():
    df = dane_testowe()
    cfg = _cfg()
    pac = tabela_pacjentow(df, cfg)
    przydzial = zbuduj_podzial(pac, cfg)
    outer_train = przydzial.index[przydzial != 0].to_numpy()
    outer_test = set(przydzial.index[przydzial == 0])
    for tr, te in podzial_wewnetrzny(pac, outer_train):
        assert not (set(tr) | set(te)) & outer_test
        assert not set(tr) & set(te)


# ---------------------------------------------------------------------------
# Artefakt na prawdziwych danych
# ---------------------------------------------------------------------------

def test_zapisany_podzial_jest_zgodny_z_danymi():
    if not (WYNIKI / "podzial_meta.json").exists():
        print("    [pominiety] brak results/podzial_meta.json — uruchom podzial.py")
        return
    przydzial, meta = wczytaj_podzial()   # rzuca wyjatek, jesli dane sie zmienily
    assert przydzial.index.is_unique
    assert len(przydzial) == meta["pacjenci"]
    assert meta["rozstepy"]["udzial_poz_pacjenci_pp"] < 0.1
    assert meta["rozstepy"]["poz_w_oknie_pp"] < 1.5


def test_zmiana_danych_uniewaznia_podzial():
    import json
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if not (WYNIKI / "podzial_meta.json").exists():
            return
        shutil.copy(WYNIKI / "pacjent_fold.csv", tmp / "pacjent_fold.csv")
        meta = json.loads((WYNIKI / "podzial_meta.json").read_text(encoding="utf-8"))
        meta["odcisk_danych_sha256"] = "0" * 16
        (tmp / "podzial_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        try:
            wczytaj_podzial(tmp)
        except RuntimeError:
            return
        raise AssertionError("podzial z innym odciskiem danych zostal przyjety")


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
