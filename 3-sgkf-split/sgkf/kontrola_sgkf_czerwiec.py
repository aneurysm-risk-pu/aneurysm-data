"""
kontrola_sgkf_czerwiec.py
=========================

Kontrola z 31.10.2026: co naprawdę było nie tak z podziałem z czerwca.

Wersja 1 to StratifiedGroupKFold(n_splits=5) bez tasowania, y = etykieta. Wcześniejsze
opisy mówiły, że taki podział zależy od kolejności wierszy i od wersji sklearn.
Ten skrypt sprawdza to wprost:

  1. Kolejność wierszy: potasowane wiersze, ten sam podział?
  2. Numery pacjentów: przemieszane numery, ten sam podział?
  3. Wersje sklearn: wynik na tym komputerze wobec zapisu z drugiego komputera
     (results/etap0_sgkf_stabilnosc.csv, sklearn 1.8, dane z 09.2026:
     data/processed/aneurysm_concatted_cleaned.csv), osobno bez tasowania i z shuffle=True.

Wynik: sgkf/results/kontrola_sgkf_czerwiec.json

Uruchomienie (~1 min):
    python 3-sgkf-split/sgkf/kontrola_sgkf_czerwiec.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from podzial import BASE_DIR, KOL_ETYKIETA, KOL_PACJENT, WYNIKI, wczytaj_rekordy  # noqa: E402

DANE_WRZESIEN = BASE_DIR / "data" / "processed" / "aneurysm_concatted_cleaned.csv"
ZAPIS_DRUGI_KOMPUTER = WYNIKI / "etap0_sgkf_stabilnosc.csv"


def przydzial(d: pd.DataFrame, **kw) -> pd.Series:
    out = {}
    sgkf = StratifiedGroupKFold(n_splits=5, **kw)
    for i, (_, te) in enumerate(sgkf.split(d, d[KOL_ETYKIETA], d[KOL_PACJENT])):
        out.update(dict.fromkeys(d[KOL_PACJENT].iloc[te].unique(), i))
    return pd.Series(out).sort_index()


def main() -> None:
    df = wczytaj_rekordy()[[KOL_PACJENT, KOL_ETYKIETA]]
    baza = przydzial(df)

    wiersze = [float((przydzial(df.sample(frac=1, random_state=s).reset_index(drop=True)) == baza).mean())
               for s in (1, 2, 3)]
    print(f"1. Potasowane wiersze: ten sam fold u {min(wiersze):.2%} pacjentów")

    ids = df[KOL_PACJENT].unique()
    mapa = dict(zip(ids, np.random.RandomState(0).permutation(ids)))
    odwrotna = {v: k for k, v in mapa.items()}
    p = przydzial(df.assign(**{KOL_PACJENT: df[KOL_PACJENT].map(mapa)}))
    p.index = [odwrotna[i] for i in p.index]
    numery = float((p.sort_index() == baza).mean())
    print(f"2. Przemieszane numery pacjentów: ten sam fold u {numery:.2%} pacjentów")

    wrzesien = pd.read_csv(DANE_WRZESIEN, usecols=[KOL_PACJENT, KOL_ETYKIETA])
    etykieta = wrzesien.groupby(KOL_PACJENT)[KOL_ETYKIETA].max()
    zapis = pd.read_csv(ZAPIS_DRUGI_KOMPUTER)
    zapis = zapis[(zapis["poziom"] == "rekordowy") & (zapis["wariant"] == "pozytywni") & (zapis["n_splits"] == 5)]

    wersje = {}
    for nazwa, kw, wiersz in (("bez_tasowania", {}, zapis[zapis["shuffle"] == False].iloc[0]),  # noqa: E712
                              ("shuffle_True_seed_42", {"shuffle": True, "random_state": 42},
                               zapis[zapis["seed"] == 42].iloc[0])):
        udzial = etykieta.groupby(przydzial(wrzesien, **kw)).mean()
        wersje[nazwa] = {
            "ten_komputer": {"min": float(udzial.min()), "max": float(udzial.max()),
                             "rozstep_pp": round(100 * float(udzial.max() - udzial.min()), 4)},
            "drugi_komputer": {"min": float(wiersz["udzial_poz_min"]), "max": float(wiersz["udzial_poz_max"]),
                               "rozstep_pp": round(100 * float(wiersz["udzial_poz_rozstep"]), 4)},
            "zgodne": bool(np.isclose(udzial.min(), wiersz["udzial_poz_min"], atol=1e-12)
                           and np.isclose(udzial.max(), wiersz["udzial_poz_max"], atol=1e-12)),
        }
        print(f"3. {nazwa}: tu {wersje[nazwa]['ten_komputer']['rozstep_pp']} pp, "
              f"zapis z drugiego komputera {wersje[nazwa]['drugi_komputer']['rozstep_pp']} pp, "
              f"zgodne: {wersje[nazwa]['zgodne']}")

    wynik = {"opis": "SGKF z czerwca (bez tasowania, y = etykieta): od czego naprawdę zależy",
             "potasowane_wiersze_ten_sam_fold": min(wiersze),
             "przemieszane_numery_ten_sam_fold": numery,
             "wersje_sklearn": wersje,
             "sklearn_na_tym_komputerze": sklearn.__version__}
    (WYNIKI / "kontrola_sgkf_czerwiec.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    print(f"\nWynik -> {(WYNIKI / 'kontrola_sgkf_czerwiec.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
