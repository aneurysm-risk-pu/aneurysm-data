"""
Punkt 2: kontrolowany test odzyskiwania — ukrywanie czesci pozytywnych.

Z grupy znanych pozytywnych losujemy pacjentow i odbieramy im widoczna
etykiete. Model traktuje ich jak nieoznaczonych, a my sprawdzamy, czy mimo to
trafiaja wysoko w rankingu ryzyka.

Zasady, ktore kod wymusza:
  - losujemy calych PACJENTOW, nigdy pojedynczych rekordow,
  - trzymamy trzy kolumny: true_label, observed_label, is_hidden,
  - kazda maska ma zapisany udzial i seed, wiec jest odtwarzalna,
  - te same maski trafiaja do wszystkich modeli.

To NIE jest grupa kontrolna. Mierzy odzyskiwanie znanych przypadkow, a nie
rzeczywisty odsetek falszywie dodatnich w KOR — tego w tych danych zmierzyc
sie nie da.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KonfiguracjaPU
from .dane import KOL_PACJENT


def maska_ukrycia(pac: pd.DataFrame, udzial: float, seed: int) -> pd.DataFrame:
    """Jedna maska: kto z pozytywnych zostaje ukryty przy danym udziale i ziarnie."""
    if not 0.0 <= udzial < 1.0:
        raise ValueError(f"udzial musi byc w [0, 1), jest {udzial}")

    m = pac[[KOL_PACJENT, "true_label"]].copy()
    pozytywni = m.loc[m["true_label"] == 1, KOL_PACJENT].to_numpy()

    rng = np.random.default_rng(seed)
    n_ukrytych = int(round(udzial * len(pozytywni)))
    ukryci = set(rng.choice(pozytywni, size=n_ukrytych, replace=False)) if n_ukrytych else set()

    m["is_hidden"] = m[KOL_PACJENT].isin(ukryci)
    m["observed_label"] = np.where(m["is_hidden"], 0, m["true_label"]).astype(int)
    m["udzial_ukrycia"] = udzial
    m["seed_ukrycia"] = seed
    return m


def wszystkie_maski(pac: pd.DataFrame, cfg: KonfiguracjaPU) -> pd.DataFrame:
    """Komplet masek: scenariusz glowny plus analizy wrazliwosci, kazdy na wszystkich ziarnach.

    Powstaje przed treningiem i zostaje zamrozony — to artefakt do raportu.
    """
    udzialy = [cfg.udzial_ukrycia_glowny, *cfg.udzialy_ukrycia_wrazliwosc]
    ramki = []
    for u in udzialy:
        for s in cfg.seedy_ukrycia:
            m = maska_ukrycia(pac, u, s)
            m["scenariusz"] = "glowny" if u == cfg.udzial_ukrycia_glowny else "wrazliwosc"
            ramki.append(m)
    return pd.concat(ramki, ignore_index=True)


def sprawdz_maske(m: pd.DataFrame) -> None:
    """Asercje, ktore musza przejsc, zanim maska pojdzie do treningu."""
    ukryci = m[m["is_hidden"]]
    assert (ukryci["true_label"] == 1).all(), "ukryto pacjenta, ktory nie byl pozytywny"
    assert (ukryci["observed_label"] == 0).all(), "ukryty pacjent ma nadal widoczna jedynke"
    jawni = m[~m["is_hidden"]]
    assert (jawni["observed_label"] == jawni["true_label"]).all(), "zmieniono etykiete nieukrytemu"
    assert m[KOL_PACJENT].is_unique, "pacjent wystepuje w masce wiecej niz raz"
