"""
Punkt 5: funkcja celu dla Optuny.

Pomysl metryki wazonej z reczna kara za false positives zostal odrzucony:
nie da sie ustalic kary za FP, skoro prawdziwych FP w KOR nie potrafimy
rozpoznac. Zostaje miara operacyjna:

    RecallHidden@q  — ilu ukrytych chorych odzyskujemy w puli, ktora realnie
                      da sie skierowac na obrazowanie

Wariant rozszerzony pilnuje dodatkowo znanych pozytywnych:

    S(q) = alfa * RecallHidden@q + (1 - alfa) * RecallKnown

`q` NIE jest hiperparametrem — wynika z przepustowosci diagnostyki i zostaje
zamrozone przed treningiem. Strojenie q w Optunie oznaczaloby wybieranie
definicji sukcesu pod wynik.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KonfiguracjaPU
from .metryki import przygotuj_tabele, recall_hidden_at_q, recall_known


def wartosc_celu(score: pd.Series, maska: pd.DataFrame, cfg: KonfiguracjaPU) -> float:
    """S(q) dla jednego foldu. Im wiecej, tym lepiej."""
    t = przygotuj_tabele(score, maska)
    rh = recall_hidden_at_q(t, cfg.q)
    if cfg.alfa >= 1.0:
        return rh
    rk = recall_known(t, cfg.q)
    return cfg.alfa * rh + (1.0 - cfg.alfa) * rk


def agreguj_po_foldach(wartosci: list[float]) -> float:
    """Jedna liczba dla Optuny: srednia po foldach, ignorujac puste foldy."""
    w = [v for v in wartosci if v is not None and not np.isnan(v)]
    return float(np.mean(w)) if w else float("nan")
