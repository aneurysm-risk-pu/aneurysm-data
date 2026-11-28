"""
Krok 2: stałe listy ukrytych chorych.

Wersja tymczasowa zadania #39 (Adrian). Ukrywamy 40% chorych (decyzja D5),
osobno w każdym foldzie, więc każdy fold ma ten sam odsetek ukrytych.
Ziarna 0–4 dają pięć niezależnych list. Ukryci w treningu dostają etykietę 0
(trafiają do nieoznaczonych), a w części oceny są tym, czego szukamy.
"""

import numpy as np
import pandas as pd

from dane import KOL_PACJENT

UDZIAL = 0.40
ZIARNA = (0, 1, 2, 3, 4)


def listy_ukrytych(pac: pd.DataFrame, udzial: float = UDZIAL, ziarna=ZIARNA) -> dict[int, set]:
    """ziarno -> zbiór patient_id ukrytych chorych (wszystkie foldy, także fold 4)."""
    chorzy = pac[pac["label"] == 1]
    wynik = {}
    for z in ziarna:
        rng = np.random.default_rng(z)
        ukryci = set()
        for _, g in chorzy.groupby("fold"):
            ids = np.sort(g[KOL_PACJENT].to_numpy())
            ukryci |= set(rng.choice(ids, size=int(round(udzial * len(ids))), replace=False).tolist())
        wynik[z] = ukryci
    return wynik
