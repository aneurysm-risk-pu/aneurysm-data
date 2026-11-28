"""
Punkt 4: metryki liczone na pacjentach, nie na rekordach.

Pacjent z 38 rekordami nie moze wazyc 38 razy wiecej niz pacjent z jednym,
a decyzja kliniczna dotyczy osoby, nie pojedynczego pobrania.

Glowna metryka to RecallHidden@q: ilu ukrytych chorych odzyskujemy w puli,
ktora realnie da sie skierowac na badanie obrazowe.

Nazewnictwo, wazne dla raportu: wynik modelu to `risk score`, a nie
prawdopodobienstwo tetniaka. Metryki liczone z KOR jako klasa negatywna sa
metrykami P-vs-U, a nie skutecznoscia wykrywania choroby.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .dane import KOL_PACJENT


def _pula_u(tabela: pd.DataFrame) -> pd.DataFrame:
    """Pula rankingowa: pacjenci, ktorych model widzi jako nieoznaczonych."""
    return tabela[tabela["observed_label"] == 0]


def _top_k(pula: pd.DataFrame, q: float) -> pd.DataFrame:
    """Gorne q puli, z deterministycznym rozstrzyganiem remisow.

    Remisy rozstrzygamy po patient_id rosnaco. Bez tego wynik zalezalby od
    kolejnosci wierszy — ten sam blad, co brak shuffle w podziale danych.
    """
    k = max(1, math.ceil(q * len(pula)))
    uporzadkowana = pula.sort_values(
        ["score", KOL_PACJENT], ascending=[False, True], kind="mergesort")
    return uporzadkowana.head(k)


def przygotuj_tabele(score: pd.Series, maska: pd.DataFrame) -> pd.DataFrame:
    """Laczy risk score z prawda o pacjencie. score musi byc indeksowany patient_id."""
    t = maska.set_index(KOL_PACJENT).loc[score.index].copy()
    t["score"] = score
    return t.reset_index()


def recall_hidden_at_q(tabela: pd.DataFrame, q: float) -> float:
    """Odsetek ukrytych pozytywnych, ktorzy trafili w gorne q puli nieoznaczonej.

    To jest glowna metryka projektu i funkcja celu dla Optuny.
    """
    u = _pula_u(tabela)
    ukryci = int(u["is_hidden"].sum())
    if ukryci == 0:
        return float("nan")
    return float(_top_k(u, q)["is_hidden"].sum()) / ukryci


def udzial_ukrytych_w_top(tabela: pd.DataFrame, q: float) -> float:
    """Jaka czesc wskazanych do badania to kontrolowani ukryci pozytywni.

    UWAGA: to nie jest estymator precision. Status pozostalych pacjentow w
    puli nieoznaczonej jest nieznany — czesc z nich moze byc chora.
    """
    u = _pula_u(tabela)
    if len(u) == 0:
        return float("nan")
    top = _top_k(u, q)
    return float(top["is_hidden"].sum()) / len(top)


def lift_at_q(tabela: pd.DataFrame, q: float) -> float:
    """Ile razy lepiej od losowego wyboru tej samej liczby pacjentow."""
    u = _pula_u(tabela)
    if len(u) == 0:
        return float("nan")
    baza = u["is_hidden"].mean()
    if baza == 0:
        return float("nan")
    return udzial_ukrytych_w_top(tabela, q) / baza


def recall_known(tabela: pd.DataFrame, q: float) -> float:
    """Czy nie gubimy oczywistych przypadkow.

    Prog wyznacza gorne q puli nieoznaczonej; sprawdzamy, ilu znanych
    pozytywnych osiaga ten sam poziom score.
    """
    u = _pula_u(tabela)
    znani = tabela[tabela["observed_label"] == 1]
    if len(u) == 0 or len(znani) == 0:
        return float("nan")
    prog = _top_k(u, q)["score"].min()
    return float((znani["score"] >= prog).mean())


def auc_p_vs_u(tabela: pd.DataFrame) -> dict:
    """ROC-AUC i PR-AUC na etykietach widocznych — czyli pozytywni vs nieoznaczeni."""
    y = tabela["observed_label"].to_numpy()
    s = tabela["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"roc_auc_p_vs_u": float("nan"), "pr_auc_p_vs_u": float("nan")}
    return {"roc_auc_p_vs_u": float(roc_auc_score(y, s)),
            "pr_auc_p_vs_u": float(average_precision_score(y, s))}


def auc_kontrolowana(tabela: pd.DataFrame) -> dict:
    """AUC liczone wylacznie wewnatrz puli nieoznaczonej: ukryci vs reszta.

    To jest uczciwsza miara odzyskiwania niz P-vs-U, bo obie grupy sa dla
    modelu tak samo nieoznaczone.
    """
    u = _pula_u(tabela)
    y = u["is_hidden"].astype(int).to_numpy()
    s = u["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"roc_auc_kontrolowana": float("nan"), "pr_auc_kontrolowana": float("nan")}
    return {"roc_auc_kontrolowana": float(roc_auc_score(y, s)),
            "pr_auc_kontrolowana": float(average_precision_score(y, s))}


def komplet_metryk(score: pd.Series, maska: pd.DataFrame, q: float) -> dict:
    """Wszystkie metryki pacjentowe dla jednego foldu i jednej maski."""
    t = przygotuj_tabele(score, maska)
    u = _pula_u(t)
    wynik = {
        "recall_hidden_at_q": recall_hidden_at_q(t, q),
        "udzial_ukrytych_w_top": udzial_ukrytych_w_top(t, q),
        "lift_at_q": lift_at_q(t, q),
        "recall_known": recall_known(t, q),
        "pacjentow_w_puli_u": len(u),
        "ukrytych_w_puli_u": int(u["is_hidden"].sum()),
        "pacjentow_skierowanych": len(_top_k(u, q)) if len(u) else 0,
        "odsetek_skierowanych": (len(_top_k(u, q)) / len(t)) if len(u) and len(t) else float("nan"),
    }
    wynik.update(auc_p_vs_u(t))
    wynik.update(auc_kontrolowana(t))
    return wynik
