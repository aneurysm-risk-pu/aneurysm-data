"""
#43 Wartości ujemne i niemożliwe: ile ich jest na każdym etapie danych.

Reguły „niemożliwe” są te same co w kroku 4 (REGULY_WARTOSCI w
3-sgkf-split/przygotowanie/przygotuj_dane.py), plus wartości ujemne w wynikach
badań i wiek powyżej 120 lat. Dane surowe zamieniamy na liczby tak jak krok 1
(średnia z listy {'values': [...]}).

Uruchomienie: python 1-data-preparation/kontrole/issue43_wartosci_niemozliwe.py
"""

import sys

import pandas as pd

from wspolne import BASE, ETYKIETA, KOHORTY, cechy_lab, liczba, po_kroku1, po_kroku2, po_kroku3, po_kroku4, surowe, zapisz

sys.path.insert(0, str(BASE / "3-sgkf-split" / "przygotowanie"))
from przygotuj_dane import REGULY_WARTOSCI  # noqa: E402

lab = cechy_lab(surowe("kor").head(1))


def policz(d: pd.DataFrame) -> dict:
    kol = [c for c in lab if c in d.columns]
    x = d[kol].apply(pd.to_numeric, errors="coerce")
    wiek = pd.to_numeric(d["patient_age"], errors="coerce")
    wynik = {"wartości ujemne": int((x < 0).sum().sum()), "wiek > 120": int((wiek > 120).sum())}
    for cecha, opis, maska in REGULY_WARTOSCI:
        wynik[opis.split(" (")[0]] = int(maska(x).fillna(False).sum())
    return wynik


def surowe_liczbowo(k: str) -> pd.DataFrame:
    d = surowe(k)
    d[lab] = d[lab].apply(lambda s: s.map(liczba))
    return d


d3, d4 = po_kroku3(), po_kroku4()
etapy = {}
for k in KOHORTY:
    etapy[(k, "dane surowe")] = policz(surowe_liczbowo(k))
    etapy[(k, "po kroku 1 (*_cleaning)")] = policz(po_kroku1(k))
    etapy[(k, "po kroku 2 (*_shortend)")] = policz(po_kroku2(k))
    etapy[(k, "po kroku 3 (concatted)")] = policz(d3[d3["label"] == ETYKIETA[k]])
    etapy[(k, "po kroku 4 (wejście SGKF)")] = policz(d4[d4["label"] == ETYKIETA[k]])

t = pd.DataFrame(etapy).T
t.index.names = ["grupa", "etap"]
print(t.to_string())

zapisz("issue43_wartosci_niemozliwe", {f"{g} | {e}": v for (g, e), v in etapy.items()})
