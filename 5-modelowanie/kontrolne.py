"""
Zadanie #31: modele kontrolne.

Model kontrolny dostaje tylko informację, która NIE jest chorobą (np. ile badań
miał pacjent). Jeśli już on dobrze „znajduje” ukrytych chorych, to znaczy, że
wynik da się osiągnąć skrótem. Właściwy model musi być od kontroli wyraźnie lepszy.

Kontrole (każda: jeden wiersz na pacjenta):
  - losowy               losowy ranking (liczony w uruchom_ocene.py), oczekiwany wynik = q
  - wiek i płeć          sama demografia
  - liczba badań         ile rekordów ma pacjent (KOR: 60% ma jeden, NEURO: 33%)
  - okres badań          data pierwszego i ostatniego badania (KOR 2019–2021, NEURO 2000–2024)
  - wzorzec braków       które badania wykonano (udział rekordów z pomiarem każdej cechy)
  - nazwy badań          NIE MA JESZCZE: czeka na #41 (Liwia)

Model odniesienia (nie kontrola): 38 cech, mediana na pacjenta, braki zostają (#30).
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from dane import KOL_DATA, KOL_PACJENT
from profil import profil

MODEL_ODNIESIENIA = "MODEL: 38 cech (mediana)"


def model() -> HistGradientBoostingClassifier:
    """Ten sam model dla wszystkich kontroli; obsługuje braki bez imputacji.
    Ustawienia jak w diagnostyce etapu 3 (kontrola_skrotow.py)."""
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def _rok(d: pd.Series) -> pd.Series:
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25


def zestawy_cech(df: pd.DataFrame, pac: pd.DataFrame, cechy: list[str]) -> dict[str, pd.DataFrame]:
    """nazwa -> cechy na pacjenta (indeks: patient_id)."""
    g = df.groupby(KOL_PACJENT)
    return {
        "wiek i płeć": g[["patient_age", "patient_sex"]].median(),
        "liczba badań": pac.set_index(KOL_PACJENT)[["n_rek"]],
        "okres badań": pd.DataFrame({"pierwsze": _rok(g[KOL_DATA].min()), "ostatnie": _rok(g[KOL_DATA].max())}),
        "wzorzec braków": df[cechy].notna().groupby(df[KOL_PACJENT]).mean(),
        MODEL_ODNIESIENIA: profil(df, cechy, "mediana"),
    }
