"""
Zadanie #30: profil pacjenta, czyli jak łączymy jego rekordy w jeden wiersz.

Model ocenia pacjenta, a pacjent ma od 1 do kilkudziesięciu rekordów.
Proponowana reguła: mediana każdej cechy (uzasadnienie: docs/30_profil_pacjenta.md).
Pozostałe reguły zostają do analizy wrażliwości.
"""

import pandas as pd

from dane import KOL_DATA, KOL_PACJENT

REGULY = ("mediana", "srednia", "ostatni", "pierwszy_rekord")


def profil(df: pd.DataFrame, cechy: list[str], regula: str = "mediana") -> pd.DataFrame:
    """Jeden wiersz na pacjenta (indeks: patient_id). Braki zostają brakami."""
    g = df.sort_values([KOL_PACJENT, KOL_DATA]).groupby(KOL_PACJENT)
    if regula == "mediana":
        return g[cechy].median()
    if regula == "srednia":
        return g[cechy].mean()
    if regula == "ostatni":           # ostatnia zmierzona wartość każdej cechy
        return g[cechy].last()
    if regula == "pierwszy_rekord":   # tylko pierwsze badanie (analiza wrażliwości)
        return g.head(1).set_index(KOL_PACJENT)[cechy]
    raise ValueError(f"nieznana reguła: {regula}")
