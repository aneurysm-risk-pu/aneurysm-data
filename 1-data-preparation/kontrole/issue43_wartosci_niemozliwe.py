"""
#43 Wartości ujemne i niemożliwe w preprocessingu Liwii (final_preprocessing.py).

Liczy na kolejnych krokach jej skryptu:
  - wartości ujemne,
  - zera w cechach, które nie mogą być zerem, i odsetki > 100% (to zamienia na brak krok 8, validate_clinical_values),
  - wiek > 120,
  - wartości niemożliwe z reguł kroku 4 projektu (REGULY_WARTOSCI w 3-sgkf-split/przygotowanie/przygotuj_dane.py).

Uruchomienie: python 1-data-preparation/kontrole/issue43_wartosci_niemozliwe.py
"""

import sys

import pandas as pd

from wspolne import BASE, KOHORTY, cechy_lab, przebieg, zapisz

sys.path.insert(0, str(BASE / "3-sgkf-split" / "przygotowanie"))
from przygotuj_dane import REGULY_WARTOSCI  # noqa: E402

# jak w validate_clinical_values(): tu zero jest dopuszczalne
ZERO_DOZWOLONE = {"%BAZO", "%EO", "%IG", "%LYMPH", "%MONO", "%NEUT", "%NRBC",
                  "BAZO", "EO", "IG", "LYMPH", "MONO", "NEUT", "NRBC"}
POKAZ = ["4 liczby", "7 wiek 18–100 i płeć", "8 walidacja wartości", "10 tylko kolumny wspólne"]


def policz(d: pd.DataFrame) -> dict:
    lab = cechy_lab(d)
    x = d[lab].apply(pd.to_numeric, errors="coerce")
    wynik = {
        "ujemne": int((x < 0).sum().sum()),
        "zera niedozwolone": int((x[[c for c in lab if c not in ZERO_DOZWOLONE]] == 0).sum().sum()),
        "odsetki > 100%": int((x[[c for c in lab if c.startswith("%")]] > 100).sum().sum()),
        "wiek > 120": int((pd.to_numeric(d["patient_age"], errors="coerce") > 120).sum()),
    }
    for _, opis, maska in REGULY_WARTOSCI:
        wynik[opis.split(" (")[0]] = int(maska(x).fillna(False).sum())
    return wynik


kroki = dict(przebieg())
wynik = {(k, n): policz(kroki[n][k]) for k in KOHORTY for n in POKAZ}
t = pd.DataFrame(wynik).T
t.index.names = ["grupa", "krok"]
print(t.to_string())

zapisz("issue43_wartosci_niemozliwe", {
    "skrypt_sprawdzany": "1-data-preparation/scripts-lf/final_preprocessing.py",
    "liczby": {f"{k} | {n}": v for (k, n), v in wynik.items()},
})
