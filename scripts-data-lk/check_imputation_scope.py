"""
Kategoryzacja kolumn pod katem imputacji.
Wypisuje: ktore kolumny medyczne imputowac, ktore pominac i dlaczego.
"""

import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
df = pd.read_csv("datasets/cleaned/neuro_cleaning.csv", low_memory=False)
n = len(df)

# Parametry medyczne — wartości surowe do imputacji
# (celowo pomijamy kolumny -norm i -unit — są pochodne lub tekstowe)
MEDICAL_PARAMS = [
    # Morfologia krwi
    "HGB", "RBC", "MCV", "MCH", "MCHC", "HCT", "WBC", "PLT", "RDW", "MPV",
    # Rozmaz krwi
    "NEUT", "%NEUT", "LYMPH", "%LYMPH", "MONO", "%MONO", "EO", "%EO", "BAZO", "%BAZO",
    "NRBC", "%NRBC", "IG", "%IG", "P-LCR",
    # Biochemia
    "KREA", "Na", "K", "GLU", "BUN", "CRP", "ALT", "AST",
    # Koagulologia
    "INR", "PT", "APTT", "WAPTT", "WPT",
    # eGFR
    "eGFR-MDRD", "eGFRCKD",
]

missing = df[MEDICAL_PARAMS].isnull().sum()
pct = (missing / n * 100).round(1)

# Progi
THRESH_DROP = 50.0   # > 50% braków → rozważyć pominięcie
THRESH_WARN = 25.0   # > 25% braków → imputować ostrożnie

to_impute   = []
to_consider = []
to_skip     = []

for col in MEDICAL_PARAMS:
    p = pct[col]
    if p == 100.0:
        to_skip.append((col, missing[col], p, "100% braków — kolumna pusta"))
    elif p > THRESH_DROP:
        to_consider.append((col, missing[col], p, f">50% braków"))
    else:
        to_impute.append((col, missing[col], p, ""))

print(f"Zbior: {n} wierszy\n")

print("=" * 65)
print("DO IMPUTACJI (<=50% brakow)")
print("=" * 65)
print(f"{'Parametr':<12} {'Braków':>8} {'%':>7}  Badanie")
print("-" * 65)
BADANIE = {
    "HGB":"morfologia","RBC":"morfologia","MCV":"morfologia","MCH":"morfologia",
    "MCHC":"morfologia","HCT":"morfologia","WBC":"morfologia","PLT":"morfologia",
    "RDW":"morfologia","MPV":"morfologia","NEUT":"rozmaz","LYMPH":"rozmaz",
    "MONO":"rozmaz","EO":"rozmaz","BAZO":"rozmaz","%NEUT":"rozmaz",
    "%LYMPH":"rozmaz","%MONO":"rozmaz","%EO":"rozmaz","%BAZO":"rozmaz",
    "P-LCR":"morfologia","KREA":"biochemia","Na":"elektrolity","K":"elektrolity",
    "GLU":"biochemia","BUN":"biochemia","CRP":"biochemia","ALT":"biochemia",
    "AST":"biochemia","INR":"koagulologia","PT":"koagulologia","APTT":"koagulologia",
    "WAPTT":"koagulologia","WPT":"koagulologia","eGFR-MDRD":"eGFR","eGFRCKD":"eGFR",
    "NRBC":"rozmaz","%NRBC":"rozmaz","IG":"rozmaz","%IG":"rozmaz",
}
for col, m, p, _ in to_impute:
    print(f"  {col:<10} {int(m):>8} {p:>6.1f}%  {BADANIE.get(col,'')}")

print()
print("=" * 65)
print("DO ROZWAZENIA (>50% brakow - imputacja ryzykowna)")
print("=" * 65)
for col, m, p, note in to_consider:
    print(f"  {col:<10} {int(m):>8} {p:>6.1f}%  {note}")

print()
print("=" * 65)
print("POMINAC (100% brakow)")
print("=" * 65)
for col, m, p, note in to_skip:
    print(f"  {col:<10} {note}")
