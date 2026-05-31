"""
Analiza brakujących wartości w neuro_cleaning.csv.
Wypisuje: ile NaN na kolumnę, % braków, sortuje od największej liczby braków.
"""

import os
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
path = "datasets/cleaned/neuro_cleaning.csv"

df = pd.read_csv(path, low_memory=False)
print(f"Zaladowano: {df.shape[0]} wierszy x {df.shape[1]} kolumn\n")

# Tylko kolumny numeryczne (wartości do imputacji)
num_cols = df.select_dtypes(include="number").columns.tolist()

missing = (
    df[num_cols]
    .isnull()
    .sum()
    .rename("brakow")
    .to_frame()
)
missing["procent"] = (missing["brakow"] / len(df) * 100).round(1)
missing = missing[missing["brakow"] > 0].sort_values("brakow", ascending=False)

print(f"Kolumny numeryczne z brakami: {len(missing)} / {len(num_cols)}\n")
print(f"{'Kolumna':<20} {'Braków':>8} {'%':>7}")
print("-" * 38)
for col, row in missing.iterrows():
    print(f"{col:<20} {int(row['brakow']):>8} {row['procent']:>6.1f}%")

# Podsumowanie
print()
total_cells = len(df) * len(num_cols)
total_missing = missing["brakow"].sum()
print(f"Łącznie brakujących komórek: {total_missing} / {total_cells} ({total_missing/total_cells*100:.1f}%)")

# Ile wierszy ma co najmniej 1 brak
rows_with_any_missing = df[num_cols].isnull().any(axis=1).sum()
print(f"Wiersze z co najmniej 1 brakiem: {rows_with_any_missing} / {len(df)} ({rows_with_any_missing/len(df)*100:.1f}%)")
