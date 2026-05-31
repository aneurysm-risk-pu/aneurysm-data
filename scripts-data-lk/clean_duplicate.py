"""
Czyszczenie neuro_cleaning.csv:
- Duplikat custom_id '194143-2017-W52' — zostawia rekord z późniejszą datą (2017-12-31)
Nadpisuje plik w miejscu.
"""

import os
import pandas as pd

TARGET = "datasets/cleaned/neuro_cleaning.csv"

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

df = pd.read_csv(TARGET, low_memory=False)
print(f"Rozmiar przed czyszczeniem: {df.shape}")

# --- 1. Duplikat custom_id ---
dup_id = "194143-2017-W52"
dup_mask = df["custom_id"] == dup_id

if dup_mask.sum() > 1:
    # Zachowaj ostatni rekord (najpóźniejsza data badania)
    df["examination_date"] = pd.to_datetime(df["examination_date"])
    keep_idx = df[dup_mask].sort_values("examination_date").index[-1]
    drop_idx = df[dup_mask].index.difference([keep_idx])
    df = df.drop(index=drop_idx)
    print(f"Usunięto duplikat custom_id='{dup_id}' (zachowano rekord z {df.loc[keep_idx, 'examination_date'].date()})")
    # Przywróć examination_date jako string (oryginalny format)
    df["examination_date"] = df["examination_date"].dt.strftime("%Y-%m-%d")
else:
    print(f"Duplikat '{dup_id}': nie znaleziono lub już usunięty")

print(f"Rozmiar po czyszczeniu:   {df.shape}")
df.to_csv(TARGET, index=False)
print(f"Zapisano: {TARGET}")


# --- 1. Duplikat custom_id ---
dup_id = "194143-2017-W52"
dup_mask = df["custom_id"] == dup_id

if dup_mask.sum() > 1:
    # Zachowaj ostatni rekord (najpóźniejsza data badania)
    df["examination_date"] = pd.to_datetime(df["examination_date"])
    keep_idx = df[dup_mask].sort_values("examination_date").index[-1]
    drop_idx = df[dup_mask].index.difference([keep_idx])
    df = df.drop(index=drop_idx)
    print(f"Usunięto duplikat custom_id='{dup_id}' (zachowano rekord z {df.loc[keep_idx, 'examination_date'].date()})")
    # Przywróć examination_date jako string (oryginalny format)
    df["examination_date"] = df["examination_date"].dt.strftime("%Y-%m-%d")
else:
    print(f"Duplikat '{dup_id}': nie znaleziono lub już usunięty")

print(f"Rozmiar po czyszczeniu:   {df.shape}")
df.to_csv(TARGET, index=False)
print(f"Zapisano: {TARGET}")
