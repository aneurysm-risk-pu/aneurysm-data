"""
Skrypt czyszczący neuro_cleaning.csv z formatu {'values': [...]}:
- Kolumny tekstowe (examination_type, descriptive_result) — usuwane
- Kolumny numeryczne — lista wartości zastępowana średnią
Wynik zapisywany w miejscu (nadpisuje neuro_cleaning.csv) - zmiany widoczne w commitach.
"""

import ast
import os
import numpy as np
import pandas as pd

TARGET = "datasets/cleaned/neuro_cleaning.csv"

# Kolumny czysto tekstowe — do usunięcia
DROP_COLS = ["examination_type", "descriptive_result"]


def parse_values_cell(val):
    """
    Jeśli komórka ma format {'values': [...]}, wyciąga wartości i zwraca średnią.
    Jeśli wartości nie są numeryczne — zwraca NaN.
    Jeśli komórka nie ma tego formatu — zwraca ją bez zmian.
    """
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    if not stripped.startswith("{'values':"):
        return val
    try:
        parsed = ast.literal_eval(stripped)
        values = parsed.get("values", [])
        if not values:
            return np.nan
        float_vals = [float(v) for v in values]
        return float(np.mean(float_vals))
    except (ValueError, TypeError, KeyError):
        return np.nan


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    print(f"Wczytuję: {TARGET}")
    df = pd.read_csv(TARGET, low_memory=False)
    print(f"Rozmiar przed czyszczeniem: {df.shape}")

    # Usuń kolumny tekstowe
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Usunięto kolumny: {cols_to_drop}")

    # Napraw komórki {'values': [...]} w kolumnach object
    fixed_total = 0
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.startswith("{'values':")
        count = mask.sum()
        if count > 0:
            df.loc[mask, col] = df.loc[mask, col].apply(parse_values_cell)
            fixed_total += count

    print(f"Naprawiono komórek {{values: [...]}}: {fixed_total}")
    print(f"Rozmiar po czyszczeniu:   {df.shape}")

    df.to_csv(TARGET, index=False)
    print(f"Zapisano: {TARGET}")
    print("\nGotowe.")
