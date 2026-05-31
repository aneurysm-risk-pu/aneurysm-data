
#Czyszczenie NEURO i KOR:
#- usuwamy kolumny z  -unit (jednostki)
#- usuwamy kolumny opisowe: examination_type, descriptive_result
#- spłaszczamy {'values': [...]} - średnia lub NaN jak pusty
#- ujednolicamy patient_sex 0/1


import ast
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATASETS = {
    "NEURO": {
        "input":  "datasets/cleaned/neuro_cleaning.csv",
        "output": "datasets/cleaned/neuro_cleaning.csv",
    },
    "KOR": {
        "input":  "datasets/org/kor_merged_aggregated_1W_mean.csv",
        "output": "datasets/cleaned/kor_cleaning.csv",
    },
}

DROP_COLS = ["examination_type", "descriptive_result"]


def parse_values_cell(val):
    """{'values': [...]} -> srednia liczbowa lub NaN."""
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


def clean(name, input_path, output_path):
    print(f"\n{'='*60}")
    print(f"[{name}] Wczytuje: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[{name}] Rozmiar wejsciowy: {df.shape}")

    # 1. Usuwamy jednostki
    unit_cols = [c for c in df.columns if c.endswith("-unit")]
    if unit_cols:
        df = df.drop(columns=unit_cols)
        print(f"[{name}] Usunieto {len(unit_cols)} kolumn -unit")

    # 2. usuwamy opisoawe
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[{name}] Usunieto kolumny opisowe: {to_drop}")

    # 3. Splaszczamy słowniki
    fixed = 0
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.startswith("{'values':")
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = df.loc[mask, col].apply(parse_values_cell)
            fixed += n
    print(f"[{name}] Komorek {{values: [...]}} splaszczonych: {fixed}")

    # 4. Ujednolicenie patient_sex 
    if "patient_sex" in df.columns:
        before = df["patient_sex"].dtype
        df["patient_sex"] = pd.to_numeric(df["patient_sex"], errors="coerce").astype("Int64")
        print(f"[{name}] patient_sex: {before} -> Int64")

    print(f"[{name}] Rozmiar wyjsciowy: {df.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[{name}] Zapisano: {output_path}")


if __name__ == "__main__":
    for name, paths in DATASETS.items():
        clean(name, paths["input"], paths["output"])
    print("\nGotowe.")
