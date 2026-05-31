"""
Krok 1: Ładowanie danych i filtrowanie do kompletnych rekordów.

Metodologia:
  - Wczytaj *_shortend.csv z katalogu root
  - Usuń kolumny identyfikatorów / dat (DROP_COLS)
  - Zostaw tylko kolumny numeryczne (patient_age, patient_sex włącznie)
  - Filtruj do wierszy bez żadnych braków (complete cases)
  - Skaluj MinMax do [0, 1]
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATASETS, DROP_COLS


def load_complete(name: str) -> dict:
    """
    Ładuje zbiór danych i zwraca słownik:
      df_complete   — DataFrame w oryginalnych jednostkach (tylko kompletne wiersze)
      df_scaled     — DataFrame po MinMax [0, 1]
      scaler        — dopasowany MinMaxScaler
      feature_cols  — lista kolumn cech
      name          — nazwa zbioru
    """
    path = DATASETS[name]
    df_raw = pd.read_csv(path)

    # Usuń kolumny identyfikatorów i dat
    drop = [c for c in DROP_COLS if c in df_raw.columns]
    df_raw = df_raw.drop(columns=drop)

    # Zostaw tylko numeryczne (patient_sex kodowane 0/1 zostaje)
    df_num = df_raw.select_dtypes(include=[np.number])

    # Kompletne wiersze
    df_complete = df_num.dropna().reset_index(drop=True)

    n_orig = len(df_raw)
    n_comp = len(df_complete)
    n_cols = df_complete.shape[1]

    print(f"[{name.upper()}] Wiersze oryginalne : {n_orig:>7,}")
    print(f"[{name.upper()}] Kompletne wiersze  : {n_comp:>7,}  ({100 * n_comp / n_orig:.1f}%)")
    print(f"[{name.upper()}] Kolumny cech       : {n_cols}")

    # MinMax scaling na kompletnym zbiorze
    scaler = MinMaxScaler()
    scaled_arr = scaler.fit_transform(df_complete.values.astype(float))
    df_scaled = pd.DataFrame(scaled_arr, columns=df_complete.columns)

    return {
        "df_complete":  df_complete,
        "df_scaled":    df_scaled,
        "scaler":       scaler,
        "feature_cols": list(df_complete.columns),
        "name":         name,
    }
