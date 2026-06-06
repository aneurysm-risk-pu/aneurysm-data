"""
imputation-final/impute_final.py
=================================

Finalna imputacja pełnego zbioru (wszystkie wiersze, w tym z brakami).

Wejście : aneurysm_concatted.csv  (kor + neuro, 78 197 wierszy, label 0/1)
Wyjście : results/aneurysm_imputed_final.csv

Metodologia:
  - Scaler dopasowany na complete cases (spójność z benchmarkiem)
  - MICE: IterativeImputer + ExtraTrees, max_iter=7, n_estimators=78, max_depth=10
  - Wartości clampowane do [0, 1] po imputacji (min_value/max_value MICE)
  - Inwersja scaler → oryginalna skala
  - Kolumny meta (patient_id, custom_id, examination_date, label) zachowane bez zmian

Finalne params (z walidacji multi-seed + cross-param):
  max_iter=7, n_estimators=78, max_depth=10  — dla obu zbiorów identyczne RMSE
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV  = BASE_DIR / "aneurysm_concatted.csv"
OUTPUT_CSV = RESULTS_DIR / "aneurysm_imputed_final.csv"

META_COLS    = ["patient_id", "custom_id", "examination_date", "label"]
FINAL_PARAMS = dict(max_iter=7, n_estimators=78, max_depth=10)

# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # 1. Wczytaj pełny zbiór
    print(f"[1] Wczytywanie {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Wiersze: {len(df):,}   Kolumny: {df.shape[1]}")
    print(f"    label=0 (KOR): {(df['label']==0).sum():,}")
    print(f"    label=1 (NEURO): {(df['label']==1).sum():,}")

    # 2. Rozdziel meta od cech
    meta_present = [c for c in META_COLS if c in df.columns]
    feat_cols    = [c for c in df.columns if c not in META_COLS]
    df_meta = df[meta_present].copy()
    df_feat = df[feat_cols].copy()

    print(f"\n[2] Kolumny cech: {len(feat_cols)}")
    missing_counts = df_feat.isnull().sum()
    n_missing_cols = (missing_counts > 0).sum()
    n_missing_vals = missing_counts.sum()
    n_total_vals   = df_feat.size
    print(f"    Kolumny z brakami: {n_missing_cols}/{len(feat_cols)}")
    print(f"    Brakujące wartości: {n_missing_vals:,} / {n_total_vals:,}  "
          f"({100*n_missing_vals/n_total_vals:.1f}%)")
    print(f"    Kompletne wiersze: {df_feat.dropna().shape[0]:,} / {len(df_feat):,}")

    # 3. Dopasuj scaler na complete cases (spójność z benchmarkiem)
    print("\n[3] Dopasowanie MinMaxScaler na complete cases ...")
    df_complete = df_feat.dropna()
    scaler = MinMaxScaler()
    scaler.fit(df_complete.values.astype(float))
    print(f"    Complete cases użyte do fitu scalera: {len(df_complete):,}")

    # 4. Skaluj pełny zbiór — ręcznie, żeby zachować NaN
    scale_range = scaler.data_max_ - scaler.data_min_
    scale_range[scale_range == 0] = 1.0  # unikaj dzielenia przez 0
    arr_full = df_feat.values.astype(float)
    arr_scaled = (arr_full - scaler.data_min_) / scale_range

    print("\n[4] Skalowanie do [0, 1] (z zachowaniem NaN) ...")
    print(f"    Zakres po skalowaniu (bez NaN): "
          f"[{np.nanmin(arr_scaled):.4f}, {np.nanmax(arr_scaled):.4f}]")

    # 5. MICE imputacja
    print(f"\n[5] MICE imputacja ...")
    print(f"    Params: max_iter={FINAL_PARAMS['max_iter']}, "
          f"n_estimators={FINAL_PARAMS['n_estimators']}, "
          f"max_depth={FINAL_PARAMS['max_depth']}")

    estimator = ExtraTreesRegressor(
        n_estimators=FINAL_PARAMS["n_estimators"],
        max_depth=FINAL_PARAMS["max_depth"],
        random_state=42,
        n_jobs=-1,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=FINAL_PARAMS["max_iter"],
        min_value=0.0,
        max_value=1.0,
        random_state=42,
    )

    t_imp = time.time()
    arr_imputed_scaled = imputer.fit_transform(arr_scaled)
    elapsed_imp = time.time() - t_imp
    print(f"    Czas imputacji: {elapsed_imp:.0f}s  ({elapsed_imp/60:.1f} min)")

    # Clamp do [0, 1] (ostrożność)
    arr_imputed_scaled = np.clip(arr_imputed_scaled, 0.0, 1.0)

    # 6. Inwersja scali → oryginalna skala
    print("\n[6] Inwersja do oryginalnej skali ...")
    arr_original = arr_imputed_scaled * scale_range + scaler.data_min_

    df_imputed_feat = pd.DataFrame(arr_original, columns=feat_cols)

    # Sprawdź brakujące po imputacji
    remaining_nan = df_imputed_feat.isnull().sum().sum()
    print(f"    Pozostałe NaN po imputacji: {remaining_nan}")

    # 7. Złóż wynik
    df_result = pd.concat(
        [df_meta.reset_index(drop=True), df_imputed_feat.reset_index(drop=True)],
        axis=1,
    )
    # Przywróć oryginalną kolejność kolumn
    original_col_order = [c for c in df.columns if c in df_result.columns]
    df_result = df_result[original_col_order]

    # 8. Zapis
    df_result.to_csv(OUTPUT_CSV, index=False)
    elapsed_total = time.time() - t0
    print(f"\n[ZAPISANO] {OUTPUT_CSV}")
    print(f"    Wiersze: {len(df_result):,}   Kolumny: {df_result.shape[1]}")
    print(f"    Czas całkowity: {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")

    # 9. Krótka weryfikacja
    print("\n[WERYFIKACJA]")
    print(f"  Brakujące wartości w wyniku: {df_result.isnull().sum().sum()}")
    print(f"  Wartości < 0 (poza meta): "
          f"{(df_result[feat_cols] < 0).sum().sum()}")
    sample_feat = feat_cols[:5]
    print(f"\n  Przykładowe statystyki (5 kolumn):")
    print(df_result[sample_feat].describe().round(4).to_string())


if __name__ == "__main__":
    main()
