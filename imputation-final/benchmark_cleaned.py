"""
imputation-final/benchmark_cleaned.py
=======================================

Benchmark MICE na zbiorze po analizie korelacji (35 cech).

Metodologia identyczna z głównym benchmarkiem:
  1. Complete cases z aneurysm_concatted_cleaned.csv
  2. MinMaxScaler [0, 1] dopasowany na complete cases
  3. Maskowanie 10% wartości (seed=42)
  4. Imputacja MICE z finalnymi params (max_iter=7, n_est=78, max_depth=10)
  5. RMSE / MAE / KL na zamaskowanych pozycjach
  6. Porównanie z poprzednim benchmarkiem (38 cech)

Osobno dla KOR (label=0) i NEURO (label=1).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import create_mask, apply_mask, compute_rmse, compute_mae, evaluate_imputation

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV   = BASE_DIR / "aneurysm_concatted_cleaned.csv"
OUTPUT_CSV  = RESULTS_DIR / "benchmark_cleaned_results.csv"

META_COLS    = ["patient_id", "custom_id", "examination_date", "label"]
MASK_SEED    = 42
MASK_FRAC    = 0.10
FINAL_PARAMS = dict(max_iter=7, n_estimators=78, max_depth=10)

# Wyniki z poprzedniego benchmarku (38 cech) do porównania
PREV_RESULTS = {
    "kor":   dict(rmse=0.0861, mae=0.0315, kl=0.00368),
    "neuro": dict(rmse=0.0984, mae=0.0432, kl=0.01158),
}


def run_benchmark(df_feat: pd.DataFrame, label: str) -> dict:
    t0 = time.time()

    # Scaler na complete cases
    df_complete = df_feat.dropna()
    scaler = MinMaxScaler()
    scaler.fit(df_complete.values.astype(float))
    df_scaled = pd.DataFrame(
        scaler.transform(df_complete.values.astype(float)),
        columns=df_feat.columns,
    )

    n_complete = len(df_complete)
    print(f"\n  [{label.upper()}] Complete cases: {n_complete:,} / {len(df_feat):,}")
    print(f"  [{label.upper()}] Kolumny cech: {len(df_feat.columns)}")

    # Maskowanie 10%
    mask = create_mask(df_scaled, MASK_FRAC, seed=MASK_SEED)
    df_masked = apply_mask(df_scaled, mask)

    n_masked = mask.sum()
    print(f"  [{label.upper()}] Zamaskowano: {n_masked:,} wartości ({100*MASK_FRAC:.0f}%)")

    # MICE
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

    arr_imputed = imputer.fit_transform(df_masked.values.astype(float))
    elapsed = time.time() - t0

    # Metryki
    original = df_scaled.values
    rmse = compute_rmse(original, arr_imputed, mask)
    mae  = compute_mae(original, arr_imputed, mask)

    # KL divergence
    result = evaluate_imputation(
        df_original=df_scaled,
        imputed_arr=arr_imputed,
        mask=mask,
        columns=list(df_feat.columns),
    )
    kl = result["KL_mean"]
    neg = result["negatives"]

    print(f"  [{label.upper()}] RMSE={rmse:.4f}  MAE={mae:.4f}  KL={kl:.5f}  "
          f"neg={neg}  ({elapsed:.0f}s / {elapsed/60:.1f} min)")

    # Porównanie z poprzednim benchmarkiem
    prev = PREV_RESULTS.get(label.lower())
    if prev:
        d_rmse = rmse - prev["rmse"]
        sign = "+" if d_rmse >= 0 else ""
        print(f"  [{label.upper()}] vs poprzedni benchmark (38 cech): "
              f"ΔRMSE={sign}{d_rmse:.4f}  "
              f"({'gorszy' if d_rmse > 0.001 else 'lepszy' if d_rmse < -0.001 else 'bez zmian'})")

    return dict(dataset=label, n_complete=n_complete, n_feat=len(df_feat.columns),
                rmse=round(rmse, 4), mae=round(mae, 4), kl=round(kl, 5),
                neg_values=neg, elapsed_s=round(elapsed))


def main():
    print("=" * 65)
    print("BENCHMARK MICE — aneurysm_concatted_cleaned.csv (35 cech)")
    print(f"Params: max_iter={FINAL_PARAMS['max_iter']}, "
          f"n_estimators={FINAL_PARAMS['n_estimators']}, "
          f"max_depth={FINAL_PARAMS['max_depth']}")
    print("=" * 65)

    df = pd.read_csv(INPUT_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    results = []
    for label_val, label_name in [(0, "kor"), (1, "neuro")]:
        df_sub = df[df["label"] == label_val][feat_cols].reset_index(drop=True)
        results.append(run_benchmark(df_sub, label_name))

    # Zapis
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 65)
    print("PODSUMOWANIE")
    print("=" * 65)
    print(df_results.to_string(index=False))
    print(f"\n[ZAPISANO] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
