"""
imputation-final/run_all.py
============================

Główny skrypt porównania metod imputacji.

Uruchomienie:
  cd imputation-final
  python run_all.py                    # oba zbiory, maska seed=42
  python run_all.py --dataset neuro    # tylko NEURO
  python run_all.py --mask-seed 7      # inna maska (weryfikacja stabilności)
  python run_all.py --no-missforest    # pomiń MissForest (wolna metoda)

Kolejność kroków:
  1. Przygotowanie danych — complete cases + MinMaxScale [0, 1]
  2. Maskowanie 10% wartości (ta sama maska dla wszystkich metod!)
  3. Imputacja: KNN / MICE / MissForest — każda z Optuną
  4. Ewaluacja na zamaskowanych pozycjach (RMSE, MAE, KL, ujemne)
  5. Porównanie i zapis wyników do results/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Umożliwia import modułów z katalogu imputation-final
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATASETS,
    MASK_FRAC,
    RANDOM_STATE,
    RESULTS_DIR,
    N_TRIALS_KNN,
    N_TRIALS_MICE,
    N_TRIALS_MISSFOREST,
    OPTUNA_VAL_ROWS,
)
from prepare_data import load_complete
from evaluate import create_mask, evaluate_imputation

from methods.knn_optuna        import optimize_and_impute as knn_impute
from methods.mice_optuna       import optimize_and_impute as mice_impute
from methods.missforest_optuna import optimize_and_impute as mf_impute

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Porównanie metod imputacji")
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()) + ["all"], default="all",
        help="Zbiór danych do przetworzenia (domyślnie: all)",
    )
    parser.add_argument(
        "--mask-seed", type=int, default=RANDOM_STATE,
        help=f"Seed do losowania maski (domyślnie: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--no-knn",        action="store_true", help="Pomiń KNN"
    )
    parser.add_argument(
        "--no-mice",       action="store_true", help="Pomiń MICE"
    )
    parser.add_argument(
        "--no-missforest", action="store_true", help="Pomiń MissForest"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Główna logika dla jednego zbioru
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, mask_seed: int, methods_cfg: list) -> pd.DataFrame:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  DATASET: {dataset_name.upper()}  |  maska seed={mask_seed}")
    print(sep)

    data = load_complete(dataset_name)
    df_scaled = data["df_scaled"]
    columns   = data["feature_cols"]

    # Jedna maska dla wszystkich metod
    mask = create_mask(df_scaled, MASK_FRAC, mask_seed)
    n_masked = int(mask.sum())
    pct      = 100 * n_masked / mask.size
    print(f"\nZamaskowano: {n_masked:,} wartości ({pct:.1f}% macierzy)\n")

    all_results  = []
    all_params   = {}

    for method_name, impute_fn, kwargs in methods_cfg:
        print(f"--- {method_name} ---")
        t0 = time.perf_counter()

        imputed_arr, best_params = impute_fn(df_scaled, mask, **kwargs)
        elapsed = time.perf_counter() - t0

        metrics = evaluate_imputation(df_scaled, imputed_arr, mask, columns)
        metrics.update({"method": method_name, "time_s": round(elapsed, 1)})
        all_results.append(metrics)
        all_params[method_name] = best_params

        print(
            f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
            f"KL={metrics['KL_mean']:.4f}  ujemne={metrics['negatives']}  "
            f"czas={elapsed:.0f}s"
        )

    # Tabela porównawcza posortowana wg RMSE
    df_res = (
        pd.DataFrame(all_results)[["method", "RMSE", "MAE", "KL_mean", "negatives", "time_s"]]
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print(f"\n{sep}")
    print(f"  WYNIKI — {dataset_name.upper()}  (seed maska={mask_seed})")
    print(sep)
    print(df_res.to_string(index=False))

    # Najlepsza metoda
    winner = df_res.iloc[0]["method"]
    print(f"\n  => Najlepsza metoda: {winner} (RMSE={df_res.iloc[0]['RMSE']:.4f})")

    # Zapis
    prefix = RESULTS_DIR / f"{dataset_name.lower()}_seed{mask_seed}"
    df_res.to_csv(f"{prefix}_comparison.csv", index=False)
    with open(f"{prefix}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2, default=str)

    print(f"\n  Wyniki zapisane → {RESULTS_DIR}")
    return df_res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Wybór zbiorów
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    # Konfiguracja metod
    methods_cfg = []
    if not args.no_knn:
        methods_cfg.append((
            "KNN", knn_impute,
            {"n_trials": N_TRIALS_KNN, "seed": RANDOM_STATE, "val_rows": None},
        ))
    if not args.no_mice:
        methods_cfg.append((
            "MICE", mice_impute,
            {"n_trials": N_TRIALS_MICE, "seed": RANDOM_STATE, "val_rows": OPTUNA_VAL_ROWS},
        ))
    if not args.no_missforest:
        methods_cfg.append((
            "MissForest", mf_impute,
            {"n_trials": N_TRIALS_MISSFOREST, "seed": RANDOM_STATE, "val_rows": OPTUNA_VAL_ROWS},
        ))

    if not methods_cfg:
        print("Brak wybranych metod. Dodaj co najmniej jedną.")
        sys.exit(1)

    all_summaries = {}
    for name in datasets:
        df_res = run_dataset(name, args.mask_seed, methods_cfg)
        all_summaries[name] = df_res

    # Zbiorczy widok jeśli więcej niż jeden zbiór
    if len(all_summaries) > 1:
        print(f"\n{'='*62}")
        print("  PODSUMOWANIE ZBIORCZE")
        print(f"{'='*62}")
        for ds_name, df in all_summaries.items():
            winner = df.iloc[0]
            print(f"  {ds_name.upper():6s} → {winner['method']:12s} "
                  f"RMSE={winner['RMSE']:.4f}  MAE={winner['MAE']:.4f}  KL={winner['KL_mean']:.4f}")


if __name__ == "__main__":
    main()
