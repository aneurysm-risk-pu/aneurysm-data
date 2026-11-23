"""
Rozszerzona optymalizacja Optuna dla MICE — większa przestrzeń poszukiwań.

Problem z poprzednim przebiegiem:
  - KOR:   n_estimators=54  (poniżej sufitu 100) — OK
  - NEURO: n_estimators=100 (sufit!)  → może potrzebować więcej drzew
  - Różnica max_iter: KOR=23 vs NEURO=18 — sprawdzamy czy wynikają z danych
    czy tylko z losowości 20 trials

Zmiany względem domyślnego run_all.py:
  - n_estimators: [10, 200]   (było [10, 100])  — eliminuje sufit NEURO
  - max_iter:     [5, 80]     (było [5, 60])
  - n_trials:     40          (było 20)          — dokładniejsze przeszukanie
  - Optuna mask seed = RANDOM_STATE + 1 = 43 (brak leakage)
  - Ewaluacja mask seed = RANDOM_STATE = 42

Użycie:
    python mice_extended_optuna.py
    python mice_extended_optuna.py --dataset neuro    # tylko NEURO
    python mice_extended_optuna.py --trials 60        # więcej trials
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_complete
from evaluate import create_mask, apply_mask, compute_rmse, compute_mae, compute_kl_mean
from config import MASK_FRAC, RANDOM_STATE, RESULTS_DIR, OPTUNA_VAL_ROWS

# ---------------------------------------------------------------------------
# Rozszerzona przestrzeń poszukiwań
# ---------------------------------------------------------------------------
N_ESTIMATORS_MAX = 200   # poprzednio 100
MAX_ITER_MAX     = 80    # poprzednio 60
MAX_DEPTH_FIXED  = 10    # bez zmian
N_TRIALS_DEFAULT = 40    # poprzednio 20


def _build_imputer(estimator_type: str, n_estimators: int, max_iter: int, seed: int):
    if estimator_type == "BayesianRidge":
        estimator = BayesianRidge()
    else:
        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=MAX_DEPTH_FIXED,
            random_state=seed,
            n_jobs=-1,
        )
    return IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        min_value=0.0,
        max_value=1.0,
        random_state=seed,
        verbose=0,
    )


def optimize_mice(data: dict, n_trials: int) -> dict:
    """Uruchamia Optunę i zwraca best_params + finalne metryki."""
    ds_name  = data["name"]
    df_scaled = data["df_scaled"]

    # Maska ewaluacyjna (seed=42) — do finalnej oceny
    mask_eval = create_mask(df_scaled, frac=MASK_FRAC, seed=RANDOM_STATE)

    # Maska Optuna (seed=43) — Optuna jej nie widzi przy ewaluacji
    mask_optuna = create_mask(df_scaled, frac=MASK_FRAC, seed=RANDOM_STATE + 1)

    # Podzbiór walidacyjny (dla KOR — przyspiesza trial bez zmiany metodologii)
    n = len(df_scaled)
    if OPTUNA_VAL_ROWS and n > OPTUNA_VAL_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=OPTUNA_VAL_ROWS, replace=False)
        df_val    = df_scaled.iloc[idx].reset_index(drop=True)
        mask_val  = mask_optuna[idx]
    else:
        df_val   = df_scaled
        mask_val = mask_optuna

    df_val_masked = apply_mask(df_val, mask_val)
    orig_val      = df_val.values

    def objective(trial: optuna.Trial) -> float:
        estimator_type = trial.suggest_categorical(
            "estimator_type", ["BayesianRidge", "ExtraTrees"]
        )
        max_iter = trial.suggest_int("max_iter", 5, MAX_ITER_MAX)
        n_estimators = (
            trial.suggest_int("n_estimators", 10, N_ESTIMATORS_MAX)
            if estimator_type == "ExtraTrees"
            else 10
        )
        imputer = _build_imputer(estimator_type, n_estimators, max_iter, RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed = imputer.fit_transform(df_val_masked)
        return compute_rmse(orig_val, imputed, mask_val)

    # --- Optuna ---
    print(f"\n[{ds_name.upper()}] Optuna MICE — {n_trials} trials "
          f"(n_estimators max={N_ESTIMATORS_MAX}, max_iter max={MAX_ITER_MAX})")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    t_start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t_start

    bp = study.best_params
    best_type  = bp["estimator_type"]
    best_iter  = bp["max_iter"]
    best_trees = bp.get("n_estimators", 10)

    print(f"[{ds_name.upper()}] Najlepszy trial: RMSE(val)={study.best_value:.4f}")
    print(f"[{ds_name.upper()}] Params: estimator={best_type}, "
          f"max_iter={best_iter}, n_estimators={best_trees}")

    # Ostrzeżenia o sufitach
    if best_type == "ExtraTrees" and best_trees >= N_ESTIMATORS_MAX:
        print(f"  ⚠️  n_estimators={best_trees} trafił w sufit "
              f"[10, {N_ESTIMATORS_MAX}] — rozważ dalsze rozszerzenie")
    if best_iter >= MAX_ITER_MAX:
        print(f"  ⚠️  max_iter={best_iter} trafił w sufit "
              f"[5, {MAX_ITER_MAX}]")

    # --- Finalna ewaluacja (maska seed=42, pełny zbiór) ---
    print(f"[{ds_name.upper()}] Finalna ewaluacja (maska seed={RANDOM_STATE}, pełny zbiór)...")
    df_eval_masked = apply_mask(df_scaled, mask_eval)
    imputer_final  = _build_imputer(best_type, best_trees, best_iter, RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0      = time.time()
        imputed = imputer_final.fit_transform(df_eval_masked)
        t_imp   = time.time() - t0

    df_imp = pd.DataFrame(imputed, columns=df_scaled.columns)
    rmse   = compute_rmse(df_scaled.values, imputed, mask_eval)
    mae    = compute_mae(df_scaled.values, imputed, mask_eval)
    kl     = compute_kl_mean(df_scaled, df_imp)

    print(f"[{ds_name.upper()}] Finalne: RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"KL={kl:.5f}  (imputacja: {t_imp:.0f}s, łącznie: {elapsed:.0f}s)")

    return {
        "dataset":       ds_name,
        "estimator_type": best_type,
        "max_iter":       best_iter,
        "n_estimators":   best_trees,
        "max_depth":      MAX_DEPTH_FIXED,
        "val_rmse":       study.best_value,
        "final_rmse":     rmse,
        "final_mae":      mae,
        "final_kl":       kl,
        "n_trials":       n_trials,
        "optuna_time_s":  elapsed,
    }


def compare_with_previous(results: list[dict]):
    """Drukuje porównanie z poprzednimi wynikami (seed=42, 20 trials)."""
    prev = {
        "kor":   {"rmse": 0.0865, "max_iter": 23, "n_estimators":  54},
        "neuro": {"rmse": 0.0978, "max_iter": 18, "n_estimators": 100},
    }
    print("\n" + "=" * 65)
    print("PORÓWNANIE z poprzednim przebiegiem (20 trials, n_est max=100)")
    print("=" * 65)
    print(f"{'Dataset':<8} {'Metryka':<14} {'Poprzedni':>10} {'Nowy':>10} {'Δ':>10}")
    print("-" * 55)
    for r in results:
        ds = r["dataset"]
        p  = prev.get(ds, {})
        for field, label in [("final_rmse", "RMSE"), ("max_iter", "max_iter"),
                              ("n_estimators", "n_estimators")]:
            old = p.get(field.replace("final_", ""), None)
            if old is None:
                continue
            new = r[field]
            diff = (new - old) if isinstance(new, (int, float)) else "—"
            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else str(diff)
            print(f"  {ds:<6} {label:<14} {old:>10} {new:>10} {diff_str:>10}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Rozszerzona Optuna dla MICE (n_estimators do 200, max_iter do 80)"
    )
    parser.add_argument("--dataset", choices=["kor", "neuro", "all"],
                        default="all", help="Który zbiór (default: all)")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Liczba trials Optuna (default: {N_TRIALS_DEFAULT})")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    datasets = ["kor", "neuro"] if args.dataset == "all" else [args.dataset]

    results = []
    for ds_name in datasets:
        data = load_complete(ds_name)
        r    = optimize_mice(data, n_trials=args.trials)
        results.append(r)

    # Zapis wyników
    df_out = pd.DataFrame(results)
    out_path = RESULTS_DIR / "mice_extended_optuna.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[ZAPISANO] {out_path}")
    print(df_out.to_string(index=False))

    compare_with_previous(results)


if __name__ == "__main__":
    main()
