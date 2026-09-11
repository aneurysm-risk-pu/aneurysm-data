"""
Walidacja stabilności MICE — dwa eksperymenty:

  A) Multi-seed: najlepsze params testowane na 10 różnych maskach
     → sprawdza czy params są generalne czy przefitowane pod jedną maskę

  B) Cross-param: params z KOR testowane na NEURO i odwrotnie
     → sprawdza czy warto mieć osobne params dla każdego zbioru

Źródło hiperparametrów (w kolejności priorytetu):
  1. results/mice_extended_optuna.csv  — jeśli istnieje (po mice_extended_optuna.py)
  2. results/kor_seed42_best_params.json / neuro_seed42_best_params.json  — fallback
  Skrypt drukuje które źródło jest używane.

Użycie:
    python validate_mice.py
    python validate_mice.py --seeds 0 1 2 3 4 5 6 7 8 9   (domyślne)
    python validate_mice.py --skip-cross                   (tylko multi-seed)
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

# Dodaj katalog imputation-final do ścieżki
sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_complete
from evaluate import create_mask, apply_mask, compute_rmse, compute_mae, compute_kl_mean
from config import MASK_FRAC, RANDOM_STATE, RESULTS_DIR

MAX_DEPTH_FIXED = 10  # spójne z mice_optuna.py i mice_extended_optuna.py

# ---------------------------------------------------------------------------
# Fallback — params z przebiegu seed=42 (run_all.py, 20 trials, n_est max=100)
# Używane tylko gdy nie ma mice_extended_optuna.csv
# ---------------------------------------------------------------------------
_FALLBACK_PARAMS = {
    "kor": {
        "estimator_type": "ExtraTrees",
        "max_iter":       23,
        "n_estimators":   54,
        "max_depth":      MAX_DEPTH_FIXED,
    },
    "neuro": {
        "estimator_type": "ExtraTrees",
        "max_iter":       18,
        "n_estimators":   100,  # ⚠ sufit poprzedniego zakresu — nieznane optimum
        "max_depth":      MAX_DEPTH_FIXED,
    },
}


def _load_params_from_extended_optuna() -> dict | None:
    """Czyta params z mice_extended_optuna.csv (wynik mice_extended_optuna.py)."""
    csv_path = RESULTS_DIR / "mice_extended_optuna.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        ds = row["dataset"]
        result[ds] = {
            "estimator_type": row["estimator_type"],
            "max_iter":       int(row["max_iter"]),
            "n_estimators":   int(row["n_estimators"]),
            "max_depth":      MAX_DEPTH_FIXED,
        }
    return result if result else None


def _load_params_from_best_json() -> dict | None:
    """Czyta params z *_seed42_best_params.json (wynik run_all.py)."""
    result = {}
    for ds in ["kor", "neuro"]:
        json_path = RESULTS_DIR / f"{ds}_seed42_best_params.json"
        if not json_path.exists():
            return None
        with open(json_path) as f:
            data = json.load(f)
        mice = data.get("MICE", {})
        if not mice:
            return None
        result[ds] = {
            "estimator_type": mice.get("estimator_type", "ExtraTrees"),
            "max_iter":       int(mice.get("max_iter", 20)),
            "n_estimators":   int(mice.get("n_estimators", 50)),
            "max_depth":      MAX_DEPTH_FIXED,
        }
    return result


def load_best_params() -> dict:
    """Ładuje hiperparametry z najlepszego dostępnego źródła."""
    # 1. Rozszerzona Optuna (preferowane)
    params = _load_params_from_extended_optuna()
    if params:
        print("[PARAMS] Źródło: results/mice_extended_optuna.csv "
              "(rozszerzona Optuna, n_estimators do 200)")
        return params

    # 2. JSON z run_all.py
    params = _load_params_from_best_json()
    if params:
        print("[PARAMS] Źródło: *_seed42_best_params.json (run_all.py, 20 trials)")
        return params

    # 3. Fallback zahardkodowany
    print("[PARAMS] Źródło: fallback zahardkodowany "
          "(UWAGA: NEURO n_estimators=100 może być suboptymalne — "
          "uruchom najpierw mice_extended_optuna.py)")
    return _FALLBACK_PARAMS


def build_imputer(params: dict, seed: int) -> IterativeImputer:
    if params["estimator_type"] == "BayesianRidge":
        estimator = BayesianRidge()
    else:
        estimator = ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=seed,
            n_jobs=-1,
        )
    return IterativeImputer(
        estimator=estimator,
        max_iter=params["max_iter"],
        min_value=0.0,
        max_value=1.0,
        random_state=seed,
        verbose=0,
    )


def run_single(data: dict, params: dict, mask_seed: int) -> dict:
    df_scaled = data["df_scaled"]
    mask = create_mask(df_scaled, frac=MASK_FRAC, seed=mask_seed)
    df_masked = apply_mask(df_scaled, mask)

    imputer = build_imputer(params, seed=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imputer.fit_transform(df_masked)

    df_imputed = pd.DataFrame(imputed, columns=df_scaled.columns)

    rmse = compute_rmse(df_scaled.values, imputed, mask)
    mae  = compute_mae(df_scaled.values, imputed, mask)
    kl   = compute_kl_mean(df_scaled, df_imputed)
    return {"rmse": rmse, "mae": mae, "kl": kl}


# ---------------------------------------------------------------------------
# A) Multi-seed validation
# ---------------------------------------------------------------------------
def multi_seed_experiment(seeds: list[int], best_params: dict) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("EKSPERYMENT A — Multi-seed (sprawdzenie generalności params)")
    print("=" * 65)

    rows = []
    for ds_name in ["kor", "neuro"]:
        print(f"\n[{ds_name.upper()}] Ładowanie danych...")
        data   = load_complete(ds_name)
        params = best_params[ds_name]
        print(f"[{ds_name.upper()}] Params: max_iter={params['max_iter']}, "
              f"n_estimators={params['n_estimators']}")
        print(f"[{ds_name.upper()}] Uruchamiam {len(seeds)} seedów: {seeds}")

        seed_results = []
        for s in seeds:
            t0 = time.time()
            r  = run_single(data, params, mask_seed=s)
            elapsed = time.time() - t0
            seed_results.append(r)
            print(f"  seed={s:>3}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  "
                  f"KL={r['kl']:.5f}  ({elapsed:.0f}s)")

        rmse_vals = [r["rmse"] for r in seed_results]
        mae_vals  = [r["mae"]  for r in seed_results]
        kl_vals   = [r["kl"]   for r in seed_results]

        print(f"\n  >>> RMSE: mean={np.mean(rmse_vals):.4f}  "
              f"std={np.std(rmse_vals):.4f}  "
              f"min={np.min(rmse_vals):.4f}  max={np.max(rmse_vals):.4f}")
        print(f"  >>> MAE:  mean={np.mean(mae_vals):.4f}  std={np.std(mae_vals):.4f}")
        print(f"  >>> KL:   mean={np.mean(kl_vals):.5f}  std={np.std(kl_vals):.5f}")

        for i, s in enumerate(seeds):
            rows.append({
                "experiment":   "multi-seed",
                "dataset":       ds_name,
                "params_from":   ds_name,
                "mask_seed":     s,
                "rmse":          seed_results[i]["rmse"],
                "mae":           seed_results[i]["mae"],
                "kl":            seed_results[i]["kl"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B) Cross-param experiment
# ---------------------------------------------------------------------------
def cross_param_experiment(best_params: dict) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("EKSPERYMENT B — Cross-param (czy jeden zestaw params dla obu?)")
    print("=" * 65)

    rows = []
    # Załaduj oba zbiory raz
    datasets = {n: load_complete(n) for n in ["kor", "neuro"]}

    for ds_name, data in datasets.items():
        for param_source, params in best_params.items():
            label = f"{ds_name} / params_from={param_source}"
            print(f"\n  {label}")
            print(f"    max_iter={params['max_iter']}, n_estimators={params['n_estimators']}")
            t0 = time.time()
            r  = run_single(data, params, mask_seed=RANDOM_STATE)
            elapsed = time.time() - t0
            print(f"    RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  "
                  f"KL={r['kl']:.5f}  ({elapsed:.0f}s)")
            rows.append({
                "experiment":   "cross-param",
                "dataset":       ds_name,
                "params_from":   param_source,
                "mask_seed":     RANDOM_STATE,
                "rmse":          r["rmse"],
                "mae":           r["mae"],
                "kl":            r["kl"],
            })

    # Podsumowanie — porównanie baseline vs cross
    df = pd.DataFrame(rows)
    print("\n  Podsumowanie cross-param:")
    print(f"  {'Dataset':<8} {'Params from':<14} {'RMSE':>8} {'vs baseline':>12}")
    for ds_name in ["kor", "neuro"]:
        sub = df[df["dataset"] == ds_name]
        baseline = sub[sub["params_from"] == ds_name]["rmse"].values[0]
        cross    = sub[sub["params_from"] != ds_name]["rmse"].values[0]
        diff     = cross - baseline
        print(f"  {ds_name:<8} {ds_name + ' (baseline)':<14} {baseline:>8.4f} {'':>12}")
        cross_src = "neuro" if ds_name == "kor" else "kor"
        delta_str = f"{'+' if diff >= 0 else ''}{diff:.4f}"
        print(f"  {ds_name:<8} {cross_src + ' (cross)':<14} {cross:>8.4f} {delta_str:>12}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Walidacja MICE: multi-seed + cross-param")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)),
                        help="Seedów do multi-seed (default: 0..9)")
    parser.add_argument("--skip-multi",  action="store_true",
                        help="Pomiń eksperyment A (multi-seed)")
    parser.add_argument("--skip-cross",  action="store_true",
                        help="Pomiń eksperyment B (cross-param)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    best_params = load_best_params()
    print(f"  KOR:   max_iter={best_params['kor']['max_iter']}, "
          f"n_estimators={best_params['kor']['n_estimators']}")
    print(f"  NEURO: max_iter={best_params['neuro']['max_iter']}, "
          f"n_estimators={best_params['neuro']['n_estimators']}")

    all_rows = []

    if not args.skip_multi:
        df_multi = multi_seed_experiment(args.seeds, best_params)
        all_rows.append(df_multi)

    if not args.skip_cross:
        df_cross = cross_param_experiment(best_params)
        all_rows.append(df_cross)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        out_path = RESULTS_DIR / "mice_validation.csv"
        df_all.to_csv(out_path, index=False)
        print(f"\n[ZAPISANO] {out_path}")


if __name__ == "__main__":
    main()
