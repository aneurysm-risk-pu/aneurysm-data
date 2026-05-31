"""
MissForest (IterativeImputer + RandomForestRegressor)
z optymalizacją hiperparametrów przez Optunę.

Dane wejściowe są już w skali [0, 1] (MinMaxScaler z prepare_data).
IterativeImputer z min_value=0.0, max_value=1.0 → brak wartości spoza zakresu.

Przestrzeń poszukiwań:
  n_estimators : int [10, 150]   — liczba drzew w RF
  max_depth    : int [5, 25]     — głębokość drzew
  max_iter     : int [3, 8]      — liczba rund IterativeImputer
"""
import warnings

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from evaluate import apply_mask, compute_rmse


def _build_imputer(n_estimators: int, max_depth: int, max_iter: int, seed: int):
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        max_iter=max_iter,
        min_value=0.0,
        max_value=1.0,
        random_state=seed,
    )


def optimize_and_impute(
    df_scaled,
    mask: np.ndarray,
    n_trials: int = 15,
    seed: int = 42,
    val_rows: int = None,
) -> tuple:
    """
    Optymalizuje hiperparametry MissForest przez Optunę, następnie imputuje
    pełny zbiór najlepszym modelem.

    Zwraca
    -------
    (imputed_arr : ndarray, best_params : dict)
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_masked_full = apply_mask(df_scaled, mask)

    # Podzbiór do szybkiej walidacji w trialu Optuna
    if val_rows and len(df_scaled) > val_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df_scaled), size=val_rows, replace=False)
        df_val = df_scaled.iloc[idx].reset_index(drop=True)
        mask_val = mask[idx]
    else:
        df_val = df_scaled
        mask_val = mask

    df_val_masked = apply_mask(df_val, mask_val)
    orig_val = df_val.values

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 10, 150)
        max_depth    = trial.suggest_int("max_depth",    5,  25)
        max_iter     = trial.suggest_int("max_iter",     3,  8)

        imputer = _build_imputer(n_estimators, max_depth, max_iter, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed = imputer.fit_transform(df_val_masked)

        return compute_rmse(orig_val, imputed, mask_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  MissForest best: n_estimators={best['n_estimators']}, "
          f"max_depth={best['max_depth']}, max_iter={best['max_iter']}  "
          f"(val RMSE={study.best_value:.4f})")

    # Finalna imputacja na pełnym zbiorze
    final_imputer = _build_imputer(
        best["n_estimators"], best["max_depth"], best["max_iter"], seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed_arr = final_imputer.fit_transform(df_masked_full)

    return imputed_arr, best
