"""
MICE (IterativeImputer) z optymalizacją hiperparametrów przez Optunę.

Metodologia A (ujednolicona z mice_compare_methodologies.py):
  Dane wejściowe są już w skali [0, 1] (MinMaxScaler z prepare_data).
  IterativeImputer z min_value=0.0, max_value=1.0 → brak wartości spoza zakresu.

Przestrzeń poszukiwań:
  estimator_type : categorical ['BayesianRidge', 'ExtraTrees']
  max_iter       : int [5, 40]
  n_estimators   : int [10, 100]  — tylko dla ExtraTrees
"""
import warnings

import numpy as np
import optuna
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor

from evaluate import apply_mask, compute_rmse


def _build_estimator(estimator_type: str, n_estimators: int, seed: int):
    if estimator_type == "BayesianRidge":
        return BayesianRidge()
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=10,          # ograniczenie głębokości — zapobiega overfittingowi
        random_state=seed,
        n_jobs=-1,
    )


def _build_imputer(estimator_type: str, n_estimators: int, max_iter: int, seed: int):
    estimator = _build_estimator(estimator_type, n_estimators, seed)
    return IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        min_value=0.0,
        max_value=1.0,
        random_state=seed,
        verbose=0,
    )


def optimize_and_impute(
    df_scaled,
    mask: np.ndarray,
    n_trials: int = 20,
    seed: int = 42,
    val_rows: int = None,
) -> tuple:
    """
    Optymalizuje hiperparametry MICE przez Optunę, następnie imputuje
    pełny zbiór najlepszym modelem.

    Zwraca
    -------
    (imputed_arr : ndarray, best_params : dict)
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pełna zamaskowana macierz do finalnej imputacji (oryginalna maska)
    df_masked_full = apply_mask(df_scaled, mask)

    # Osobna maska do trialu Optuna — unikamy leakage parametrów
    from evaluate import create_mask
    mask_val_full = create_mask(df_scaled, frac=mask.mean(), seed=seed + 1)

    # Podzbiór do szybkiej walidacji w trialu Optuna
    if val_rows and len(df_scaled) > val_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df_scaled), size=val_rows, replace=False)
        df_val = df_scaled.iloc[idx].reset_index(drop=True)
        mask_val = mask_val_full[idx]
    else:
        df_val = df_scaled
        mask_val = mask_val_full

    df_val_masked = apply_mask(df_val, mask_val)
    orig_val = df_val.values

    def objective(trial: optuna.Trial) -> float:
        estimator_type = trial.suggest_categorical(
            "estimator_type", ["BayesianRidge", "ExtraTrees"]
        )
        max_iter = trial.suggest_int("max_iter", 5, 60)

        # n_estimators tylko dla ExtraTrees — conditional hyperparameter
        if estimator_type == "ExtraTrees":
            n_estimators = trial.suggest_int("n_estimators", 10, 100)
        else:
            n_estimators = 10  # ignorowane dla BayesianRidge

        imputer = _build_imputer(estimator_type, n_estimators, max_iter, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed = imputer.fit_transform(df_val_masked)

        return compute_rmse(orig_val, imputed, mask_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    # Uzupełnij n_estimators jeśli BayesianRidge (nie było w trial)
    best.setdefault("n_estimators", 10)

    print(f"  MICE best: estimator={best['estimator_type']}, "
          f"max_iter={best['max_iter']}"
          + (f", n_estimators={best['n_estimators']}"
             if best["estimator_type"] == "ExtraTrees" else "")
          + f"  (val RMSE={study.best_value:.4f})")

    # Finalna imputacja na pełnym zbiorze
    final_imputer = _build_imputer(
        best["estimator_type"], best["n_estimators"], best["max_iter"], seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed_arr = final_imputer.fit_transform(df_masked_full)

    return imputed_arr, best
