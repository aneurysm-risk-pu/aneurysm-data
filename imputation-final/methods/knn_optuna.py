"""
KNN Imputer z optymalizacją hiperparametrów przez Optunę.

Przestrzeń poszukiwań:
  n_neighbors : int [1, 30]
  weights     : categorical ['uniform', 'distance']

Metodologia:
  Dane wejściowe są już w skali [0, 1] (MinMaxScaler z prepare_data).
  KNN jako uśrednianie sąsiadów naturalnie produkuje wartości w [0, 1],
  więc clipping nie jest potrzebny.
"""
import numpy as np
import optuna
from sklearn.impute import KNNImputer

from evaluate import apply_mask, compute_rmse


def optimize_and_impute(
    df_scaled,
    mask: np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
    val_rows: int = None,
) -> tuple:
    """
    Optymalizuje hiperparametry KNN przez Optunę, następnie imputuje
    pełny zbiór najlepszym modelem.

    Zwraca
    -------
    (imputed_arr : ndarray, best_params : dict)
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pełna zamaskowana macierz do finalnej imputacji
    df_masked_full = apply_mask(df_scaled, mask)
    orig_full = df_scaled.values

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
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputed = imputer.fit_transform(df_val_masked)
        return compute_rmse(orig_val, imputed, mask_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"  KNN best: n_neighbors={best_params['n_neighbors']}, "
          f"weights={best_params['weights']}  "
          f"(val RMSE={study.best_value:.4f})")

    # Finalna imputacja na pełnym zbiorze
    final_imputer = KNNImputer(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
    )
    imputed_arr = final_imputer.fit_transform(df_masked_full)

    return imputed_arr, best_params
