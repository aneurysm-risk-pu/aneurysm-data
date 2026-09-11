"""
Wspólne funkcje ewaluacji imputacji.

Wszystkie metryki liczone są w skali [0, 1] (po MinMaxScaler),
co zapewnia porównywalność między metodami i zbiorami.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy


# ---------------------------------------------------------------------------
# Maskowanie
# ---------------------------------------------------------------------------

def create_mask(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """
    Tworzy boolowską macierz maski: True = wartość zamaskowana.
    Losowanie globalne (~frac * n_cells wartości = NaN).
    """
    rng = np.random.default_rng(seed)
    return rng.random(df.shape) < frac


def apply_mask(df_scaled: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Zwraca kopię DataFrame z NaN w zamaskowanych pozycjach."""
    arr = df_scaled.values.astype(float).copy()
    arr[mask] = np.nan
    return pd.DataFrame(arr, columns=df_scaled.columns)


# ---------------------------------------------------------------------------
# Metryki punktowe
# ---------------------------------------------------------------------------

def compute_rmse(original: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> float:
    o = original[mask]
    i = imputed[mask]
    return float(np.sqrt(np.mean((o - i) ** 2)))


def compute_mae(original: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> float:
    o = original[mask]
    i = imputed[mask]
    return float(np.mean(np.abs(o - i)))


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------

def _kl_col(p_vals: np.ndarray, q_vals: np.ndarray, bins: int = 30) -> float:
    """KL(P||Q) — histogram-based, epsilon-smoothed."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    eps = 1e-10
    P, _ = np.histogram(p_vals, bins=edges, density=True)
    Q, _ = np.histogram(q_vals, bins=edges, density=True)
    P = P + eps
    Q = Q + eps
    return float(entropy(P / P.sum(), Q / Q.sum()))


def compute_kl_mean(df_original: pd.DataFrame, df_imputed: pd.DataFrame, bins: int = 30) -> float:
    """Średnie KL divergence po wszystkich kolumnach."""
    return float(np.mean([
        _kl_col(df_original[col].values, df_imputed[col].values, bins)
        for col in df_original.columns
    ]))


# ---------------------------------------------------------------------------
# Sprawdzenie wartości ujemnych (po inverse_transform lub bezpośrednio)
# ---------------------------------------------------------------------------

def count_negatives(arr: np.ndarray) -> int:
    """Liczba wartości < 0 w tablicy (po imputacji na danych [0,1])."""
    return int((arr < 0).sum())


# ---------------------------------------------------------------------------
# Pełna ewaluacja
# ---------------------------------------------------------------------------

def evaluate_imputation(
    df_original: pd.DataFrame,
    imputed_arr: np.ndarray,
    mask: np.ndarray,
    columns: list,
) -> dict:
    """
    Oblicza wszystkie metryki dla danej imputacji.

    Parametry
    ----------
    df_original : DataFrame ze skalowanymi oryginałami [0, 1]
    imputed_arr : ndarray po imputacji (ten sam kształt)
    mask        : bool ndarray — True = pozycje oceniane
    columns     : nazwy kolumn

    Zwraca
    -------
    dict z kluczami: RMSE, MAE, KL_mean, negatives
    """
    df_imputed = pd.DataFrame(imputed_arr, columns=columns)
    orig_arr = df_original.values

    return {
        "RMSE":      compute_rmse(orig_arr, imputed_arr, mask),
        "MAE":       compute_mae(orig_arr, imputed_arr, mask),
        "KL_mean":   compute_kl_mean(df_original, df_imputed),
        "negatives": count_negatives(imputed_arr),
    }
