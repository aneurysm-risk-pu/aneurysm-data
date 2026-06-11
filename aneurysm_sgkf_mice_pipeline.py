"""
aneurysm_sgkf_mice_pipeline.py
================================

Podział StratifiedGroupKFold + imputacja MICE wewnątrz pętli foldów.

Metodologia (zapobieganie data leakage):
  1. Split na train/test (StratifiedGroupKFold, n_splits=5)
  2. MinMaxScaler fitowany TYLKO na wierszach train bez braków (complete cases)
  3. IterativeImputer (MICE) fitowany TYLKO na X_train po skalowaniu
  4. X_test transformowany imputer.transform() — nigdy fit_transform()
  5. Inwersja skalera → oryginalna skala na wyjściu

Parametry MICE (finalne, z imputation-final/RAPORT_IMPUTACJA.md):
  ExtraTrees, max_iter=7, n_estimators=78, max_depth=10
  min_value=0.0, max_value=1.0 (dane są w [0,1] po skalowaniu)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
INPUT_CSV  = BASE_DIR / "aneurysm_concatted_cleaned.csv"

META_COLS    = ["patient_id", "custom_id", "examination_date", "label"]
TARGET_COL   = "label"
GROUP_COL    = "patient_id"

N_SPLITS     = 5
RANDOM_STATE = 42

# Finalne parametry MICE (imputation-final/RAPORT_IMPUTACJA.md, przebieg 4 + cross-param)
MICE_PARAMS = dict(
    max_iter     = 7,
    n_estimators = 78,
    max_depth    = 10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_mice_imputer(seed: int = RANDOM_STATE) -> IterativeImputer:
    estimator = ExtraTreesRegressor(
        n_estimators = MICE_PARAMS["n_estimators"],
        max_depth    = MICE_PARAMS["max_depth"],
        random_state = seed,
        n_jobs       = -1,
    )
    return IterativeImputer(
        estimator    = estimator,
        max_iter     = MICE_PARAMS["max_iter"],
        min_value    = 0.0,   # dane są w [0,1] po MinMaxScaler
        max_value    = 1.0,
        random_state = seed,
        verbose      = 0,
    )


def prepare_fold(
    X_train_raw: pd.DataFrame,
    X_test_raw:  pd.DataFrame,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Skaluje i imputuje jeden fold.

    Zwraca
    -------
    X_train_imp, X_test_imp : ndarray w oryginalnej skali (po inwersji scalera)
    """
    feat_cols = X_train_raw.columns.tolist()

    # --- Scaler fitowany na complete cases z train ---
    train_complete = X_train_raw.dropna()
    scaler = MinMaxScaler()
    scaler.fit(train_complete.values.astype(float))

    X_train_scaled = scaler.transform(X_train_raw.values.astype(float))
    X_test_scaled  = scaler.transform(X_test_raw.values.astype(float))

    # Przywróć NaN (transform wypełnia NaN → NaN, ale upewnijmy się)
    train_nan_mask = X_train_raw.isna().values
    test_nan_mask  = X_test_raw.isna().values
    X_train_scaled[train_nan_mask] = np.nan
    X_test_scaled[test_nan_mask]   = np.nan

    # --- MICE: fit na train, transform na test ---
    imputer = build_mice_imputer(seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train_imp_scaled = imputer.fit_transform(X_train_scaled)
        X_test_imp_scaled  = imputer.transform(X_test_scaled)

    # --- Inwersja skalera → oryginalna skala ---
    X_train_imp = scaler.inverse_transform(X_train_imp_scaled)
    X_test_imp  = scaler.inverse_transform(X_test_imp_scaled)

    return X_train_imp, X_test_imp


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------

def main():
    print(f"[1] Wczytywanie {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Kształt: {df.shape}   |   label=0: {(df[TARGET_COL]==0).sum():,}   label=1: {(df[TARGET_COL]==1).sum():,}")

    meta_present = [c for c in META_COLS if c in df.columns]
    feat_cols    = [c for c in df.columns if c not in META_COLS]
    print(f"    Kolumny cech: {len(feat_cols)}")

    X      = df[feat_cols]
    y      = df[TARGET_COL]
    groups = df[GROUP_COL]

    missing_pct = X.isna().sum().sum() / X.size * 100
    print(f"    Braki w cechach: {X.isna().sum().sum():,} ({missing_pct:.1f}%)\n")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)

    splits = []   # lista słowników — gotowe do przekazania do modelu

    print(f"[2] Podział StratifiedGroupKFold (n_splits={N_SPLITS}) + MICE ...\n")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"  Fold {fold + 1}/{N_SPLITS} — imputacja MICE ...")

        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw  = X.iloc[test_idx].reset_index(drop=True)
        y_train     = y.iloc[train_idx].reset_index(drop=True)
        y_test      = y.iloc[test_idx].reset_index(drop=True)
        g_train     = groups.iloc[train_idx].reset_index(drop=True)
        g_test      = groups.iloc[test_idx].reset_index(drop=True)

        # Weryfikacja: brak wspólnych pacjentów
        overlap = set(g_train).intersection(set(g_test))
        assert len(overlap) == 0, f"Fold {fold+1}: pacjenci w obu zbiorach! {overlap}"

        # Imputacja
        X_train_imp, X_test_imp = prepare_fold(X_train_raw, X_test_raw, seed=RANDOM_STATE)

        # Weryfikacja: brak NaN po imputacji
        assert not np.isnan(X_train_imp).any(), f"Fold {fold+1}: NaN w X_train po imputacji!"
        assert not np.isnan(X_test_imp).any(),  f"Fold {fold+1}: NaN w X_test po imputacji!"

        # Proporcje klas
        train_pos = y_train.mean()
        test_pos  = y_test.mean()

        print(f"    Rozmiar: train={len(train_idx):,}  test={len(test_idx):,}")
        print(f"    Proporcja label=1: train={train_pos:.2%}  test={test_pos:.2%}")
        print(f"    Wspólne patient_id: {len(overlap)}  (oczekiwane: 0)")
        print(f"    NaN po imputacji:   train={np.isnan(X_train_imp).sum()}  test={np.isnan(X_test_imp).sum()}")
        print()

        splits.append({
            "fold":        fold + 1,
            "X_train":     pd.DataFrame(X_train_imp, columns=feat_cols),
            "X_test":      pd.DataFrame(X_test_imp,  columns=feat_cols),
            "y_train":     y_train,
            "y_test":      y_test,
            "groups_train": g_train,
            "groups_test":  g_test,
        })

    print("[3] Gotowe. Zwrócono listę `splits` z kluczami:")
    print("    X_train, X_test  — ndarray/DataFrame, zaimputowane, oryginalna skala")
    print("    y_train, y_test  — Series z labelami")
    print("    groups_train, groups_test — patient_id")

    return splits


if __name__ == "__main__":
    splits = main()
