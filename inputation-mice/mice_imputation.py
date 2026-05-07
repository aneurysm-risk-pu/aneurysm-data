"""
MICE Imputation — aneurysm-risk-pu project
==========================================
Autor: kjubig
Gałąź: data-cleaning-lk

Skrypt wykonuje:
1. Wczytanie wyczyszczonych danych (neuro + kor)
2. Podział na train/test na poziomie patient_id (80/20)
3. Imputację MICE (IterativeImputer z sklearn) — fit tylko na train
4. Walidację: KL divergence + test KS (przed vs po imputacji)
5. Zapis wyników

Użycie:
    python mice_imputation.py

Wymagane biblioteki:
    pip install scikit-learn pandas scipy numpy
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ─── ŚCIEŻKI ─────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "datasets", "cleaned")
OUTPUT_DIR  = os.path.join(BASE_DIR, "inputation-mice", "results")

NEURO_FILE  = os.path.join(CLEANED_DIR, "neuro_cleaning.csv")
KOR_FILE    = os.path.join(CLEANED_DIR, "kor_cleaning.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── KONFIGURACJA ────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MICE_MAX_ITER = 30       # zwiększono z 10 — NEURO nie zbiegało przy 10
MICE_MIN_VALUE = 0.0     # wartości laboratoryjne nie mogą być ujemne
N_BINS_KL = 50           # liczba binów do aproksymacji KL divergence

# Kolumny meta — nie imputujemy
META_COLS = ["custom_id", "patient_id", "examination_date",
             "patient_age", "patient_sex", "diagnosis_id"]

# Kolumny flag norm — pomijamy przy imputacji (odtwarzalne z wartości)
NORM_SUFFIX = "-norm"

# ─── FUNKCJE POMOCNICZE ──────────────────────────────────────────────────────

def get_value_cols(df: pd.DataFrame) -> list[str]:
    """Zwraca kolumny do imputacji: numeryczne, bez meta i bez flag norm."""
    exclude = set(META_COLS)
    value_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.endswith(NORM_SUFFIX)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return value_cols


def kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray,
                  n_bins: int = N_BINS_KL) -> float:
    """
    Aproksymacja KL divergence KL(P || Q) przez histogramy.
    Niższa wartość = lepiej (rozkłady bardziej podobne).
    Zwraca np.inf jeśli dane są puste.
    """
    if len(p_samples) == 0 or len(q_samples) == 0:
        return np.inf

    combined = np.concatenate([p_samples, q_samples])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    # Dodajemy małą wartość epsilon żeby uniknąć log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalizacja
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(np.sum(rel_entr(p_hist, q_hist)))


def validate_imputation(before_df: pd.DataFrame, after_df: pd.DataFrame,
                        value_cols: list[str]) -> pd.DataFrame:
    """
    Porównuje rozkłady oryginalnych wartości z imputowanymi (tylko braki).

    Dla każdej kolumny:
    - before = wartości istniejące przed imputacją
    - after  = wartości wstawione w miejscach braków

    Zwraca DataFrame z metrykami walidacyjnymi.
    """
    results = []

    for col in value_cols:
        missing_mask = before_df[col].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            results.append({
                "feature":        col,
                "n_total":        len(before_df),
                "n_missing":      0,
                "missing_pct":    0.0,
                "ks_statistic":   np.nan,
                "ks_pvalue":      np.nan,
                "kl_divergence":  np.nan,
                "validation":     "brak braków"
            })
            continue

        original_vals  = before_df[col].dropna().values
        imputed_vals   = after_df.loc[missing_mask, col].values

        # Test KS
        ks_stat, ks_pval = ks_2samp(original_vals, imputed_vals)

        # KL divergence
        kl = kl_divergence(original_vals, imputed_vals)

        # Ocena: p-value > 0.05 → rozkłady statystycznie nieróżne (dobrze)
        assessment = "OK (p>0.05)" if ks_pval > 0.05 else "UWAGA (p<=0.05)"

        results.append({
            "feature":        col,
            "n_total":        len(before_df),
            "n_missing":      int(n_missing),
            "missing_pct":    round(n_missing / len(before_df) * 100, 2),
            "ks_statistic":   round(ks_stat, 6),
            "ks_pvalue":      round(ks_pval, 6),
            "kl_divergence":  round(kl, 6),
            "validation":     assessment
        })

    return pd.DataFrame(results).sort_values("kl_divergence", ascending=True)


def split_by_patient(df: pd.DataFrame, test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE):
    """Podział train/test na poziomie patient_id (nie rekordów)."""
    patients = df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    train = df[df["patient_id"].isin(train_patients)].copy()
    test  = df[df["patient_id"].isin(test_patients)].copy()
    return train, test


# ─── GŁÓWNA PROCEDURA ────────────────────────────────────────────────────────

def run_mice(name: str, filepath: str):
    """
    Pełna procedura MICE dla jednego zbioru danych (neuro lub kor).

    name     : etykieta zbioru (np. 'neuro', 'kor')
    filepath : ścieżka do pliku CSV
    """
    print(f"\n{'='*60}")
    print(f"  MICE — {name.upper()}")
    print(f"{'='*60}")

    # 1. Wczytaj dane
    df = pd.read_csv(filepath)
    print(f"Wczytano: {df.shape[0]} rekordów, {df.shape[1]} kolumn")

    value_cols = get_value_cols(df)
    print(f"Kolumny do imputacji: {len(value_cols)}")

    # Sprawdź kolumny z 100% brakami — wyłącz z imputacji
    fully_missing = [c for c in value_cols if df[c].isna().all()]
    if fully_missing:
        print(f"Pominięto (100% braków): {fully_missing}")
        value_cols = [c for c in value_cols if c not in fully_missing]

    # 2. Podział train/test na poziomie patient_id
    train, test = split_by_patient(df)
    print(f"Train: {len(train)} rekordów ({train['patient_id'].nunique()} pacjentów)")
    print(f"Test:  {len(test)} rekordów ({test['patient_id'].nunique()} pacjentów)")

    # 3. Zachowaj kolumny meta osobno
    train_meta = train[META_COLS + [c for c in df.columns
                                     if c.endswith(NORM_SUFFIX)]].copy()
    test_meta  = test[META_COLS  + [c for c in df.columns
                                     if c.endswith(NORM_SUFFIX)]].copy()

    # 4. MICE — fit TYLKO na train
    print(f"\nUruchamianie IterativeImputer (max_iter={MICE_MAX_ITER})...")
    imputer = IterativeImputer(
        max_iter=MICE_MAX_ITER,
        random_state=RANDOM_STATE,
        min_value=MICE_MIN_VALUE,
        verbose=1
    )

    imputer.fit(train[value_cols])
    print("Fit zakończony.")

    # 5. Transform
    train_imputed_arr = imputer.transform(train[value_cols])
    test_imputed_arr  = imputer.transform(test[value_cols])

    train_imputed_vals = pd.DataFrame(train_imputed_arr, columns=value_cols,
                                      index=train.index)
    test_imputed_vals  = pd.DataFrame(test_imputed_arr,  columns=value_cols,
                                      index=test.index)

    # 6. Złóż wynikowe DataFrame (meta + wartości + flagi norm)
    train_result = pd.concat([train_meta.reset_index(drop=True),
                               train_imputed_vals.reset_index(drop=True)], axis=1)
    test_result  = pd.concat([test_meta.reset_index(drop=True),
                               test_imputed_vals.reset_index(drop=True)], axis=1)

    # Przywróć oryginalną kolejność kolumn
    train_result = train_result.reindex(columns=df.columns)
    test_result  = test_result.reindex(columns=df.columns)

    # 7. Walidacja (na zbiorze treningowym)
    print("\nWalidacja imputacji (train)...")
    validation_df = validate_imputation(train[value_cols], train_imputed_vals,
                                        value_cols)

    n_ok    = (validation_df["validation"] == "OK (p>0.05)").sum()
    n_warn  = (validation_df["validation"] == "UWAGA (p<=0.05)").sum()
    n_nonan = (validation_df["validation"] == "brak braków").sum()
    print(f"  OK (p>0.05):    {n_ok}")
    print(f"  UWAGA (p≤0.05): {n_warn}")
    print(f"  Brak braków:    {n_nonan}")

    # 8. Zapis wyników
    train_out     = os.path.join(OUTPUT_DIR, f"{name}_mice_train.csv")
    test_out      = os.path.join(OUTPUT_DIR, f"{name}_mice_test.csv")
    validation_out = os.path.join(OUTPUT_DIR, f"{name}_mice_validation.csv")

    train_result.to_csv(train_out,      index=False)
    test_result.to_csv(test_out,        index=False)
    validation_df.to_csv(validation_out, index=False)

    print(f"\nZapisano:")
    print(f"  {train_out}")
    print(f"  {test_out}")
    print(f"  {validation_out}")

    return validation_df


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    val_neuro = run_mice("neuro", NEURO_FILE)
    val_kor   = run_mice("kor",   KOR_FILE)

    print("\n" + "="*60)
    print("  PODSUMOWANIE WALIDACJI — TOP 10 (najwyższe KL divergence)")
    print("="*60)

    for name, val_df in [("NEURO", val_neuro), ("KOR", val_kor)]:
        print(f"\n{name}:")
        top = val_df[val_df["kl_divergence"].notna()].nlargest(10, "kl_divergence")
        print(top[["feature", "missing_pct", "ks_pvalue",
                   "kl_divergence", "validation"]].to_string(index=False))

    print("\nGotowe.")
