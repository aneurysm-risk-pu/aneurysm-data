"""
Wspólna konfiguracja — imputation-final
"""
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent   # c:\...\PU DATASET
RESULTS_DIR = Path(__file__).parent / "results"

DATASETS = {
    "kor":   BASE_DIR / "kor_shortend.csv",
    "neuro": BASE_DIR / "neuro_shortend.csv",
}

# Kolumny identyfikatorów / dat do usunięcia przed imputacją
DROP_COLS = ["patient_id", "custom_id", "examination_date"]

# Ułamek wartości maskowanych do ewaluacji (10%)
MASK_FRAC    = 0.10
RANDOM_STATE = 42

# Budżety Optuna (liczba trials)
N_TRIALS_KNN        = 30
N_TRIALS_MICE       = 20
N_TRIALS_MISSFOREST = 15

# Maks. wierszy używanych w każdym trialu Optuna (None = wszystkie)
# Przyspiesza MICE i MissForest na dużych zbiorach (np. KOR)
OPTUNA_VAL_ROWS = 5_000
