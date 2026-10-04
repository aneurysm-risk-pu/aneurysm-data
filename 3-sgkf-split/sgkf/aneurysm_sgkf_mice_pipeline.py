"""
aneurysm_sgkf_mice_pipeline.py
================================

Podział na foldy po pacjentach + imputacja MICE wewnątrz pętli foldów.

Metodologia (zapobieganie data leakage):
  1. Foldy z zamrożonego przydziału `results/pacjent_fold.csv` (podzial.py)
  2. MinMaxScaler fitowany TYLKO na wierszach train bez braków (complete cases)
  3. IterativeImputer (MICE) fitowany TYLKO na X_train po skalowaniu
  4. X_test transformowany imputer.transform() — nigdy fit_transform()
  5. Inwersja skalera → oryginalna skala na wyjściu

Parametry MICE (finalne, z 2-imputation/RAPORT_IMPUTACJA.md):
  ExtraTrees, max_iter=7, n_estimators=78, max_depth=10
  min_value=0.0, max_value=1.0 (dane są w [0,1] po skalowaniu)

Zmiany w wersji 2 (04.10.2026):
  - foldy nie są liczone w locie przez StratifiedGroupKFold(shuffle=False),
    tylko wczytywane z zapisanego przydziału pacjent -> fold;
    uzasadnienie w RAPORT_SGKF_MICE.md, sekcja 6
  - domyślnie 38 cech (rekomendacja 0.4), 35 dostępne przez --cechy 35
  - dodatkowe kontrole: każdy pacjent dokładnie w jednym foldzie testowym,
    brak wartości ujemnych, udział imputowanych komórek per kohorta
  - --podprobka / --szybkie-mice do szybkiego sprawdzenia całej ścieżki

Uruchomienie:
    python 3-sgkf-split/podzial.py                                   # raz: zbuduj i zapisz foldy
    python 3-sgkf-split/aneurysm_sgkf_mice_pipeline.py               # pełny przebieg (~5 x 20 min)
    python 3-sgkf-split/aneurysm_sgkf_mice_pipeline.py --podprobka 3000 --szybkie-mice
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent))
from podzial import (DANE, KOL_ETYKIETA, KOL_PACJENT, META_KOLUMNY,  # noqa: E402
                     indeksy_foldow, wczytaj_podzial)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

TARGET_COL   = KOL_ETYKIETA
GROUP_COL    = KOL_PACJENT
RANDOM_STATE = 42   # ziarno MICE; ziarno podziału jest w results/podzial_meta.json

# Finalne parametry MICE (2-imputation/RAPORT_IMPUTACJA.md, przebieg 4 + cross-param)
MICE_PARAMS = dict(
    max_iter     = 7,
    n_estimators = 78,
    max_depth    = 10,
)

# Tylko do sprawdzenia, czy cała ścieżka działa — nie do wyników
MICE_PARAMS_SZYBKIE = dict(max_iter=2, n_estimators=10, max_depth=6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_mice_imputer(seed: int = RANDOM_STATE, params: dict = MICE_PARAMS) -> IterativeImputer:
    estimator = ExtraTreesRegressor(
        n_estimators = params["n_estimators"],
        max_depth    = params["max_depth"],
        random_state = seed,
        n_jobs       = -1,
    )
    return IterativeImputer(
        estimator    = estimator,
        max_iter     = params["max_iter"],
        min_value    = 0.0,   # dane są w [0,1] po MinMaxScaler
        max_value    = 1.0,
        random_state = seed,
        verbose      = 0,
    )


def prepare_fold(
    X_train_raw: pd.DataFrame,
    X_test_raw:  pd.DataFrame,
    seed: int = RANDOM_STATE,
    params: dict = MICE_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Skaluje i imputuje jeden fold.

    Zwraca
    -------
    X_train_imp, X_test_imp : ndarray w oryginalnej skali (po inwersji scalera)
    """
    # --- Scaler fitowany na complete cases z train ---
    # MinMaxScaler ignoruje NaN przy fit i zachowuje je przy transform.
    # Poza zakresem kompletnych wierszy leży ~0,01% obserwowanych wartości.
    train_complete = X_train_raw.dropna()
    scaler = MinMaxScaler()
    scaler.fit(train_complete.values.astype(float))

    X_train_scaled = scaler.transform(X_train_raw.values.astype(float))
    X_test_scaled  = scaler.transform(X_test_raw.values.astype(float))

    # --- MICE: fit na train, transform na test ---
    imputer = build_mice_imputer(seed=seed, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train_imp_scaled = imputer.fit_transform(X_train_scaled)
        X_test_imp_scaled  = imputer.transform(X_test_scaled)

    # --- Inwersja skalera → oryginalna skala ---
    X_train_imp = scaler.inverse_transform(X_train_imp_scaled)
    X_test_imp  = scaler.inverse_transform(X_test_imp_scaled)

    return X_train_imp, X_test_imp


def udzial_imputowanych(X_raw: pd.DataFrame, y: pd.Series) -> dict:
    """Odsetek komórek uzupełnionych przez MICE, osobno dla KOR i NEURO.

    Braki rozkładają się bardzo nierówno między kohortami (np. NRBC: 2,6% w KOR,
    47,7% w NEURO), a imputer uczy się głównie na KOR. Ta liczba mówi, jaka
    część profilu NEURO jest de facto odtworzona przez model.
    """
    braki = X_raw.isna().to_numpy()
    return {k: float(braki[(y == k).to_numpy()].mean()) for k in (0, 1)}


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Foldy po pacjentach + MICE wewnątrz foldu")
    ap.add_argument("--podprobka", type=int, default=None,
                    help="losowa podpróbka pacjentów (szybki test ścieżki)")
    ap.add_argument("--szybkie-mice", action="store_true",
                    help="lekkie parametry MICE — tylko do testu, nie do wyników")
    args = ap.parse_args(argv)
    params = MICE_PARAMS_SZYBKIE if args.szybkie_mice else MICE_PARAMS

    przydzial, meta = wczytaj_podzial()
    input_csv = DANE[meta["konfiguracja"]["zestaw_cech"]]
    print(f"[1] Wczytywanie {input_csv.name} ...")
    print(f"    Foldy: results/pacjent_fold.csv (seed={meta['konfiguracja']['seed']}, "
          f"warstwy={'+'.join(meta['konfiguracja']['warstwy'])}, "
          f"odcisk danych {meta['odcisk_danych_sha256']})")
    df = pd.read_csv(input_csv)

    if args.podprobka:
        wybrani = pd.Series(przydzial.index).sample(args.podprobka, random_state=0)
        przydzial = przydzial.loc[wybrani.sort_values().to_numpy()]
        print(f"    [podpróbka] {len(przydzial):,} pacjentów — wynik tylko do sprawdzenia ścieżki")

    df = df[df[GROUP_COL].isin(set(przydzial.index))].reset_index(drop=True)
    print(f"    Kształt: {df.shape}   |   label=0: {(df[TARGET_COL]==0).sum():,}   label=1: {(df[TARGET_COL]==1).sum():,}")

    feat_cols = [c for c in df.columns if c not in META_KOLUMNY]
    print(f"    Kolumny cech: {len(feat_cols)}")

    X      = df[feat_cols]
    y      = df[TARGET_COL]
    groups = df[GROUP_COL]

    missing_pct = X.isna().sum().sum() / X.size * 100
    print(f"    Braki w cechach: {X.isna().sum().sum():,} ({missing_pct:.1f}%)\n")

    n_folds = przydzial.nunique()
    splits = []   # lista słowników — gotowe do przekazania do modelu
    widziani_w_tescie = []

    print(f"[2] Foldy po pacjentach (n={n_folds}) + MICE ...\n")

    for fold, train_idx, test_idx in indeksy_foldow(df, przydzial):
        print(f"  Fold {fold + 1}/{n_folds} — imputacja MICE ...")

        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw  = X.iloc[test_idx].reset_index(drop=True)
        y_train     = y.iloc[train_idx].reset_index(drop=True)
        y_test      = y.iloc[test_idx].reset_index(drop=True)
        g_train     = groups.iloc[train_idx].reset_index(drop=True)
        g_test      = groups.iloc[test_idx].reset_index(drop=True)

        # Weryfikacja: brak wspólnych pacjentów
        overlap = set(g_train).intersection(set(g_test))
        assert len(overlap) == 0, f"Fold {fold+1}: pacjenci w obu zbiorach! {overlap}"
        widziani_w_tescie.extend(g_test.unique())

        # Imputacja
        X_train_imp, X_test_imp = prepare_fold(X_train_raw, X_test_raw,
                                               seed=RANDOM_STATE, params=params)

        # Weryfikacja: brak NaN i wartości ujemnych po imputacji
        assert not np.isnan(X_train_imp).any(), f"Fold {fold+1}: NaN w X_train po imputacji!"
        assert not np.isnan(X_test_imp).any(),  f"Fold {fold+1}: NaN w X_test po imputacji!"
        assert (X_train_imp >= 0).all() and (X_test_imp >= 0).all(), \
            f"Fold {fold+1}: wartości ujemne po imputacji!"

        imp_test = udzial_imputowanych(X_test_raw, y_test)

        print(f"    Rozmiar: train={len(train_idx):,}  test={len(test_idx):,}  "
              f"(pacjenci test: {g_test.nunique():,})")
        print(f"    Proporcja label=1 (rekordy): train={y_train.mean():.2%}  test={y_test.mean():.2%}")
        print(f"    Wspólne patient_id: {len(overlap)}  (oczekiwane: 0)")
        print(f"    Imputowane komórki w test: KOR={imp_test[0]:.1%}  NEURO={imp_test[1]:.1%}")
        print()

        splits.append({
            "fold":        fold,          # numeracja jak w results/pacjent_fold.csv (od 0)
            "X_train":     pd.DataFrame(X_train_imp, columns=feat_cols),
            "X_test":      pd.DataFrame(X_test_imp,  columns=feat_cols),
            "y_train":     y_train,
            "y_test":      y_test,
            "groups_train": g_train,
            "groups_test":  g_test,
        })

    # Każdy pacjent dokładnie raz w części testowej — warunek predykcji out-of-fold
    assert len(widziani_w_tescie) == len(set(widziani_w_tescie)) == len(przydzial), \
        "pacjent pominięty albo powtórzony w częściach testowych"

    print("[3] Gotowe. Zwrócono listę `splits` z kluczami:")
    print("    X_train, X_test  — DataFrame, zaimputowane, oryginalna skala")
    print("    y_train, y_test  — Series z labelami")
    print("    groups_train, groups_test — patient_id")

    return splits


if __name__ == "__main__":
    splits = main()
