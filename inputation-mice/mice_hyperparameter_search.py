"""
Optymalizacja hiperparametrow MICE - Metodyka A
================================================

MinMax -> MICE (fit_transform na zamaskowanych znorm.) -> RMSE/MAE na [0,1]
Maskowanie: calosciowe 10% (jak KNN kolegi)

Testowane parametry:
  estimator : BayesianRidge | ExtraTreesRegressor(n_estimators=10)
  max_iter  : [5, 10, 20, 30, 50]

Walidacja: max VAL_SAMPLE_SIZE kompletnych wierszy (dla szybkosci ExtraTrees)
Finalny model: najlepsza konfiguracja, imputacja calego zbioru
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- SCIEZKI ------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "inputation-mice", "results")
NEURO_FILE = os.path.join(BASE_DIR, "neuro_shortend.csv")
KOR_FILE   = os.path.join(BASE_DIR, "kor_shortend.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- KONFIGURACJA ------------------------------------------------------------

RANDOM_STATE    = 42
MASK_FRAC       = 0.10
VAL_SAMPLE_SIZE = 5000   # max wierszy do walidacji (szybkosc ExtraTrees)

META_COLS = ["custom_id", "patient_id", "examination_date",
             "patient_age", "patient_sex"]

ESTIMATORS = {
    "BayesianRidge":    BayesianRidge(),
    "ExtraTrees(n=10)": ExtraTreesRegressor(
                            n_estimators=10,
                            random_state=RANDOM_STATE,
                            n_jobs=-1),
}
MAX_ITER_OPTIONS = [5, 10, 20, 30, 50]

KNN_REF = {
    "neuro": {"RMSE": 0.1169, "MAE": 0.0579, "info": "K=7,  distance"},
    "kor":   {"RMSE": 0.0993, "MAE": 0.0436, "info": "K=21, distance"},
}

# --- FUNKCJE -----------------------------------------------------------------

def get_value_cols(df):
    exclude = set(META_COLS) & set(df.columns)
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def apply_scaler_preserving_nan(df_vals, scaler, value_cols):
    """MinMaxScaler.transform() na DataFrame z NaN (NaN pozostaje NaN)."""
    nan_mask = df_vals.isna()
    arr_norm = scaler.transform(df_vals.fillna(0.0).values.astype(float))
    df_norm  = pd.DataFrame(arr_norm, columns=value_cols, index=df_vals.index)
    df_norm[nan_mask] = np.nan
    return df_norm


def make_masked_data(complete_norm, rng):
    """Calosciowe maskowanie 10% (jak KNN kolegi)."""
    mask   = rng.random(complete_norm.shape) < MASK_FRAC
    masked = complete_norm.copy()
    masked[mask] = np.nan
    return masked, mask


def run_single(estimator, max_iter, masked_df, original_norm, mask):
    """Jeden eksperyment: fit_transform na zamaskowanych znorm. -> RMSE/MAE."""
    imp = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        min_value=0.0,
        max_value=1.0,
        verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imp.fit_transform(masked_df)

    actual    = original_norm[mask]
    predicted = imputed[mask]
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    return rmse, mae


def impute_full_dataset(df, value_cols, scaler, meta_present,
                        best_estimator, best_max_iter, name):
    """Imputuje caly zbior najlepsza konfiguracja (Metodyka A)."""
    df_norm = apply_scaler_preserving_nan(df[value_cols], scaler, value_cols)

    imp = IterativeImputer(
        estimator=best_estimator,
        max_iter=best_max_iter,
        random_state=RANDOM_STATE,
        min_value=0.0,
        max_value=1.0,
        verbose=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp.fit(df_norm)
        arr_norm = imp.transform(df_norm)

    arr_raw = scaler.inverse_transform(arr_norm)

    result = df[meta_present].copy()
    for i, col in enumerate(value_cols):
        result[col] = arr_raw[:, i]
    result = result.reindex(columns=df.columns)

    out_path = os.path.join(OUTPUT_DIR, f"{name}_mice_best_imputed.csv")
    result.to_csv(out_path, index=False)
    print(f"  Zapisano: {out_path}")


def make_plot(df_res, name, knn_ref):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        sns.lineplot(
            data=df_res, x="max_iter", y=metric, hue="Estymator",
            marker="o", linewidth=2.5, markersize=8, ax=ax,
            palette=["#1f77b4", "#ff7f0e"]
        )
        # Linia referencyjna KNN
        ax.axhline(knn_ref[metric], color="red", linestyle="--",
                   linewidth=1.5, label=f"KNN ref ({knn_ref[metric]:.4f})")
        ax.set_title(f"{metric} — {name.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("max_iter", fontsize=11)
        ax.set_ylabel(f"Blad {metric} [0,1]", fontsize=11)
        ax.set_xticks(MAX_ITER_OPTIONS)
        ax.legend(title="Estymator / ref", fontsize=9, title_fontsize=10)

    plt.suptitle(
        f"Optymalizacja hiperparametrow MICE — {name.upper()}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"wykres_mice_{name}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Wykres: {out}")


# --- GLOWNA PROCEDURA --------------------------------------------------------

def run_experiment(name, filepath):
    print(f"\n{'='*65}")
    print(f"  ZBIOR: {name.upper()}")
    print(f"{'='*65}")

    df = pd.read_csv(filepath)
    value_cols = get_value_cols(df)
    fully_missing = [c for c in value_cols if df[c].isna().all()]
    if fully_missing:
        value_cols = [c for c in value_cols if c not in fully_missing]
    meta_present = [c for c in META_COLS if c in df.columns]

    complete = df[value_cols].dropna()
    print(f"Wiersze: {df.shape[0]}, kolumny: {len(value_cols)}, "
          f"kompletnych: {len(complete)}")

    # Scaler na kompletnych wierszach
    scaler = MinMaxScaler()
    scaler.fit(complete.values.astype(float))

    # Probka walidacyjna (max VAL_SAMPLE_SIZE)
    rng = np.random.default_rng(RANDOM_STATE)
    if len(complete) > VAL_SAMPLE_SIZE:
        idx = rng.choice(len(complete), size=VAL_SAMPLE_SIZE, replace=False)
        val_complete = complete.iloc[idx]
        print(f"Probka walidacyjna: {VAL_SAMPLE_SIZE} wierszy (z {len(complete)})")
    else:
        val_complete = complete
        print(f"Probka walidacyjna: {len(complete)} wierszy (wszystkie)")

    val_norm = scaler.transform(val_complete.values.astype(float))

    # Calosciowe maskowanie (jak KNN kolegi)
    masked_norm, mask = make_masked_data(val_norm, rng)
    masked_df         = pd.DataFrame(masked_norm, columns=value_cols)
    print(f"Zamaskowano {mask.sum()} wartosci ({mask.mean()*100:.1f}%)\n")

    # --- Petla eksperymentow --------------------------------------------------
    results = []
    for est_name, estimator in ESTIMATORS.items():
        for max_iter in MAX_ITER_OPTIONS:
            print(f"  [{est_name:>18s}, max_iter={max_iter:2d}] ...",
                  end=" ", flush=True)
            rmse, mae = run_single(estimator, max_iter, masked_df,
                                   val_norm, mask)
            print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")
            results.append({
                "Estymator": est_name,
                "max_iter":  max_iter,
                "RMSE":      round(rmse, 6),
                "MAE":       round(mae, 6),
            })

    df_res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    # --- Ranking --------------------------------------------------------------
    knn = KNN_REF[name]
    print(f"\n{'='*55}")
    print(f"RANKING — {name.upper()} (sortowane po RMSE):")
    print(f"{'='*55}")
    print(df_res.to_string(index=False))
    print(f"\n  KNN referencja ({knn['info']}): "
          f"RMSE={knn['RMSE']}  MAE={knn['MAE']}")

    best = df_res.iloc[0]
    print(f"\nNajlepszy: Estymator={best['Estymator']}, "
          f"max_iter={int(best['max_iter'])}, "
          f"RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}")

    # --- Zapis wynikow --------------------------------------------------------
    res_path = os.path.join(OUTPUT_DIR, f"{name}_mice_hyperparameter_results.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\n  Wyniki: {res_path}")

    # --- Wykres ---------------------------------------------------------------
    print("  Generowanie wykresu...")
    make_plot(df_res, name, knn)

    # --- Imputacja finalna najlepsza konfiguracja ----------------------------
    best_est_name = best["Estymator"]
    best_max_iter = int(best["max_iter"])
    print(f"\nImputacja finalnego zbioru [{best_est_name}, max_iter={best_max_iter}]...")
    impute_full_dataset(df, value_cols, scaler, meta_present,
                        ESTIMATORS[best_est_name], best_max_iter, name)

    return df_res


# --- ENTRY POINT -------------------------------------------------------------

if __name__ == "__main__":
    res_neuro = run_experiment("neuro", NEURO_FILE)
    res_kor   = run_experiment("kor",   KOR_FILE)

    # Polaczony plik wynikowy
    res_neuro["Zbior"] = "NEURO"
    res_kor["Zbior"]   = "KOR"
    combined = pd.concat([res_neuro, res_kor], ignore_index=True)
    combined = combined[["Zbior", "Estymator", "max_iter", "RMSE", "MAE"]]

    combined_path = os.path.join(OUTPUT_DIR, "mice_hyperparameter_all_results.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n{'='*65}")
    print(f"  ZBIORCZY PLIK: {combined_path}")
    print(f"{'='*65}")
    print(combined.to_string(index=False))
    print("\nGotowe.")
