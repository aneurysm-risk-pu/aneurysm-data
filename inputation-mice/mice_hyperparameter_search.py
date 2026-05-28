"""
Optymalizacja hiperparametrow MICE - Metodyka A
================================================

MinMax -> MICE (fit_transform na zamaskowanych znorm.) -> RMSE/MAE/KL na [0,1]
Maskowanie: calosciowe 10% (jak KNN kolegi)

Testowane parametry:
  estimator    : BayesianRidge | ExtraTreesRegressor(n=10/50/100)
  max_iter     : [5, 10, 20, 30, 50]

Metryki walidacji:
  RMSE, MAE  — dokladnosc punktowa (porowywalna z KNN kolegi)
  KL         — wiernosc rozkladu (istotna dla downstream PU modelu)

Walidacja: max VAL_SAMPLE_SIZE kompletnych wierszy (dla szybkosci ExtraTrees)
Finalny model: najlepsza konfiguracja wg RMSE, imputacja calego zbioru
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
KL_BINS         = 50     # liczba binow histogramu dla KL divergence

META_COLS = ["custom_id", "patient_id", "examination_date",
             "patient_age", "patient_sex"]

MAX_ITER_OPTIONS      = [5, 10, 20, 30, 50]
N_ESTIMATORS_OPTIONS  = [10, 50, 100]   # dla ExtraTrees

KNN_REF = {
    "neuro": {"RMSE": 0.1169, "MAE": 0.0579, "info": "K=7,  distance"},
    "kor":   {"RMSE": 0.0993, "MAE": 0.0436, "info": "K=21, distance"},
}

# --- FUNKCJE -----------------------------------------------------------------

def get_value_cols(df):
    exclude = set(META_COLS) & set(df.columns)
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_configs():
    """Generuje liste wszystkich kombinacji (nazwa, estymator, max_iter)."""
    configs = []
    for max_iter in MAX_ITER_OPTIONS:
        configs.append({
            "name":      "BayesianRidge",
            "estimator": BayesianRidge(),
            "max_iter":  max_iter,
        })
    for n_est in N_ESTIMATORS_OPTIONS:
        for max_iter in MAX_ITER_OPTIONS:
            configs.append({
                "name":      f"ExtraTrees(n={n_est})",
                "estimator": ExtraTreesRegressor(
                                 n_estimators=n_est,
                                 max_depth=20,
                                 random_state=RANDOM_STATE,
                                 n_jobs=1),
                "max_iter":  max_iter,
            })
    return configs


def apply_scaler_preserving_nan(df_vals, scaler, value_cols):
    """MinMaxScaler.transform() na DataFrame z NaN (NaN pozostaje NaN)."""
    nan_mask = df_vals.isna()
    arr_norm = scaler.transform(df_vals.fillna(0.0).values.astype(float))
    df_norm  = pd.DataFrame(arr_norm, columns=value_cols, index=df_vals.index)
    df_norm[nan_mask] = np.nan
    return df_norm


def make_masked_data(complete_norm, rng):
    """Calosciowe maskowanie ~10% (jak KNN kolegi)."""
    mask   = rng.random(complete_norm.shape) < MASK_FRAC
    masked = complete_norm.copy()
    masked[mask] = np.nan
    return masked, mask


def kl_divergence(p_vals, q_vals, bins=KL_BINS):
    """
    KL divergence KL(P||Q) miedzy dwoma zbiorami wartosci.
    P = oryginalne wartosci, Q = wartosci imputowane.
    Im mniejsza wartosc, tym lepiej (0 = identyczne rozklady).
    """
    if len(p_vals) < 2 or len(q_vals) < 2:
        return np.nan
    all_vals = np.concatenate([p_vals, q_vals])
    v_min, v_max = all_vals.min(), all_vals.max()
    if v_max == v_min:
        return 0.0
    bin_edges = np.linspace(v_min, v_max, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bin_edges)
    q_hist, _ = np.histogram(q_vals, bins=bin_edges)
    eps = 1e-10
    p_hist = p_hist.astype(float) + eps
    q_hist = q_hist.astype(float) + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def run_single(estimator, max_iter, masked_df, original_norm, mask, value_cols):
    """Jeden eksperyment: fit_transform na zamaskowanych znorm. -> RMSE/MAE/KL."""
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

    # KL divergence per kolumna -> srednia globalna
    kl_vals = []
    for col_idx in range(original_norm.shape[1]):
        col_mask = mask[:, col_idx]
        if col_mask.sum() < 2:
            continue
        kl = kl_divergence(original_norm[col_mask, col_idx],
                           imputed[col_mask, col_idx])
        if not np.isnan(kl):
            kl_vals.append(kl)
    mean_kl = float(np.mean(kl_vals)) if kl_vals else np.nan

    return rmse, mae, mean_kl


def impute_full_dataset(df, value_cols, scaler, meta_present,
                        best_estimator, best_max_iter, name):
    """Imputuje caly zbior najlepsza konfiguracja (Metodyka A)."""
    print(f"\n  Fit na pelnym zbiorze (moze trwac dlugo dla ExtraTrees + duzy zbior)...")
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
    """Wykres RMSE i KL vs max_iter, hue = konfiguracja estymatora."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=130)

    for ax, metric in zip(axes, ["RMSE", "MAE", "KL"]):
        sns.lineplot(
            data=df_res, x="max_iter", y=metric, hue="Estymator",
            marker="o", linewidth=2, markersize=7, ax=ax
        )
        if metric in knn_ref:
            ax.axhline(knn_ref[metric], color="red", linestyle="--",
                       linewidth=1.5, label=f"KNN ref ({knn_ref[metric]:.4f})")
        ax.set_title(f"{metric} — {name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("max_iter", fontsize=10)
        ax.set_ylabel(f"{metric} [0,1]", fontsize=10)
        ax.set_xticks(MAX_ITER_OPTIONS)
        ax.legend(title="Estymator", fontsize=8, title_fontsize=9)

    plt.suptitle(
        f"Optymalizacja hiperparametrow MICE — {name.upper()}",
        fontsize=13, fontweight="bold", y=1.01
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
    configs           = build_configs()
    total_configs     = len(configs)
    print(f"Zamaskowano {mask.sum()} wartosci ({mask.mean()*100:.1f}%), "
          f"konfiguracji do testowania: {total_configs}\n")

    # --- Petla eksperymentow --------------------------------------------------
    results = []
    for i, cfg in enumerate(configs, 1):
        est_name, estimator, max_iter = cfg["name"], cfg["estimator"], cfg["max_iter"]
        print(f"  [{i:2d}/{total_configs}] [{est_name:>22s}, max_iter={max_iter:2d}] ...",
              end=" ", flush=True)
        rmse, mae, mean_kl = run_single(estimator, max_iter, masked_df,
                                        val_norm, mask, value_cols)
        print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  KL={mean_kl:.4f}")
        results.append({
            "Estymator": est_name,
            "max_iter":  max_iter,
            "RMSE":      round(rmse, 6),
            "MAE":       round(mae, 6),
            "KL":        round(mean_kl, 6),
        })

    df_res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    # --- Ranking --------------------------------------------------------------
    knn = KNN_REF[name]
    print(f"\n{'='*65}")
    print(f"RANKING — {name.upper()} (sortowane po RMSE):")
    print(f"{'='*65}")
    print(df_res.to_string(index=False))
    print(f"\n  KNN referencja ({knn['info']}): "
          f"RMSE={knn['RMSE']}  MAE={knn['MAE']}")

    best = df_res.iloc[0]
    print(f"\nNajlepszy: {best['Estymator']}, max_iter={int(best['max_iter'])}, "
          f"RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}, KL={best['KL']:.4f}")

    # --- Zapis wynikow --------------------------------------------------------
    res_path = os.path.join(OUTPUT_DIR, f"{name}_mice_hyperparameter_results.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\n  Wyniki: {res_path}")

    # --- Wykres ---------------------------------------------------------------
    print("  Generowanie wykresu...")
    make_plot(df_res, name, knn)

    # --- Imputacja finalna najlepsza konfiguracja ----------------------------
    best_est_name  = best["Estymator"]
    best_max_iter  = int(best["max_iter"])
    best_estimator = next(c["estimator"] for c in configs
                          if c["name"] == best_est_name
                          and c["max_iter"] == best_max_iter)
    print(f"\nImputacja finalnego zbioru [{best_est_name}, max_iter={best_max_iter}]...")
    impute_full_dataset(df, value_cols, scaler, meta_present,
                        best_estimator, best_max_iter, name)

    return df_res


# --- ENTRY POINT -------------------------------------------------------------

if __name__ == "__main__":
    res_neuro = run_experiment("neuro", NEURO_FILE)
    res_kor   = run_experiment("kor",   KOR_FILE)

    # Polaczony plik wynikowy
    res_neuro["Zbior"] = "NEURO"
    res_kor["Zbior"]   = "KOR"
    combined = pd.concat([res_neuro, res_kor], ignore_index=True)
    combined = combined[["Zbior", "Estymator", "max_iter", "RMSE", "MAE", "KL"]]

    combined_path = os.path.join(OUTPUT_DIR, "mice_hyperparameter_all_results.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n{'='*65}")
    print(f"  ZBIORCZY PLIK: {combined_path}")
    print(f"{'='*65}")
    print(combined.to_string(index=False))
    print("\nGotowe.")
