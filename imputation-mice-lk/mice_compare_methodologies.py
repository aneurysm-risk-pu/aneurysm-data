"""
Porownanie trzech metodyk imputacji MICE
=========================================

Metodyka A (end-to-end normalizacja):
  MinMax -> MICE -> InverseMinMax
  Imputer widzi znormalizowane dane [0,1], wyjscie w oryginalnych jednostkach.
  Walidacja RMSE na [0,1]. <- POPRAWNA metodologicznie

Metodyka B (surowe dane):
  MICE na surowych wartosciach, walidacja RMSE na surowych.
  RMSE nieporownywalny miedzy cechami ani z KNN.

Metodyka C (hybryda - poprzedni skrypt):
  MICE na surowych wartosciach, ale RMSE liczony na [0,1].
  Imputer byl trenowany na innych wartosciach niz te, na ktorych liczymy blad.

Referencja KNN (kolega):
  NEURO K=7 distance: RMSE=0.1169 MAE=0.0579
  KOR  K=21 distance: RMSE=0.0993 MAE=0.0436
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

# --- SCIEZKI ------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "inputation-mice", "results")
NEURO_FILE = os.path.join(BASE_DIR, "neuro_shortend.csv")
KOR_FILE   = os.path.join(BASE_DIR, "kor_shortend.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- KONFIGURACJA ------------------------------------------------------------

RANDOM_STATE = 42
MAX_ITER     = 30
MASK_FRAC    = 0.10
META_COLS    = ["custom_id", "patient_id", "examination_date",
                "patient_age", "patient_sex"]

# --- FUNKCJE POMOCNICZE ------------------------------------------------------

def get_value_cols(df):
    exclude = set(META_COLS) & set(df.columns)
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def make_imputer(min_val=0.0, max_val=None):
    return IterativeImputer(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        min_value=min_val,
        max_value=max_val,
        verbose=0
    )


def apply_scaler_preserving_nan(df_vals, scaler, value_cols):
    """MinMaxScaler.transform() nie obsluguje NaN — robimy to recznie."""
    nan_mask = df_vals.isna()
    arr = df_vals.fillna(0.0).values.astype(float)
    arr_norm = scaler.transform(arr)
    df_norm = pd.DataFrame(arr_norm, columns=value_cols, index=df_vals.index)
    df_norm[nan_mask] = np.nan
    return df_norm


def masked_validation(complete_df, value_cols, seed=RANDOM_STATE,
                      post_scale_for_rmse=False, scaler=None,
                      imputer_max_val=None):
    """
    Walidacja przez maskowanie (wzorowana na metodzie KNN kolegi):
    - 10% wartosci z kompletnych wierszy zostaje zamaskowanych
    - fit_transform na zamaskowanych danych (jak u kolegi)
    - RMSE/MAE wzgledem oryginalnych wartosci

    post_scale_for_rmse=True: przed liczeniem RMSE normalizuje przez scaler
                               (uzywane w Metodyce C)
    """
    rng  = np.random.default_rng(seed)
    data = complete_df.values.astype(float)
    n_rows, n_cols = data.shape

    masked = data.copy()
    for col_idx in range(n_cols):
        rows = rng.choice(n_rows, size=int(n_rows * MASK_FRAC), replace=False)
        masked[rows, col_idx] = np.nan

    imp = make_imputer(min_val=0.0, max_val=imputer_max_val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imp.fit_transform(masked)

    if post_scale_for_rmse and scaler is not None:
        orig_for_rmse   = scaler.transform(data)
        imputed_for_rmse = scaler.transform(imputed)
    else:
        orig_for_rmse   = data
        imputed_for_rmse = imputed

    mask_bool = np.isnan(masked)
    results   = []
    all_o, all_i = [], []

    for col_idx, col in enumerate(value_cols):
        m = mask_bool[:, col_idx]
        o = orig_for_rmse[m, col_idx]
        i = imputed_for_rmse[m, col_idx]
        rmse = float(np.sqrt(np.mean((o - i) ** 2)))
        mae  = float(np.mean(np.abs(o - i)))
        results.append({"feature": col,
                        "RMSE": round(rmse, 6),
                        "MAE":  round(mae, 6)})
        all_o.extend(o)
        all_i.extend(i)

    all_o = np.array(all_o)
    all_i = np.array(all_i)
    g_rmse = float(np.sqrt(np.mean((all_o - all_i) ** 2)))
    g_mae  = float(np.mean(np.abs(all_o - all_i)))
    return pd.DataFrame(results), g_rmse, g_mae


# --- METODYKI ----------------------------------------------------------------

def metodyka_A(name, df, value_cols, scaler, meta_present):
    """MinMax -> MICE (na znorm.) -> InverseMinMax | RMSE na [0,1]"""
    print(f"  [A] MinMax -> MICE -> InverseMinMax")

    df_vals = df[value_cols]
    complete = df_vals.dropna()

    # Normalizuj caly zbior (zachowujac NaN)
    df_norm = apply_scaler_preserving_nan(df_vals, scaler, value_cols)

    # Fit + transform na znormalizowanych danych
    imp = make_imputer(min_val=0.0, max_val=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp.fit(df_norm)
        arr_norm = imp.transform(df_norm)

    # Odwroc normalizacje -> oryginalne jednostki
    arr_raw = scaler.inverse_transform(arr_norm)

    # Walidacja: kompletne wiersze -> znormalizuj -> maskuj -> fit_transform (norm) -> RMSE [0,1]
    complete_norm = pd.DataFrame(
        scaler.transform(complete.values.astype(float)),
        columns=value_cols
    )
    val_df, g_rmse, g_mae = masked_validation(
        complete_norm, value_cols,
        post_scale_for_rmse=False,   # dane sa juz znorm.
        imputer_max_val=1.0
    )

    print(f"      RMSE [0,1]: {g_rmse:.4f}  MAE: {g_mae:.4f}")

    # Zapis imputed (oryginalne jednostki)
    result = df[meta_present].copy()
    for i, col in enumerate(value_cols):
        result[col] = arr_raw[:, i]
    result = result.reindex(columns=df.columns)
    result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_A_imputed.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_A_validation.csv"), index=False)

    return g_rmse, g_mae


def metodyka_B(name, df, value_cols, scaler, meta_present):
    """MICE na surowych | RMSE na surowych wartosciach"""
    print(f"  [B] MICE na surowych danych")

    df_vals  = df[value_cols]
    complete = df_vals.dropna()

    imp = make_imputer(min_val=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp.fit(df_vals)
        arr_raw = imp.transform(df_vals)

    # Walidacja na surowych
    val_df_raw, g_rmse_raw, g_mae_raw = masked_validation(
        complete, value_cols,
        post_scale_for_rmse=False
    )
    # Dla informacji: RMSE na znorm. (zeby wiedziec jak bardzo odstaje od A i C)
    _, g_rmse_norm, g_mae_norm = masked_validation(
        complete, value_cols,
        post_scale_for_rmse=True, scaler=scaler
    )

    print(f"      RMSE surowe:  {g_rmse_raw:.4f}  MAE: {g_mae_raw:.4f}")
    print(f"      RMSE [0,1]:   {g_rmse_norm:.4f}  MAE: {g_mae_norm:.4f}  (info)")

    # Zapis
    result = df[meta_present].copy()
    for i, col in enumerate(value_cols):
        result[col] = arr_raw[:, i]
    result = result.reindex(columns=df.columns)
    result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_B_imputed.csv"), index=False)
    val_df_raw.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_B_validation.csv"), index=False)

    return g_rmse_raw, g_mae_raw, g_rmse_norm, g_mae_norm


def metodyka_C(name, df, value_cols, scaler, meta_present):
    """MICE na surowych | RMSE na [0,1] (hybryda — poprzedni skrypt)"""
    print(f"  [C] MICE na surowych + walidacja na [0,1] (hybryda)")

    df_vals  = df[value_cols]
    complete = df_vals.dropna()

    imp = make_imputer(min_val=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp.fit(df_vals)
        arr_raw = imp.transform(df_vals)

    # Walidacja: surowe wiersze, ale RMSE na znorm. (hybryda)
    val_df, g_rmse, g_mae = masked_validation(
        complete, value_cols,
        post_scale_for_rmse=True, scaler=scaler
    )

    print(f"      RMSE [0,1]: {g_rmse:.4f}  MAE: {g_mae:.4f}")

    result = df[meta_present].copy()
    for i, col in enumerate(value_cols):
        result[col] = arr_raw[:, i]
    result = result.reindex(columns=df.columns)
    result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_C_imputed.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_mice_C_validation.csv"), index=False)

    return g_rmse, g_mae


# --- ENTRY POINT -------------------------------------------------------------

def run_dataset(name, filepath):
    print(f"\n{'='*65}")
    print(f"  ZBIOR: {name.upper()}")
    print(f"{'='*65}")

    df = pd.read_csv(filepath)
    value_cols = get_value_cols(df)
    fully_missing = [c for c in value_cols if df[c].isna().all()]
    if fully_missing:
        print(f"  Pominieto (100% brakow): {fully_missing}")
        value_cols = [c for c in value_cols if c not in fully_missing]

    meta_present = [c for c in META_COLS if c in df.columns]
    complete = df[value_cols].dropna()

    print(f"  Wiersze: {df.shape[0]}, kolumny: {len(value_cols)}, "
          f"kompletnych wierszy: {len(complete)}\n")

    scaler = MinMaxScaler()
    scaler.fit(complete.values.astype(float))

    rA, mA = metodyka_A(name, df, value_cols, scaler, meta_present)
    rBr, mBr, rBn, mBn = metodyka_B(name, df, value_cols, scaler, meta_present)
    rC, mC = metodyka_C(name, df, value_cols, scaler, meta_present)

    return {
        "A":      (rA,  mA),
        "B_raw":  (rBr, mBr),
        "B_norm": (rBn, mBn),
        "C":      (rC,  mC),
    }


if __name__ == "__main__":
    res_neuro = run_dataset("neuro", NEURO_FILE)
    res_kor   = run_dataset("kor",   KOR_FILE)

    KNN_REF = {
        "neuro": (0.1169, 0.0579, "K=7,  distance"),
        "kor":   (0.0993, 0.0436, "K=21, distance"),
    }

    print("\n" + "="*72)
    print("  TABELA POROWNAN  (RMSE na skali [0,1] o ile nie zaznaczono)")
    print("="*72)
    print(f"\n{'Zbior':<8} {'Metodyka':<38} {'RMSE':>8} {'MAE':>8}")
    print("-" * 66)

    for zbior, res in [("neuro", res_neuro), ("kor", res_kor)]:
        kr, km, kinfo = KNN_REF[zbior]
        rows = [
            ("A: MinMax->MICE->InvMinMax [0,1]",  res["A"]),
            ("C: MICE surowe, RMSE norm. [0,1]",   res["C"]),
            ("B: MICE surowe, RMSE surowe (*)",     res["B_raw"]),
            ("B: MICE surowe, RMSE norm. [0,1]",   res["B_norm"]),
            (f"KNN ({kinfo}) [0,1]",               (kr, km)),
        ]
        for label, (r, m) in rows:
            marker = " <-- poprawna" if label.startswith("A:") else ""
            print(f"{zbior.upper():<8} {label:<38} {r:>8.4f} {m:>8.4f}{marker}")
        print()

    print("(*) RMSE surowe: nieporownywalny miedzy cechami i metodami")
    print("\nGotowe.")
