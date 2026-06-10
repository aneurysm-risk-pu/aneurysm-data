"""
imputation-final/validate_imputed.py
======================================

Walidacja statystyczna zaimputowanego zbioru.

Testy:
  1. Statystyki opisowe (przed vs po) — mean, std, median per kolumna
  2. Kolmogorov-Smirnov test — rozkład complete cases vs zaimputowane
  3. Zachowanie korelacji — Pearson przed vs po (Frobenius norm diff)
  4. Osobno KOR (label=0) vs NEURO (label=1)

Wejście:
  - aneurysm_concatted.csv        (oryginał)
  - results/aneurysm_imputed_final.csv  (po imputacji)

Wyjście:
  - results/validation_stats.csv        (statystyki opisowe)
  - results/validation_ks.csv           (KS test per kolumna per grupa)
  - wydruk na stdout z podsumowaniem
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

ORIG_CSV    = BASE_DIR / "aneurysm_concatted.csv"
IMPUTED_CSV = RESULTS_DIR / "aneurysm_imputed_final.csv"

META_COLS = ["patient_id", "custom_id", "examination_date", "label"]

# ---------------------------------------------------------------------------

def feat_cols_of(df):
    return [c for c in df.columns if c not in META_COLS]


def ks_summary(orig_complete, imputed_all, feat_cols, label="ALL"):
    """KS test: complete cases (przed) vs wszystkie wiersze po imputacji."""
    rows = []
    for col in feat_cols:
        a = orig_complete[col].dropna().values
        b = imputed_all[col].values
        stat, pval = stats.ks_2samp(a, b)
        rows.append({
            "dataset": label,
            "column": col,
            "ks_stat": round(stat, 4),
            "p_value": round(pval, 4),
            "pass": "✅" if pval > 0.05 else "⚠️",
        })
    return pd.DataFrame(rows)


def describe_comparison(orig, imputed, feat_cols, label=""):
    rows = []
    for col in feat_cols:
        a = orig[col].dropna()
        b = imputed[col]
        rows.append({
            "dataset": label,
            "column": col,
            "mean_before": round(a.mean(), 4),
            "mean_after":  round(b.mean(), 4),
            "mean_delta":  round(b.mean() - a.mean(), 4),
            "std_before":  round(a.std(), 4),
            "std_after":   round(b.std(), 4),
            "median_before": round(a.median(), 4),
            "median_after":  round(b.median(), 4),
        })
    return pd.DataFrame(rows)


def corr_diff(orig_complete, imputed_all, feat_cols):
    """Frobenius norm różnicy macierzy korelacji."""
    C_before = orig_complete[feat_cols].corr().values
    C_after  = imputed_all[feat_cols].corr().values
    diff = C_after - C_before
    frob = np.sqrt((diff ** 2).sum())
    max_diff = np.abs(diff).max()
    return frob, max_diff


# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("WALIDACJA STATYSTYCZNA IMPUTACJI")
    print("=" * 65)

    # Wczytaj
    print(f"\n[1] Wczytywanie danych ...")
    orig    = pd.read_csv(ORIG_CSV)
    imputed = pd.read_csv(IMPUTED_CSV)
    feat    = feat_cols_of(orig)

    print(f"    Oryginał:     {orig.shape}   NaN: {orig[feat].isnull().sum().sum():,}")
    print(f"    Zaimputowany: {imputed.shape}  NaN: {imputed[feat].isnull().sum().sum()}")

    # Complete cases z oryginału = referencja
    orig_complete = orig[feat].dropna()
    print(f"    Complete cases (referencja): {len(orig_complete):,} wierszy")

    # -----------------------------------------------------------------------
    # 2. Statystyki opisowe
    # -----------------------------------------------------------------------
    print("\n[2] Statystyki opisowe (mean/std/median przed vs po) ...")

    all_stats = []
    for label_val, label_name in [(None, "ALL"), (0, "KOR"), (1, "NEURO")]:
        if label_val is None:
            o = orig_complete
            i = imputed[feat]
        else:
            o = orig[orig["label"] == label_val][feat].dropna()
            i = imputed[imputed["label"] == label_val][feat]
        all_stats.append(describe_comparison(o, i, feat, label=label_name))

    df_stats = pd.concat(all_stats, ignore_index=True)
    df_stats.to_csv(RESULTS_DIR / "validation_stats.csv", index=False)

    # Pokaż kolumny z największą zmianą meany (top 5)
    top_delta = (df_stats[df_stats["dataset"] == "ALL"]
                 .assign(abs_delta=lambda x: x["mean_delta"].abs())
                 .nlargest(5, "abs_delta")[["column", "mean_before", "mean_after", "mean_delta"]])
    print(f"\n    Top 5 kolumn z największą zmianą mean (ALL):")
    print(top_delta.to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. KS test
    # -----------------------------------------------------------------------
    print("\n[3] Kolmogorov-Smirnov test (rozkład complete vs po imputacji) ...")

    ks_results = []
    for label_val, label_name in [(None, "ALL"), (0, "KOR"), (1, "NEURO")]:
        if label_val is None:
            o = orig_complete
            i = imputed[feat]
        else:
            o = orig[orig["label"] == label_val][feat].dropna()
            i = imputed[imputed["label"] == label_val][feat]
        ks_results.append(ks_summary(o, i, feat, label=label_name))

    df_ks = pd.concat(ks_results, ignore_index=True)
    df_ks.to_csv(RESULTS_DIR / "validation_ks.csv", index=False)

    for label_name in ["ALL", "KOR", "NEURO"]:
        sub = df_ks[df_ks["dataset"] == label_name]
        n_pass = (sub["pass"] == "✅").sum()
        n_fail = (sub["pass"] == "⚠️").sum()
        print(f"\n    [{label_name}]  ✅ pass (p>0.05): {n_pass}/{len(sub)}   ⚠️ fail: {n_fail}")
        if n_fail > 0:
            failed = sub[sub["pass"] == "⚠️"][["column", "ks_stat", "p_value"]]
            print(failed.to_string(index=False))

    # -----------------------------------------------------------------------
    # 4. Zachowanie korelacji
    # -----------------------------------------------------------------------
    print("\n[4] Zachowanie struktury korelacji (Frobenius norm) ...")

    for label_val, label_name in [(None, "ALL"), (0, "KOR"), (1, "NEURO")]:
        if label_val is None:
            o = orig_complete
            i = imputed[feat]
        else:
            o = orig[orig["label"] == label_val][feat].dropna()
            i = imputed[imputed["label"] == label_val][feat]
        frob, max_d = corr_diff(o, i, feat)
        print(f"    [{label_name}]  Frobenius norm: {frob:.4f}   Max |Δcorr|: {max_d:.4f}")

    # -----------------------------------------------------------------------
    # 5. Podsumowanie
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PODSUMOWANIE")
    print("=" * 65)
    n_all_pass = (df_ks[df_ks["dataset"] == "ALL"]["pass"] == "✅").sum()
    n_feat = len(feat)
    print(f"  NaN po imputacji: {imputed[feat].isnull().sum().sum()}")
    print(f"  Wartości < 0:     {(imputed[feat] < 0).sum().sum()}")
    print(f"  KS test ALL: {n_all_pass}/{n_feat} kolumn p>0.05")
    print(f"\n  Zapisano:")
    print(f"    results/validation_stats.csv")
    print(f"    results/validation_ks.csv")


if __name__ == "__main__":
    main()
