"""
Ewaluacja KL divergence na pelnych zbiorach po imputacji MICE
=============================================================

Metodologia:
  P = rozklad obserwowanych wartosci w kolumnie X (oryginalne dane, NaN pomijane)
  Q = rozklad tej samej kolumny po imputacji (wszystkie wiersze, w tym imputowane)
  KL(P||Q) = sum( P * log(P/Q) )

Liczba binow: regula Sturges'a  k = floor(log2(n) + 1), min 5

Uruchomienie:
  python inputation-mice/mice_kl_evaluation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── sciezki ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
RESULTS  = Path(__file__).parent / "results"

DATASETS = [
    {
        "name":     "NEURO",
        "original": BASE / "neuro_shortend.csv",
        "imputed":  RESULTS / "neuro_mice_best_imputed.csv",
    },
    {
        "name":     "KOR",
        "original": BASE / "kor_shortend.csv",
        "imputed":  RESULTS / "kor_mice_best_imputed.csv",
    },
]

# ── funkcje ───────────────────────────────────────────────────────────────────

def sturges_bins(n: int) -> int:
    return max(5, int(np.log2(n) + 1))


def kl_divergence(p_vals: np.ndarray, q_vals: np.ndarray, k: int) -> float:
    """KL(P||Q) histogram-based. Dodaje epsilon zeby uniknac log(0)."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max()) + 1e-9
    bins = np.linspace(lo, hi, k + 1)
    eps = 1e-10
    P, _ = np.histogram(p_vals, bins=bins, density=True)
    Q, _ = np.histogram(q_vals, bins=bins, density=True)
    P = P + eps;  P /= P.sum()
    Q = Q + eps;  Q /= Q.sum()
    return float(np.sum(P * np.log(P / Q)))


def evaluate(name: str, orig_path: Path, imp_path: Path) -> pd.DataFrame:
    if not imp_path.exists():
        print(f"[{name}] BRAK PLIKU: {imp_path.name} — pomijam.")
        return pd.DataFrame()

    orig = pd.read_csv(orig_path)
    imp  = pd.read_csv(imp_path)

    # kolumny numeryczne wspolne dla obu plikow
    num_cols = [
        c for c in orig.columns
        if c in imp.columns and pd.api.types.is_numeric_dtype(orig[c])
    ]

    rows = []
    for col in num_cols:
        p = orig[col].dropna().values
        q = imp[col].values          # po imputacji nie powinno byc NaN
        q = q[~np.isnan(q)]
        if len(p) < 5 or len(q) < 5:
            continue
        k   = sturges_bins(len(p))
        kl  = kl_divergence(p, q, k)
        missing_pct = orig[col].isna().mean() * 100
        rows.append({
            "kolumna":      col,
            "n_obs":        len(p),
            "braki_%":      round(missing_pct, 1),
            "bins":         k,
            "KL":           round(kl, 6),
        })

    return pd.DataFrame(rows)


# ── glowna petla ──────────────────────────────────────────────────────────────

for ds in DATASETS:
    name = ds["name"]
    print(f"\n{'='*65}")
    print(f"  ZBIOR: {name}")
    print(f"{'='*65}")

    df = evaluate(name, ds["original"], ds["imputed"])
    if df.empty:
        continue

    print(df.to_string(index=False))

    worst = df.loc[df.KL.idxmax()]
    best  = df.loc[df.KL.idxmin()]

    print()
    print(f"Srednie KL : {df.KL.mean():.4f}")
    print(f"Mediana KL : {df.KL.median():.4f}")
    print(f"Max KL     : {worst.KL:.4f}  [{worst.kolumna}]  (braki: {worst['braki_%']}%)")
    print(f"Min KL     : {best.KL:.4f}  [{best.kolumna}]")

    # zapis CSV
    out_path = RESULTS / f"{name.lower()}_mice_kl_full.csv"
    df.to_csv(out_path, index=False)
    print(f"\nZapisano: {out_path.name}")
