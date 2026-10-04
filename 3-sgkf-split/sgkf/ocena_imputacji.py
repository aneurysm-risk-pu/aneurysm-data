"""
ocena_imputacji.py
==================

CZĘŚĆ B etapu 3 — ocena jakości imputacji w foldach SGKF protokołem z etapu 2.

Błąd imputacji da się zmierzyć tylko na wartościach, które znamy. Dlatego, jak
w benchmarku z etapu 2, bierzemy kompletne wiersze, sztucznie ukrywamy część
wartości i porównujemy uzupełnienia z prawdą. Różnica względem etapu 2: ocena
jest out-of-fold — imputer i scaler dopasowane na treningu foldu dokładnie tak
jak w produkcji (dopasuj_fold), a oceniane są kompletne wiersze z części
testowej, czyli pacjenci niewidziani przy dopasowaniu.

Dwie maski na kompletnych wierszach części testowej każdego foldu:

  etap2    losowe 10% komórek — create_mask(frac=0.10, seed=42) z
           2-imputation/final/evaluate.py, ten sam protokół co w benchmarku
  wzorce   każdy kompletny wiersz dostaje wzorzec braków losowo wybranego
           NIEkompletnego wiersza tej samej kohorty z części treningowej
           (seed 42) — braki „całymi panelami”, jak w prawdziwych danych

Metryki z etapu 2 (te same funkcje): RMSE i MAE na zamaskowanych komórkach,
KL_mean po kolumnach, w skali [0, 1]; osobno dla KOR i NEURO.
Punkt odniesienia: benchmark MICE z parametrami produkcyjnymi (cross-param,
2-imputation/final/results/mice_validation.csv): KOR 0,0861, NEURO 0,0984.

Uwaga o skali: RMSE w skali [0, 1] zależy od skalera. W etapie 2 każda kohorta
miała własny MinMaxScaler; tu skaler pochodzi z treningu foldu (w ~96% KOR),
więc dla NEURO liczby nie są wprost porównywalne z benchmarkiem. Dlatego obok
RMSE raportujemy miarę niezależną od skali: RMSE MICE / RMSE uzupełniania
średnią z treningu (na tych samych maskach; < 1 = MICE lepsze od średniej).

Wynik: sgkf/results/ocena_imputacji.json

Uruchomienie (~15 min — jedno dopasowanie imputera na fold):
    python 3-sgkf-split/sgkf/ocena_imputacji.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aneurysm_sgkf_mice_pipeline import MICE_PARAMS, RANDOM_STATE, dopasuj_fold  # noqa: E402
from podzial import (BASE_DIR, KOL_ETYKIETA, WYNIKI, indeksy_foldow,  # noqa: E402
                     kolumny_cech, wczytaj_podzial, wczytaj_rekordy)

sys.path.insert(0, str(BASE_DIR / "2-imputation" / "final"))
from evaluate import compute_kl_mean, compute_mae, compute_rmse, create_mask  # noqa: E402

warnings.filterwarnings("ignore")
MASK_FRAC, SEED = 0.10, 42
BENCHMARK = BASE_DIR / "2-imputation" / "final" / "results" / "mice_validation.csv"


def odniesienie_etap2() -> dict:
    """Benchmark MICE z parametrami produkcyjnymi (dobranymi na KOR) — wiersze cross-param."""
    b = pd.read_csv(BENCHMARK)
    b = b[(b["experiment"] == "cross-param") & (b["params_from"] == "kor")]
    return {r.dataset: {"rmse": round(r.rmse, 4), "mae": round(r.mae, 4), "kl": round(r.kl, 4)}
            for r in b.itertuples()}


def maska_wzorce(kompletne_y: np.ndarray, braki_train: np.ndarray, y_train: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Dla każdego kompletnego wiersza: wzorzec braków losowego niekompletnego wiersza tej samej kohorty."""
    maska = np.zeros((len(kompletne_y), braki_train.shape[1]), dtype=bool)
    for k in (0, 1):
        wzorce = braki_train[(y_train == k) & braki_train.any(axis=1)]
        idx = np.flatnonzero(kompletne_y == k)
        maska[idx] = wzorce[rng.integers(0, len(wzorce), size=len(idx))]
    return maska


def metryki(prawda: np.ndarray, uzup: np.ndarray, maska: np.ndarray, kolumny: list,
            uzup_srednia: np.ndarray) -> dict:
    rmse, rmse_sr = compute_rmse(prawda, uzup, maska), compute_rmse(prawda, uzup_srednia, maska)
    return {"rmse": rmse, "mae": compute_mae(prawda, uzup, maska),
            "kl_mean": compute_kl_mean(pd.DataFrame(prawda, columns=kolumny), pd.DataFrame(uzup, columns=kolumny)),
            "rmse_srednia_z_treningu": rmse_sr, "rmse_wzgledem_sredniej": rmse / rmse_sr,
            "zamaskowane_komorki": int(maska.sum()), "udzial_zamaskowanych": float(maska.mean())}


def ocen_fold(df: pd.DataFrame, feat: list, train_idx: np.ndarray, test_idx: np.ndarray, fold: int) -> dict:
    X_train = df[feat].iloc[train_idx].reset_index(drop=True)
    y_train = df[KOL_ETYKIETA].iloc[train_idx].to_numpy()
    test = df.iloc[test_idx].reset_index(drop=True)

    t0 = time.time()
    scaler, imputer, _ = dopasuj_fold(X_train, seed=RANDOM_STATE, params=MICE_PARAMS)
    czas = time.time() - t0
    srednie_train = np.nanmean(scaler.transform(X_train.to_numpy(dtype=float)), axis=0)   # punkt odniesienia

    kompletne = test.dropna(subset=feat).reset_index(drop=True)
    prawda = scaler.transform(kompletne[feat].to_numpy(dtype=float))
    y = kompletne[KOL_ETYKIETA].to_numpy()

    maski = {
        "etap2": create_mask(kompletne[feat], MASK_FRAC, SEED),
        "wzorce": maska_wzorce(y, X_train.isna().to_numpy(), y_train, np.random.default_rng(SEED)),
    }
    wynik = {"fold": fold, "czas_dopasowania_s": round(czas, 1),
             "kompletne_wiersze_testu": {"kor": int((y == 0).sum()), "neuro": int((y == 1).sum())}}
    for nazwa, maska in maski.items():
        zamaskowane = prawda.copy()
        zamaskowane[maska] = np.nan
        uzup = imputer.transform(zamaskowane)
        uzup_sr = np.where(maska, srednie_train[None, :], prawda)
        wynik[nazwa] = {}
        for k, kohorta in ((0, "kor"), (1, "neuro")):
            sel = y == k
            wynik[nazwa][kohorta] = metryki(prawda[sel], uzup[sel], maska[sel], feat, uzup_sr[sel])
        # RMSE per cecha dla NEURO — gdzie imputacja myli się najbardziej
        sel = y == 1
        per = {}
        for j, c in enumerate(feat):
            m = maska[sel, j]
            if m.sum() >= 10:
                per[c] = {"rmse": float(np.sqrt(np.mean((prawda[sel, j][m] - uzup[sel, j][m]) ** 2))),
                          "n": int(m.sum())}
        wynik[nazwa]["neuro_per_cecha"] = per
    return wynik


def podsumuj(foldy: list) -> dict:
    out = {}
    for maska in ("etap2", "wzorce"):
        out[maska] = {}
        for k in ("kor", "neuro"):
            for m in ("rmse", "mae", "kl_mean", "rmse_srednia_z_treningu", "rmse_wzgledem_sredniej",
                      "udzial_zamaskowanych"):
                v = [f[maska][k][m] for f in foldy]
                out[maska].setdefault(k, {})[m] = {"srednia": round(float(np.mean(v)), 4),
                                                   "odch_std": round(float(np.std(v)), 4),
                                                   "min_max": [round(min(v), 4), round(max(v), 4)]}
        # najtrudniejsze cechy NEURO (średni RMSE po foldach, tylko cechy obecne we wszystkich foldach)
        cechy = set.intersection(*[set(f[maska]["neuro_per_cecha"]) for f in foldy])
        sr = {c: float(np.mean([f[maska]["neuro_per_cecha"][c]["rmse"] for f in foldy])) for c in cechy}
        out[maska]["neuro_najtrudniejsze_cechy"] = {c: round(v, 4) for c, v in
                                                    sorted(sr.items(), key=lambda kv: -kv[1])[:6]}
    return out


def main() -> None:
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    feat = kolumny_cech(df, "38")
    odn = odniesienie_etap2()
    print(f"Odniesienie (etap 2, MICE z parametrami produkcyjnymi): {odn}\n")

    foldy = []
    for fold, tr, te in indeksy_foldow(df, przydzial):
        print(f"  Fold {fold + 1}/{przydzial.nunique()} — dopasowanie i ocena ...", flush=True)
        w = ocen_fold(df, feat, tr, te, fold)
        foldy.append(w)
        for maska in ("etap2", "wzorce"):
            print(f"    {maska:7s} RMSE KOR {w[maska]['kor']['rmse']:.4f}  NEURO {w[maska]['neuro']['rmse']:.4f}  "
                  f"| względem średniej KOR {w[maska]['kor']['rmse_wzgledem_sredniej']:.2f}  "
                  f"NEURO {w[maska]['neuro']['rmse_wzgledem_sredniej']:.2f}  "
                  f"(zamaskowane: KOR {w[maska]['kor']['udzial_zamaskowanych']:.1%}, "
                  f"NEURO {w[maska]['neuro']['udzial_zamaskowanych']:.1%})", flush=True)

    pods = podsumuj(foldy)
    wynik = {"protokol": {"maska_etap2": f"losowe {MASK_FRAC:.0%} komórek, seed {SEED} (create_mask z etapu 2)",
                          "maska_wzorce": f"wzorce braków niekompletnych wierszy tej samej kohorty z treningu foldu, seed {SEED}",
                          "imputer": {**MICE_PARAMS, "random_state": RANDOM_STATE},
                          "ocena": "out-of-fold: kompletne wiersze części testowej, metryki w skali [0,1]"},
             "odniesienie_etap2": odn, "podsumowanie": pods, "foldy": foldy,
             "odcisk_danych": meta["odcisk_danych_sha256"]}
    (WYNIKI / "ocena_imputacji.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")

    print("\n=== Średnio po 5 foldach (RMSE, skala [0,1])")
    for maska in ("etap2", "wzorce"):
        for k in ("kor", "neuro"):
            r = pods[maska][k]["rmse"]
            w = pods[maska][k]["rmse_wzgledem_sredniej"]
            ref = f" | etap 2: {odn[k]['rmse']}" if maska == "etap2" else ""
            print(f"  {maska:7s} {k.upper():5s} {r['srednia']:.4f} ± {r['odch_std']:.4f}  "
                  f"(względem średniej {w['srednia']:.2f}){ref}")
    print(f"\nZapisano -> {(WYNIKI / 'ocena_imputacji.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
