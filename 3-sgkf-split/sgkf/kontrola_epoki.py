"""
kontrola_epoki.py
=================

Jednorazowa kontrola testu B4 z analiza_wynikow.py (06.10.2026).

B4 pokazał, że rekordy NEURO ze wspólnego okna dat kohort da się odróżnić od
rekordów NEURO z innych lat z AUC 0,83. Test liczono na cechach PO imputacji,
więc nie rozstrzyga, gdzie siedzi sygnał epoki: w samych wartościach
laboratoryjnych czy w tym, które badania zlecano (wzorzec braków).

Ten skrypt powtarza B4 na danych PRZED imputacją, na tych samych zamrożonych
foldach (HistGradientBoosting, braki jako NaN):
  1. wszystkie 38 cech, braki jako NaN,
  2. tylko wzorzec braków (0/1),
  3. wyłącznie wartości zmierzone: rekordy z kompletnym panelem, bez imputacji.

Nie jest częścią uruchom_sgkf.py; nie wymaga zaimputowanych foldów (~1 min).

Uruchomienie:
    python 3-sgkf-split/sgkf/kontrola_epoki.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from podzial import (KOL_DATA, KOL_ETYKIETA, KOL_PACJENT, WYNIKI,  # noqa: E402
                     kolumny_cech, wczytaj_podzial, wczytaj_rekordy, wspolne_okno_dat)

MORFOLOGIA = ["HGB", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW", "PLT", "WBC"]
ROZMAZ = ["MPV", "LYMPH", "%LYMPH", "EO", "%EO", "BAZO", "%BAZO", "MONO", "%MONO"]
ELEKTROLITY = ["Na", "K", "KREA"]


def _auc_po_foldach(X: pd.DataFrame, y: pd.Series, fold: pd.Series) -> dict:
    """Ten sam model co w analiza_wynikow.py; ocena na części testowej każdego foldu."""
    auc = []
    for f in sorted(fold.unique()):
        tr, te = (fold != f).to_numpy(), (fold == f).to_numpy()
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        auc.append(roc_auc_score(y[te], m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return {"auc_srednia": round(float(np.mean(auc)), 3),
            "auc_min_max": [round(float(min(auc)), 3), round(float(max(auc)), 3)]}


def main() -> None:
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    feat = kolumny_cech(df)
    od, do = wspolne_okno_dat(df)

    n = df[df[KOL_ETYKIETA] == 1].reset_index(drop=True)
    fold = n[KOL_PACJENT].map(przydzial)
    y = n[KOL_DATA].between(od, do).astype(int)

    wyniki = {}

    def licz(nazwa: str, X: pd.DataFrame, maska: pd.Series = None) -> None:
        m = pd.Series(True, index=n.index) if maska is None else maska
        r = _auc_po_foldach(X[m].reset_index(drop=True), y[m].reset_index(drop=True),
                            fold[m].reset_index(drop=True))
        r.update({"rekordy": int(m.sum()), "rekordy_w_oknie": int(y[m].sum())})
        wyniki[nazwa] = r
        print(f"  {nazwa:52s} AUC {r['auc_srednia']:.3f} {r['auc_min_max']}  "
              f"(rekordy {r['rekordy']:,}, w oknie {r['rekordy_w_oknie']:,})", flush=True)

    print(f"NEURO: rekordy z okna {od.date()} – {do.date()} vs pozostałe, 5 zamrożonych foldów")
    licz("38 cech, braki jako NaN (bez imputacji)", n[feat])
    licz("tylko wzorzec braków (0/1)", n[feat].isna().astype(int))
    for nazwa, cechy in (("morfologia podstawowa, tylko zmierzone", MORFOLOGIA),
                         ("morfologia + rozmaz, tylko zmierzone", MORFOLOGIA + ROZMAZ),
                         ("morfologia + Na, K, KREA, tylko zmierzone", MORFOLOGIA + ELEKTROLITY)):
        licz(nazwa, n[cechy], n[cechy].notna().all(axis=1))

    wynik = {"opis": "Kontrola testu B4: czy epokę w NEURO widać w wartościach, czy we wzorcu braków",
             "okno": [str(od.date()), str(do.date())], "odcisk_danych": meta["odcisk_danych_sha256"],
             "testy": wyniki}
    (WYNIKI / "kontrola_epoki.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(f"\nWynik -> {(WYNIKI / 'kontrola_epoki.json').relative_to(WYNIKI.parents[2])}")


if __name__ == "__main__":
    main()
