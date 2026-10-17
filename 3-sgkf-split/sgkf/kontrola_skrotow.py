"""
kontrola_skrotow.py
===================

Kontrola z 17.10.2026: ile „odzyskiwania ukrytych pozytywnych” dają same skróty.

Ukrywamy losowo wybranych chorych z NEURO, więc ukryci zachowują wszystkie
cechy kohorty NEURO: okres badań, wzorzec braków, ślad imputacji. Model, który
nauczył się tylko tych cech, też wypchnie ich na górę rankingu. Wysoki
RecallHidden@q nie jest więc sam w sobie dowodem wykrywania choroby; potrzebny
jest punkt odniesienia. Skrypt liczy RecallHidden@q modeli kontrolnych na tych
samych maskach i zamrożonych foldach:

  losowy ranking pacjentów             punkt zerowy (oczekiwane q)
  losowy wynik rekordów, mediana       ten sam los, ale agregowany do pacjenta jak predykcje
  wiek i płeć
  sam wzorzec braków (0/1)             tylko to, które badania zlecono
  38 cech, braki jako NaN              bez imputacji
  38 cech po imputacji MICE            zapisane foldy (foldy_imputowane_38/)

oraz te same skróty w wariancie z profilem pacjenta (mediana cech, udział
braków), bo jednostka treningu nie jest jeszcze ustalona.

Dodatkowo B3 z analiza_wynikow.py BEZ imputacji: NEURO vs KOR tylko na rekordach
ze wspólnego okna dat, braki jako NaN. W B3 okno wybierano po imputacji
dopasowanej także na innych okresach; ta wersja sprawdza, czy wynik od tego zależał.

Protokół jest ORIENTACYJNY (do zamrożenia w pakiecie 1, 4-pu-setup/PLAN_MODELOWANIA.md):
  ukrywani całi pacjenci NEURO, 40%, ziarna 0–4;
  trening na treningu foldu z etykietą widzianą przez model (ukryci = 0);
  pula oceny: pacjenci KOR i ukryci NEURO z części testowej foldu;
  top-q: ceil(q · |pula|), remisy rozstrzygane po patient_id;
  wynik pacjenta w wariancie rekordowym = mediana wyników jego rekordów;
  model jak w diagnostyce skrótów (HistGradientBoosting, analiza_wynikow.py).

Wynik: sgkf/results/kontrola_skrotow.json
Nie jest częścią uruchom_sgkf.py; wymaga zapisanych foldów po imputacji (~3 min).

Uruchomienie:
    python 3-sgkf-split/sgkf/kontrola_skrotow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from podzial import (BASE_DIR, KOL_DATA, KOL_ETYKIETA, KOL_PACJENT, WYNIKI,  # noqa: E402
                     kolumny_cech, wczytaj_podzial, wczytaj_rekordy, wspolne_okno_dat)

FOLDY_IMP = WYNIKI / "foldy_imputowane_38"
UDZIAL_UKRYTYCH = 0.40
ZIARNA = range(5)
Q = (0.01, 0.05, 0.10)


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def recall_hidden(wynik: pd.Series, ukryci_te: set, q: float) -> float:
    """wynik: patient_id -> wynik na puli (KOR i ukryci NEURO z części testowej)."""
    pula = (wynik.rename("s").rename_axis(KOL_PACJENT).reset_index()
            .sort_values(["s", KOL_PACJENT], ascending=[False, True]))
    top = set(pula[KOL_PACJENT].iloc[:int(np.ceil(q * len(pula)))])
    return len(top & ukryci_te) / len(ukryci_te)


def _podsumuj(v: list) -> dict:
    return {"srednia": round(float(np.mean(v)), 3), "min_max": [round(float(min(v)), 3), round(float(max(v)), 3)]}


def odzyskiwanie_ukrytych(df: pd.DataFrame, feat: list, przydzial: pd.Series) -> tuple[dict, dict]:
    n_fold = przydzial.nunique()
    fold_rek = df[KOL_PACJENT].map(przydzial).to_numpy()
    etyk_pac = df.groupby(KOL_PACJENT)[KOL_ETYKIETA].max()
    neuro = np.sort(etyk_pac.index[etyk_pac == 1].to_numpy())

    rekordowe = {"wiek i płeć": df[["patient_age", "patient_sex"]],
                 "sam wzorzec braków (0/1)": df[feat].isna().astype(int),
                 "38 cech, braki jako NaN (bez imputacji)": df[feat]}
    imputowane = [(pd.read_parquet(FOLDY_IMP / f"fold{f}_train.parquet"),
                   pd.read_parquet(FOLDY_IMP / f"fold{f}_test.parquet")) for f in range(n_fold)]
    profile = {"profil pacjenta: udział braków każdej cechy": df[feat].isna().groupby(df[KOL_PACJENT]).mean(),
               "profil pacjenta: mediana 38 cech (NaN)": df.groupby(KOL_PACJENT)[feat].median()}
    fold_pac = przydzial.loc[next(iter(profile.values())).index].to_numpy()

    nazwy = (["losowy ranking pacjentów", "losowy wynik rekordów, mediana do pacjenta"] + list(rekordowe)
             + ["38 cech po imputacji MICE (zapisane foldy)"] + list(profile))
    wyn = {k: {q: [] for q in Q} for k in nazwy}
    ukrytych_na_fold = []

    for z in ZIARNA:
        rng = np.random.default_rng(z)
        ukryci = set(rng.choice(neuro, size=int(round(UDZIAL_UKRYTYCH * len(neuro))), replace=False))
        widoczni = set(neuro) - ukryci
        widziana_rek = (df[KOL_ETYKIETA].eq(1) & ~df[KOL_PACJENT].isin(ukryci)).astype(int).to_numpy()

        def do_puli(pacjenci: np.ndarray, s: np.ndarray) -> pd.Series:
            w = pd.Series(s).groupby(pacjenci).median()
            return w[~w.index.isin(widoczni)]

        for f in range(n_fold):
            tr, te = fold_rek != f, fold_rek == f
            p_te = df.loc[te, KOL_PACJENT].to_numpy()
            pula = np.unique(p_te[~np.isin(p_te, list(widoczni))])
            ukryci_te = set(pula) & ukryci
            ukrytych_na_fold.append(len(ukryci_te))
            los = np.random.default_rng(1000 + 10 * z + f)

            wyniki_f = {"losowy ranking pacjentów": pd.Series(los.random(len(pula)), index=pula),
                        "losowy wynik rekordów, mediana do pacjenta": do_puli(p_te, los.random(te.sum()))}
            for nazwa, X in rekordowe.items():
                s = _model().fit(X[tr], widziana_rek[tr]).predict_proba(X[te])[:, 1]
                wyniki_f[nazwa] = do_puli(p_te, s)
            imp_tr, imp_te = imputowane[f]
            y_imp = (imp_tr[KOL_ETYKIETA].eq(1) & ~imp_tr[KOL_PACJENT].isin(ukryci)).astype(int)
            s = _model().fit(imp_tr[feat], y_imp).predict_proba(imp_te[feat])[:, 1]
            wyniki_f["38 cech po imputacji MICE (zapisane foldy)"] = do_puli(imp_te[KOL_PACJENT].to_numpy(), s)
            for nazwa, X in profile.items():
                y_pac = (X.index.isin(widoczni)).astype(int)
                m = _model().fit(X[fold_pac != f], y_pac[fold_pac != f])
                s = pd.Series(m.predict_proba(X[fold_pac == f])[:, 1], index=X.index[fold_pac == f])
                wyniki_f[nazwa] = s[~s.index.isin(widoczni)]

            for nazwa, w in wyniki_f.items():
                assert set(w.index) == set(pula), nazwa
                for q in Q:
                    wyn[nazwa][q].append(recall_hidden(w, ukryci_te, q))
        print(f"  ziarno {z}: gotowe", flush=True)

    tabela = {k: {f"q_{int(round(q * 100))}%": _podsumuj(v[q]) for q in Q} for k, v in wyn.items()}
    protokol = {"udzial_ukrytych": UDZIAL_UKRYTYCH, "ziarna": list(ZIARNA), "q": list(Q),
                "przebiegi_na_model": len(ZIARNA) * n_fold,
                "ukrytych_na_fold_srednio": round(float(np.mean(ukrytych_na_fold)), 1),
                "pula": "pacjenci KOR i ukryci NEURO z części testowej foldu",
                "agregacja_rekordow": "mediana wyników rekordów pacjenta"}
    return tabela, protokol


def b3_bez_imputacji(df: pd.DataFrame, feat: list, przydzial: pd.Series) -> dict:
    od, do = wspolne_okno_dat(df)
    fold_rek = df[KOL_PACJENT].map(przydzial).to_numpy()
    w = df[KOL_DATA].between(od, do).to_numpy()
    rek, pac = [], []
    for f in sorted(przydzial.unique()):
        tr, te = w & (fold_rek != f), w & (fold_rek == f)
        y_te = df.loc[te, KOL_ETYKIETA]
        s = _model().fit(df.loc[tr, feat], df.loc[tr, KOL_ETYKIETA]).predict_proba(df.loc[te, feat])[:, 1]
        rek.append(roc_auc_score(y_te, s))
        d = (pd.DataFrame({"p": df.loc[te, KOL_PACJENT].to_numpy(), "s": s, "y": y_te.to_numpy()})
             .groupby("p").agg(s=("s", "median"), y=("y", "max")))
        pac.append(roc_auc_score(d["y"], d["s"]))
    return {"opis": "NEURO vs KOR, tylko rekordy ze wspólnego okna dat, 38 cech z brakami jako NaN (bez imputacji)",
            "okno": [str(od.date()), str(do.date())],
            "auc_rekordy_srednia": round(float(np.mean(rek)), 3),
            "auc_pacjenci_srednia": round(float(np.mean(pac)), 3),
            "auc_pacjenci_min_max": [round(float(min(pac)), 3), round(float(max(pac)), 3)]}


def main() -> None:
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    feat = kolumny_cech(df, "38")

    print("Odzyskiwanie ukrytych pozytywnych przez modele kontrolne ...", flush=True)
    tabela, protokol = odzyskiwanie_ukrytych(df, feat, przydzial)
    print("B3 bez imputacji ...", flush=True)
    b3 = b3_bez_imputacji(df, feat, przydzial)

    wynik = {"opis": "Kontrola: RecallHidden@q modeli kontrolnych (skróty) i B3 bez imputacji",
             "protokol_orientacyjny": protokol, "recall_hidden": tabela, "B3_bez_imputacji": b3,
             "odcisk_danych": meta["odcisk_danych_sha256"],
             "wersje": {"sklearn": sklearn.__version__, "pandas": pd.__version__}}
    (WYNIKI / "kontrola_skrotow.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")

    print(f"\nRecallHidden@q, {protokol['udzial_ukrytych']:.0%} ukrytych, {protokol['przebiegi_na_model']} przebiegów "
          f"na model (średnio {protokol['ukrytych_na_fold_srednio']} ukrytych na fold):")
    print(f"  {'model':52s} " + "  ".join(f"q={k[2:]:>4s}" for k in next(iter(tabela.values()))))
    for nazwa, v in tabela.items():
        print(f"  {nazwa:52s} " + "  ".join(f"{x['srednia']:6.3f}" for x in v.values()))
    print(f"\nB3 bez imputacji: AUC rekordy {b3['auc_rekordy_srednia']}, pacjenci {b3['auc_pacjenci_srednia']} "
          f"{b3['auc_pacjenci_min_max']}")
    print(f"\nZapisano -> {(WYNIKI / 'kontrola_skrotow.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
