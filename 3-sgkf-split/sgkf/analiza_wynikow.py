"""
analiza_wynikow.py
==================

CZĘŚĆ B etapu 3 — analiza tego, co wyszło z SGKF (na zapisanych foldach).
To NIE jest model ryzyka tętniaka. To diagnostyka zbioru przygotowanego do
modelowania: czy imputacja jest wiarygodna i na jakich „skrótach” model
mógłby się oprzeć zamiast na chorobie.

  A. Rozkłady uzupełnień per kohorta — wartości uzupełnione vs obserwowane
     w tej samej kohorcie, z naciskiem na cechy z dużymi brakami w NEURO: czy
     uzupełnienia NEURO „przesuwają się” w stronę KOR. To KONTEKST, nie miara
     jakości: prawdziwych brakujących wartości nie znamy, a przy brakach MAR
     uzupełnienia mogą legalnie różnić się od obserwowanych. Błąd imputacji
     mierzy ocena_imputacji.py (protokół z etapu 2).
  B. Diagnostyka skrótów na zamrożonych foldach (HistGradientBoosting, ocena
     na części testowej każdego foldu, AUC):
       B1  NEURO vs KOR na cechach po imputacji         — ile w ogóle da się rozdzielić
       B2  NEURO vs KOR na samym wzorcu braków          — ile mówi samo „czego nie zmierzono”
       B3  NEURO vs KOR tylko we wspólnym oknie dat     — rozdzielność bez różnicy epok
       B4  NEURO z okna dat KOR vs NEURO spoza okna     — ile epoki „siedzi” w wynikach
  C. Profil kohort względem orientacyjnych zakresów referencyjnych — czy KOR
     wygląda na populację ogólną, czy szpitalną.
  D. Zgodność imputacji między foldami.

Wynik: sgkf/results/analiza_wynikow.json

Uruchomienie (po aneurysm_sgkf_mice_pipeline.py --zapisz):
    python 3-sgkf-split/sgkf/analiza_wynikow.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from podzial import (BASE_DIR, KOL_DATA, KOL_ETYKIETA, KOL_PACJENT, WYNIKI,  # noqa: E402
                     kolumny_cech, wczytaj_podzial, wczytaj_rekordy, wspolne_okno_dat)

warnings.filterwarnings("ignore")
FOLDY = WYNIKI / "foldy_imputowane_38"

# Orientacyjne górne / dolne granice zakresów referencyjnych dla dorosłych
# (wartości typowe dla laboratoriów; służą tylko do opisu profilu kohort).
ZAKRESY = {
    "CRP": ("powyżej", 5.0, "mg/l"),
    "WBC": ("powyżej", 10.0, "G/l"),
    "NEUT": ("powyżej", 7.0, "G/l"),
    "GLU": ("powyżej", 99.0, "mg/dl"),
    "HGB": ("poniżej", 12.0, "g/dl"),
    "PLT": ("poniżej", 150.0, "G/l"),
}


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def wczytaj_foldy(n: int) -> list:
    return [(pd.read_parquet(FOLDY / f"fold{f}_train.parquet"), pd.read_parquet(FOLDY / f"fold{f}_test.parquet"))
            for f in range(n)]


# ---------------------------------------------------------------------------
# A. Imputacja per kohorta
# ---------------------------------------------------------------------------

def a_imputacja(test_wszystkie: pd.DataFrame, braki: pd.DataFrame, feat: list) -> dict:
    t = test_wszystkie.set_index("custom_id")
    b = braki.loc[t.index]
    etyk = t[KOL_ETYKIETA]
    wynik = {}
    for c in feat:
        sd = t.loc[~b[c], c].std()
        wiersz = {}
        for k, nazwa in ((0, "kor"), (1, "neuro")):
            m_obs = (~b[c]) & (etyk == k)
            m_imp = b[c] & (etyk == k)
            obs, imp = t.loc[m_obs, c], t.loc[m_imp, c]
            wiersz[nazwa] = {
                "uzupelnione": int(m_imp.sum()), "udzial_uzupelnionych": _r(m_imp.sum() / (etyk == k).sum()),
                "mediana_obserwowanych": _r(obs.median()), "mediana_uzupelnionych": _r(imp.median()) if len(imp) else None,
                "ks_uzupelnione_vs_obserwowane": _r(ks_2samp(imp, obs).statistic) if len(imp) > 20 else None,
            }
        # Gdzie leży mediana uzupełnień NEURO między medianą obserwowanych NEURO (0) a KOR (1)
        mo_n, mo_k = wiersz["neuro"]["mediana_obserwowanych"], wiersz["kor"]["mediana_obserwowanych"]
        mi_n = wiersz["neuro"]["mediana_uzupelnionych"]
        if mi_n is not None and sd and abs(mo_k - mo_n) > 0.1 * sd:
            wiersz["neuro_polozenie_0_neuro_1_kor"] = _r((mi_n - mo_n) / (mo_k - mo_n), 2)
        wiersz["odch_std_cechy"] = _r(sd)
        wynik[c] = wiersz
    return wynik


# ---------------------------------------------------------------------------
# B. Diagnostyka skrótów
# ---------------------------------------------------------------------------

def _model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def _auc_po_foldach(foldy: list, przygotuj) -> dict:
    """przygotuj(train, test) -> (X_tr, y_tr, X_te, y_te, pacjenci_te) albo None."""
    rek, pac = [], []
    for tr, te in foldy:
        dane = przygotuj(tr, te)
        if dane is None:
            continue
        X_tr, y_tr, X_te, y_te, p_te = dane
        if y_te.nunique() < 2:
            continue
        s = _model().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rek.append(roc_auc_score(y_te, s))
        if p_te is not None:
            d = pd.DataFrame({"p": p_te.to_numpy(), "s": s, "y": y_te.to_numpy()}).groupby("p").agg(s=("s", "median"), y=("y", "max"))
            pac.append(roc_auc_score(d["y"], d["s"]))
    out = {"auc_rekordy_srednia": _r(np.mean(rek), 3), "auc_rekordy_min_max": [_r(min(rek), 3), _r(max(rek), 3)]}
    if pac:
        out.update({"auc_pacjenci_srednia": _r(np.mean(pac), 3), "auc_pacjenci_min_max": [_r(min(pac), 3), _r(max(pac), 3)]})
    return out


def b_skroty(foldy: list, braki: pd.DataFrame, feat: list, okno: tuple) -> dict:
    od, do = okno

    def b1(tr, te):
        return tr[feat], tr[KOL_ETYKIETA], te[feat], te[KOL_ETYKIETA], te[KOL_PACJENT]

    def b2(tr, te):
        btr = braki.loc[tr["custom_id"]].astype(int).reset_index(drop=True)
        bte = braki.loc[te["custom_id"]].astype(int).reset_index(drop=True)
        return btr, tr[KOL_ETYKIETA].reset_index(drop=True), bte, te[KOL_ETYKIETA].reset_index(drop=True), \
            te[KOL_PACJENT].reset_index(drop=True)

    def b3(tr, te):
        mtr, mte = tr[KOL_DATA].between(od, do), te[KOL_DATA].between(od, do)
        return tr.loc[mtr, feat], tr.loc[mtr, KOL_ETYKIETA], te.loc[mte, feat], te.loc[mte, KOL_ETYKIETA], te.loc[mte, KOL_PACJENT]

    def b4(tr, te):
        ntr, nte = tr[tr[KOL_ETYKIETA] == 1], te[te[KOL_ETYKIETA] == 1]
        return (ntr[feat], ntr[KOL_DATA].between(od, do).astype(int), nte[feat],
                nte[KOL_DATA].between(od, do).astype(int), None)

    wyn = {}
    for nazwa, fn, opis in (("B1_neuro_vs_kor_cechy", b1, "NEURO vs KOR, 38 cech po imputacji"),
                            ("B2_neuro_vs_kor_wzorzec_brakow", b2, "NEURO vs KOR, tylko wskaźniki braków (0/1)"),
                            ("B3_neuro_vs_kor_wspolne_okno", b3, "NEURO vs KOR, tylko rekordy ze wspólnego okna dat"),
                            ("B4_neuro_epoka", b4, "NEURO z okna dat KOR vs NEURO spoza okna (rekordy)")):
        print(f"    {nazwa} ...", flush=True)
        wyn[nazwa] = {"opis": opis, **_auc_po_foldach(foldy, fn)}
    return wyn


# ---------------------------------------------------------------------------
# C. Profil kohort względem zakresów referencyjnych
# ---------------------------------------------------------------------------

def c_profil(df: pd.DataFrame) -> dict:
    wyn = {}
    for c, (kier, prog, jedn) in ZAKRESY.items():
        w = {}
        for k, nazwa in ((0, "kor"), (1, "neuro")):
            v = df.loc[df[KOL_ETYKIETA] == k, c].dropna()
            poza = (v > prog) if kier == "powyżej" else (v < prog)
            w[nazwa] = {"mediana": _r(v.median(), 2), "udzial_poza_zakresem": _r(poza.mean(), 3), "n": int(len(v))}
        wyn[c] = {"granica": f"{kier} {prog} {jedn}", **w}
    return wyn


# ---------------------------------------------------------------------------
# D. Zgodność imputacji między foldami
# ---------------------------------------------------------------------------

def d_zgodnosc(foldy: list, braki: pd.DataFrame, cechy: list) -> dict:
    wyn = {}
    for c in cechy:
        med = {"kor": [], "neuro": []}
        for _, te in foldy:
            m = braki.loc[te["custom_id"], c].to_numpy()
            for k, nazwa in ((0, "kor"), (1, "neuro")):
                sel = m & (te[KOL_ETYKIETA] == k).to_numpy()
                if sel.sum() > 10:
                    med[nazwa].append(float(np.median(te.loc[sel, c])))
        wyn[c] = {n: {"mediany_per_fold": [_r(x, 3) for x in v], "rozstep": _r(max(v) - min(v), 3) if v else None}
                  for n, v in med.items()}
    return wyn


# ---------------------------------------------------------------------------

def main() -> None:
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    feat = kolumny_cech(df, "38")
    braki = df.set_index("custom_id")[feat].isna()
    foldy = wczytaj_foldy(przydzial.nunique())
    test_wszystkie = pd.concat([te for _, te in foldy], ignore_index=True)
    assert test_wszystkie["custom_id"].is_unique and len(test_wszystkie) == len(df), "części testowe nie pokrywają danych"

    print("[A] Imputacja per kohorta ...")
    a = a_imputacja(test_wszystkie, braki, feat)
    print("[B] Diagnostyka skrótów (5 foldów) ...")
    b = b_skroty(foldy, braki, feat, wspolne_okno_dat(df))
    print("[C] Profil kohort względem zakresów referencyjnych ...")
    c = c_profil(df)
    print("[D] Zgodność imputacji między foldami ...")
    najwiecej_brakow_neuro = sorted(feat, key=lambda x: -a[x]["neuro"]["udzial_uzupelnionych"])[:6]
    d = d_zgodnosc(foldy, braki, najwiecej_brakow_neuro)

    wynik = {"A_imputacja_per_kohorta": a, "B_diagnostyka_skrotow": b, "C_profil_kohort": c,
             "D_zgodnosc_miedzy_foldami": d, "odcisk_danych": meta["odcisk_danych_sha256"]}
    (WYNIKI / "analiza_wynikow.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Najważniejsze liczby")
    for k, v in b.items():
        print(f"  {k:34s} AUC rekordy {v['auc_rekordy_srednia']}" +
              (f" | pacjenci {v['auc_pacjenci_srednia']}" if "auc_pacjenci_srednia" in v else ""))
    print("  Cechy z największymi brakami w NEURO (udział uzupełnionych, położenie mediany uzupełnień 0=NEURO, 1=KOR):")
    for x in najwiecej_brakow_neuro:
        print(f"    {x:10s} {a[x]['neuro']['udzial_uzupelnionych']:.1%}  położenie {a[x].get('neuro_polozenie_0_neuro_1_kor')}")
    print("  Profil KOR / NEURO (udział poza zakresem referencyjnym):")
    for x, v in c.items():
        print(f"    {x:5s} {v['granica']:20s} KOR {v['kor']['udzial_poza_zakresem']:.1%}  NEURO {v['neuro']['udzial_poza_zakresem']:.1%}")
    print(f"\nZapisano -> {(WYNIKI / 'analiza_wynikow.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
