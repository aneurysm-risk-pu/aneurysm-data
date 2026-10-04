"""
porownanie_cech.py
==================

CZĘŚĆ B etapu 3 — jednorazowa analiza porównawcza 38 vs 35 cech (04.10.2026),
podkładka pod decyzję o jednym zestawie (38). Nie jest częścią standardowego
przebiegu uruchom_sgkf.py; odtworzenie:

    python 3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py --zapisz --cechy 38
    python 3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py --zapisz --cechy 35
    python 3-sgkf-split/sgkf/porownanie_cech.py

Czy wybór zestawu cech (38 vs 35, czyli z CRP, MONO, %MONO albo bez) zmienia
imputację wspólnych 35 cech? Decyzja 0.4 z 04.10.2026: liczymy oba warianty
na tym samym podziale i sprawdzamy wpływ.

Dwie miary:

  1. Test kontrolowany (jak benchmark imputacji z etapu 2): kompletne wiersze
     treningu foldu 0, maska 10% komórek WYŁĄCZNIE w 35 wspólnych cechach,
     MICE z parametrami produkcyjnymi uczony raz na 38, raz na 35 kolumnach.
     RMSE/MAE w skali [0,1] na zamaskowanych komórkach, ogółem i per kohorta.
     Mówi, czy 3 dodatkowe kolumny pomagają odtwarzać pozostałe.

  2. Faktyczne foldy (wymaga wcześniejszego uruchomienia pipeline'u z --zapisz
     dla obu wariantów): dla komórek, które były brakami, różnica wartości
     uzupełnionych w wariancie 38 i 35, w odchyleniach standardowych cechy.
     Mówi, jak bardzo różnią się dane, które trafią do modelu.

Uruchomienie:
    python 3-sgkf-split/sgkf/porownanie_cech.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aneurysm_sgkf_mice_pipeline import MICE_PARAMS, RANDOM_STATE, build_mice_imputer  # noqa: E402
from podzial import (CECHY_POZA_35, KOL_ETYKIETA, KOL_PACJENT, WYNIKI,  # noqa: E402
                     kolumny_cech, wczytaj_podzial, wczytaj_rekordy)

warnings.filterwarnings("ignore")

UDZIAL_MASKI = 0.10
SEED_MASKI = 42


def test_kontrolowany(df: pd.DataFrame, przydzial: pd.Series, fold: int = 0) -> dict:
    cechy38 = kolumny_cech(df, "38")
    cechy35 = kolumny_cech(df, "35")
    train = df[df[KOL_PACJENT].map(przydzial).ne(fold) & df[KOL_PACJENT].isin(przydzial.index)]
    kompletne = train.dropna(subset=cechy38).reset_index(drop=True)
    etykieta = kompletne[KOL_ETYKIETA].to_numpy()

    X = MinMaxScaler().fit_transform(kompletne[cechy38])
    idx35 = [cechy38.index(c) for c in cechy35]
    rng = np.random.default_rng(SEED_MASKI)
    maska = np.zeros_like(X, dtype=bool)
    maska[:, idx35] = rng.random((len(X), len(idx35))) < UDZIAL_MASKI

    wyniki = {"fold": fold, "wiersze_kompletne": int(len(X)),
              "wiersze_neuro": int(etykieta.sum()), "zamaskowane_komorki": int(maska.sum())}
    imputowane = {}
    for wariant, kolumny in (("38", list(range(len(cechy38)))), ("35", idx35)):
        Xm = X[:, kolumny].copy()
        m = maska[:, kolumny]
        Xm[m] = np.nan
        t0 = time.time()
        Xi = build_mice_imputer(RANDOM_STATE, MICE_PARAMS).fit_transform(Xm)
        # wyniki tylko na 35 wspólnych kolumnach, w tej samej kolejności
        pozycje = [kolumny.index(i) for i in idx35]
        imputowane[wariant] = Xi[:, pozycje]
        print(f"    wariant {wariant}: MICE {time.time() - t0:.0f} s")

    prawda = X[:, idx35]
    m35 = maska[:, idx35]
    for wariant, Xi in imputowane.items():
        blad = (Xi - prawda)[m35]
        kohorta = np.broadcast_to(etykieta[:, None], m35.shape)[m35]
        wyniki[f"rmse_{wariant}"] = float(np.sqrt(np.mean(blad ** 2)))
        wyniki[f"mae_{wariant}"] = float(np.mean(np.abs(blad)))
        for k, nazwa in ((0, "kor"), (1, "neuro")):
            wyniki[f"rmse_{wariant}_{nazwa}"] = float(np.sqrt(np.mean(blad[kohorta == k] ** 2)))

    # cechy, którym 3 dodatkowe kolumny pomagają najbardziej
    per_cecha = {}
    for j, c in enumerate(cechy35):
        mj = m35[:, j]
        r38 = np.sqrt(np.mean((imputowane["38"][mj, j] - prawda[mj, j]) ** 2))
        r35 = np.sqrt(np.mean((imputowane["35"][mj, j] - prawda[mj, j]) ** 2))
        per_cecha[c] = {"rmse_38": float(r38), "rmse_35": float(r35), "roznica": float(r35 - r38)}
    wyniki["per_cecha_najwieksze_roznice"] = dict(
        sorted(per_cecha.items(), key=lambda kv: -abs(kv[1]["roznica"]))[:5])
    return wyniki


def porownanie_foldow(df: pd.DataFrame, n_foldow: int) -> dict:
    k38, k35 = WYNIKI / "foldy_imputowane_38", WYNIKI / "foldy_imputowane_35"
    if not (k38.exists() and k35.exists()):
        return {"pominiete": "brak zapisanych foldów obu wariantów"}
    cechy35 = kolumny_cech(df, "35")
    braki = df.set_index("custom_id")[cechy35].isna()
    odch = df[cechy35].std()
    wiersze = []
    for f in range(n_foldow):
        for czesc in ("train", "test"):
            a = pd.read_parquet(k38 / f"fold{f}_{czesc}.parquet").set_index("custom_id")
            b = pd.read_parquet(k35 / f"fold{f}_{czesc}.parquet").set_index("custom_id")
            b = b.loc[a.index]
            m = braki.loc[a.index].to_numpy()
            roznica = ((a[cechy35] - b[cechy35]).abs() / odch).to_numpy()
            obserwowane_rowne = np.allclose(a[cechy35].to_numpy()[~m], b[cechy35].to_numpy()[~m])
            etykieta = np.broadcast_to(a[KOL_ETYKIETA].to_numpy()[:, None], m.shape)
            wiersze.append({
                "fold": f, "czesc": czesc,
                "uzupelnione_komorki": int(m.sum()),
                "srednia_roznica_sd": float(roznica[m].mean()),
                "mediana_roznica_sd": float(np.median(roznica[m])),
                "udzial_roznic_ponad_0_5_sd": float((roznica[m] > 0.5).mean()),
                "srednia_roznica_sd_kor": float(roznica[m & (etykieta == 0)].mean()),
                "srednia_roznica_sd_neuro": float(roznica[m & (etykieta == 1)].mean()),
                "obserwowane_identyczne": bool(obserwowane_rowne),
            })
    t = pd.DataFrame(wiersze)
    return {"per_fold": wiersze,
            "srednia_roznica_sd_test": float(t.loc[t.czesc == "test", "srednia_roznica_sd"].mean()),
            "udzial_roznic_ponad_0_5_sd_test": float(t.loc[t.czesc == "test", "udzial_roznic_ponad_0_5_sd"].mean()),
            "obserwowane_identyczne_wszedzie": bool(t["obserwowane_identyczne"].all())}


def main() -> None:
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    print(f"Dodatkowe kolumny wariantu 38: {', '.join(CECHY_POZA_35)}\n")

    print("[1] Test kontrolowany na kompletnych wierszach treningu foldu 0 ...")
    kontrola = test_kontrolowany(df, przydzial)
    for w in ("38", "35"):
        print(f"    {w} cech: RMSE {kontrola[f'rmse_{w}']:.4f} (KOR {kontrola[f'rmse_{w}_kor']:.4f}, "
              f"NEURO {kontrola[f'rmse_{w}_neuro']:.4f}), MAE {kontrola[f'mae_{w}']:.4f}")
    print(f"    różnica RMSE (35 - 38): {kontrola['rmse_35'] - kontrola['rmse_38']:+.4f}")

    print("\n[2] Wartości uzupełnione w zapisanych foldach: 38 vs 35 ...")
    foldy = porownanie_foldow(df, przydzial.nunique())
    if "pominiete" in foldy:
        print(f"    pominięte: {foldy['pominiete']}")
    else:
        print(f"    średnia |różnica| uzupełnionych wartości (test): {foldy['srednia_roznica_sd_test']:.3f} SD")
        print(f"    udział różnic > 0,5 SD (test): {foldy['udzial_roznic_ponad_0_5_sd_test']:.2%}")
        print(f"    wartości obserwowane identyczne w obu wariantach: {foldy['obserwowane_identyczne_wszedzie']}")

    katalog = WYNIKI / "porownanie_38_35"
    katalog.mkdir(parents=True, exist_ok=True)
    (katalog / "porownanie_cech_38_35.json").write_text(
        json.dumps({"test_kontrolowany": kontrola, "foldy": foldy}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nZapisano -> {WYNIKI.name}/porownanie_38_35/porownanie_cech_38_35.json")


if __name__ == "__main__":
    main()
