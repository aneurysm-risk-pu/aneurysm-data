"""
etap0_agregacja.py
==================

Punkt 0.3 z 02_PLAN_PRZED_MODELOWANIEM.md — wybor reguly agregacji rekordow
do pacjenta.

41,3% pacjentow ma wiecej niz jeden rekord, wiec regula decyduje o tym, jaki
profil trafia do modelu. Skrypt sprawdza cztery kandydatury (srednia, mediana,
maksimum, ostatni pomiar) pod katem czterech ryzyk:

  1. Czy reguly w ogole daja rozne profile — jesli nie, decyzja jest kosmetyczna.
  2. Czy regula przemyca liczbe badan pacjenta. Pacjenci NEURO maja srednio
     dwukrotnie wiecej rekordow niz KOR, wiec kazda regula wrazliwa na liczbe
     pomiarow (przede wszystkim maksimum) moze tworzyc sygnal z samej
     czestosci badania, a nie ze stanu zdrowia.
  3. Czy regula zmienia pozorna sile sygnalu (AUC cechy wzgledem etykiety).
  4. Czy regula wciaga wartosci nierealne fizjologicznie.

Nic nie modyfikuje. Wyniki ladują w results/.

Uruchomienie:
    python 4-pu-setup/etap0_agregacja.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "aneurysm_concatted_cleaned.csv"
OUTPUT_DIR = Path(__file__).parent / "results"

TARGET_COL = "label"
GROUP_COL = "patient_id"
DATE_COL = "examination_date"
META = [GROUP_COL, "custom_id", DATE_COL, TARGET_COL]

REGULY = ["srednia", "mediana", "maksimum", "ostatni"]

# Orientacyjne granice przezycia - te same co w etap0_diagnostyka.py
ZAKRESY_FIZJOLOGICZNE = {"WBC": (0.1, 200.0), "Na": (100.0, 180.0),
                         "K": (1.5, 9.0), "KREA": (0.1, 2000.0)}


def naglowek(tekst: str) -> None:
    print()
    print("=" * 78)
    print(f"  {tekst}")
    print("=" * 78)


def agreguj(df: pd.DataFrame, feat: list, regula: str) -> pd.DataFrame:
    """Jeden wiersz na pacjenta wedlug podanej reguly."""
    if regula == "ostatni":
        d = df.sort_values(DATE_COL).groupby(GROUP_COL)[feat].last()
    else:
        fn = {"srednia": "mean", "mediana": "median", "maksimum": "max"}[regula]
        d = df.groupby(GROUP_COL)[feat].agg(fn)
    return d.sort_index()


def main():
    print(f"Wczytywanie {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    feat = [c for c in df.columns if c not in META]
    print(f"  {len(df):,} rekordow, {df[GROUP_COL].nunique():,} pacjentow, {len(feat)} cech")

    etykieta = df.groupby(GROUP_COL)[TARGET_COL].max().sort_index()
    n_rek = df.groupby(GROUP_COL).size().sort_index()

    # ---------------------------------------------------------------
    naglowek("Ile rekordow ma pacjent — w rozbiciu na klasy")
    st = df.groupby([TARGET_COL, GROUP_COL]).size().groupby(TARGET_COL).agg(
        pacjentow="size", mediana="median", srednia="mean", maks="max")
    st["pow_1_rekord"] = df.groupby([TARGET_COL, GROUP_COL]).size().gt(1).groupby(TARGET_COL).mean()
    st.index = ["KOR (label=0)", "NEURO (label=1)"]
    print()
    print(st.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nJesli klasy roznia sie liczba badan, regula wrazliwa na te liczbe")
    print("(maksimum, w mniejszym stopniu srednia) tworzy sygnal z samej czestosci.")

    # ---------------------------------------------------------------
    profile = {r: agreguj(df, feat, r) for r in REGULY}
    wielokrotni = n_rek[n_rek > 1].index          # tylko oni odczuwaja regule

    naglowek("1. Czy reguly daja rozne profile (pacjenci z >1 rekordem)")
    print("\nKorelacja Spearmana miedzy regulami, mediana po cechach:\n")
    print(f"  {'para regul':28s} {'mediana rho':>12s} {'min rho':>10s}")
    for i, a in enumerate(REGULY):
        for b in REGULY[i + 1:]:
            rho = [spearmanr(profile[a].loc[wielokrotni, c],
                             profile[b].loc[wielokrotni, c],
                             nan_policy="omit").statistic for c in feat]
            rho = [r for r in rho if not np.isnan(r)]
            print(f"  {a + ' vs ' + b:28s} {np.median(rho):12.3f} {min(rho):10.3f}")

    # ---------------------------------------------------------------
    naglowek("2. Czy regula przemyca liczbe badan (tylko pacjenci KOR)")
    print("\nKorelacja Spearmana miedzy liczba rekordow a wartoscia cechy.")
    print("Liczona wewnatrz KOR, wiec nie miesza sie z efektem choroby.\n")
    kor = etykieta[etykieta == 0].index
    kor_wiel = [p for p in kor if n_rek[p] > 1]
    print(f"  {'regula':12s} {'mediana |rho|':>14s} {'maks |rho|':>12s} {'cech |rho|>0.2':>16s}")
    wiersze_2 = []
    for r in REGULY:
        rho = [abs(spearmanr(n_rek[kor_wiel], profile[r].loc[kor_wiel, c],
                             nan_policy="omit").statistic) for c in feat]
        rho = np.array([x for x in rho if not np.isnan(x)])
        print(f"  {r:12s} {np.median(rho):14.3f} {rho.max():12.3f} {int((rho > 0.2).sum()):16d}")
        wiersze_2.append({"regula": r, "mediana_abs_rho_nrek": np.median(rho),
                          "maks_abs_rho_nrek": rho.max(), "cech_rho_pow_02": int((rho > 0.2).sum())})

    # ---------------------------------------------------------------
    naglowek("3. Czy regula zmienia pozorna sile sygnalu")
    print("\nAUC pojedynczej cechy wzgledem etykiety pacjentowej (wszyscy pacjenci).")
    print("Odchylenie od 0.5 to sila zwiazku; interesuje nas roznica miedzy regulami.\n")
    auc_tab = {}
    for r in REGULY:
        aucs = []
        for c in feat:
            v = profile[r][c]
            maska = v.notna()
            if maska.sum() > 100 and etykieta[maska].nunique() == 2:
                aucs.append(abs(roc_auc_score(etykieta[maska], v[maska]) - 0.5))
        auc_tab[r] = np.array(aucs)
        print(f"  {r:12s} mediana |AUC-0.5| {np.median(aucs):.4f} | "
              f"maks {max(aucs):.4f} | cech powyzej 0.10: {int((np.array(aucs) > 0.10).sum())}")

    # ---------------------------------------------------------------
    naglowek("4. Czy regula wciaga wartosci nierealne fizjologicznie")
    print()
    wiersze_4 = []
    for c, (lo, hi) in ZAKRESY_FIZJOLOGICZNE.items():
        if c not in feat:
            continue
        opis = []
        for r in REGULY:
            v = profile[r][c].dropna()
            poza = int(((v < lo) | (v > hi)).sum())
            opis.append(f"{r} {poza}")
            wiersze_4.append({"cecha": c, "regula": r, "pacjentow_poza_zakresem": poza})
        print(f"  {c:6s} pacjentow poza zakresem [{lo}, {hi}]:  " + " | ".join(opis))

    # ---------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(wiersze_2).to_csv(OUTPUT_DIR / "etap0_agregacja_nrek.csv", index=False)
    pd.DataFrame(wiersze_4).to_csv(OUTPUT_DIR / "etap0_agregacja_zakresy.csv", index=False)
    pd.DataFrame({r: pd.Series(a) for r, a in auc_tab.items()}).to_csv(
        OUTPUT_DIR / "etap0_agregacja_auc.csv", index=False)
    print(f"\nTabele -> {OUTPUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
