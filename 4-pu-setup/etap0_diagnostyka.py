"""
etap0_diagnostyka.py
=====================

Diagnostyka do etapu 0 z 02_PLAN_PRZED_MODELOWANIEM.md — dostarcza danych
do rozstrzygnięć, które muszą zapaść przed pisaniem pipeline'u modelowania.

Nic nie modyfikuje. Tylko czyta dane i liczy statystyki:

  0.1  63 pacjentów występujących jednocześnie w KOR i NEURO
  0.2  chronologia: czy pomiary NEURO wyglądają na sprzed diagnozy
  0.3  agregacja rekordów do pacjenta — czy wybór reguły ma znaczenie
  0.5  wartości skrajne w WBC, Na, K, KREA

Uruchomienie:
    python 4-pu-setup/etap0_diagnostyka.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "aneurysm_concatted_cleaned.csv"

TARGET_COL = "label"
GROUP_COL = "patient_id"
DATE_COL = "examination_date"


def naglowek(tekst: str) -> None:
    print()
    print("=" * 78)
    print(f"  {tekst}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 0.1 — pacjenci z obiema etykietami
# ---------------------------------------------------------------------------

def analiza_pacjentow_mieszanych(df: pd.DataFrame) -> pd.DataFrame:
    naglowek("0.1  Pacjenci wystepujacy jednoczesnie w KOR i NEURO")

    etykiety = df.groupby(GROUP_COL)[TARGET_COL].nunique()
    mieszani = etykiety[etykiety > 1].index
    print(f"\nLiczba pacjentow z obiema etykietami: {len(mieszani)}")

    sub = df[df[GROUP_COL].isin(mieszani)].copy()
    sub[DATE_COL] = pd.to_datetime(sub[DATE_COL])

    wiersze = []
    for pid, g in sub.groupby(GROUP_COL):
        kor = g[g[TARGET_COL] == 0]
        neuro = g[g[TARGET_COL] == 1]
        wiersze.append({
            "patient_id": pid,
            "rek_kor": len(kor),
            "rek_neuro": len(neuro),
            "kor_od": kor[DATE_COL].min(),
            "kor_do": kor[DATE_COL].max(),
            "neuro_od": neuro[DATE_COL].min(),
            "neuro_do": neuro[DATE_COL].max(),
        })
    info = pd.DataFrame(wiersze)

    # Relacja czasowa: czy rekordy KOR poprzedzaja NEURO?
    info["kor_przed_neuro"] = info["kor_do"] < info["neuro_od"]
    info["neuro_przed_kor"] = info["neuro_do"] < info["kor_od"]
    info["przeplatane"] = ~(info["kor_przed_neuro"] | info["neuro_przed_kor"])

    print(f"\nRekordy na pacjenta (mediana): KOR={info['rek_kor'].median():.0f}, "
          f"NEURO={info['rek_neuro'].median():.0f}")
    print(f"Laczna liczba wierszy tych pacjentow: {len(sub):,} "
          f"(KOR: {info['rek_kor'].sum():,}, NEURO: {info['rek_neuro'].sum():,})")

    print("\nRelacja czasowa miedzy rekordami KOR i NEURO tego samego pacjenta:")
    print(f"  wszystkie KOR przed NEURO : {info['kor_przed_neuro'].sum():3d}  "
          f"<- pacjent 'przeszedl' z populacji ogolnej do chorych")
    print(f"  wszystkie NEURO przed KOR : {info['neuro_przed_kor'].sum():3d}  "
          f"<- badania po rozpoznaniu")
    print(f"  przeplatane w czasie      : {info['przeplatane'].sum():3d}")

    odstep = (info.loc[info["kor_przed_neuro"], "neuro_od"]
              - info.loc[info["kor_przed_neuro"], "kor_do"]).dt.days
    if len(odstep):
        print(f"\nDla 'KOR przed NEURO' odstep miedzy ostatnim KOR a pierwszym NEURO:")
        print(f"  mediana {odstep.median():.0f} dni, min {odstep.min()} dni, max {odstep.max()} dni")

    return info


# ---------------------------------------------------------------------------
# 0.2 — chronologia pomiarow
# ---------------------------------------------------------------------------

def analiza_chronologii(df: pd.DataFrame) -> None:
    naglowek("0.2  Chronologia pomiarow — czy NEURO to badania sprzed diagnozy")

    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])

    for nazwa, kod in [("KOR  (label=0)", 0), ("NEURO(label=1)", 1)]:
        g = d[d[TARGET_COL] == kod]
        print(f"\n{nazwa}")
        print(f"  zakres dat        : {g[DATE_COL].min().date()} .. {g[DATE_COL].max().date()}")
        rozpietosc = g.groupby(GROUP_COL)[DATE_COL].agg(lambda s: (s.max() - s.min()).days)
        wielokrotni = rozpietosc[g.groupby(GROUP_COL).size() > 1]
        print(f"  pacjentow         : {g[GROUP_COL].nunique():,}")
        print(f"  z >1 rekordem     : {(g.groupby(GROUP_COL).size() > 1).sum():,}")
        if len(wielokrotni):
            print(f"  rozpietosc badan u pacjentow z >1 rekordem (dni):")
            print(f"     mediana {wielokrotni.median():.0f} | "
                  f"srednia {wielokrotni.mean():.0f} | max {wielokrotni.max():.0f}")
            print(f"     odsetek miesczacych sie w 30 dniach: "
                  f"{(wielokrotni <= 30).mean():.1%}")

    print("\nUwaga interpretacyjna:")
    print("  Jesli pomiary NEURO skupiaja sie w krotkim oknie, przypomina to")
    print("  pojedynczy epizod hospitalizacji — czyli badania z okresu diagnozy")
    print("  lub po niej, a nie profil ryzyka sprzed rozpoznania.")


# ---------------------------------------------------------------------------
# 0.3 — agregacja rekordow do pacjenta
# ---------------------------------------------------------------------------

def analiza_agregacji(df: pd.DataFrame, feat_cols: list) -> None:
    naglowek("0.3  Agregacja rekordow do pacjenta — czy wybor reguly ma znaczenie")

    liczba = df.groupby(GROUP_COL).size()
    print(f"\nPacjentow ogolem            : {len(liczba):,}")
    print(f"  z 1 rekordem (bez wplywu) : {(liczba == 1).sum():,} ({(liczba == 1).mean():.1%})")
    print(f"  z >1 rekordem (wybor wazy): {(liczba > 1).sum():,} ({(liczba > 1).mean():.1%})")

    # Jak bardzo srednia rozni sie od maksimum u pacjentow z wieloma rekordami
    wielo = df[df[GROUP_COL].isin(liczba[liczba > 1].index)]
    probka = [c for c in ["WBC", "GLU", "KREA", "PLT"] if c in feat_cols]

    print("\nRozrzut wartosci w obrebie jednego pacjenta (tylko >1 rekord):")
    print(f"  {'cecha':8s} {'mediana |max-min|':>18s} {'mediana odch.std':>18s}")
    for c in probka:
        g = wielo.groupby(GROUP_COL)[c]
        rozstep = (g.max() - g.min()).dropna()
        odch = g.std().dropna()
        print(f"  {c:8s} {rozstep.median():18.2f} {odch.median():18.2f}")

    print("\n  Im wiekszy rozrzut, tym mocniej wynik zalezy od tego, czy wezmiemy")
    print("  srednia, mediane, maksimum czy ostatni pomiar.")


# ---------------------------------------------------------------------------
# 0.5 — wartosci skrajne
# ---------------------------------------------------------------------------

ZAKRESY_FIZJOLOGICZNE = {
    # cecha: (dolna granica przezycia, gorna granica przezycia) - orientacyjne
    "WBC":  (0.1, 200.0),
    "Na":   (100.0, 180.0),
    "K":    (1.5, 9.0),
    "KREA": (0.1, 2000.0),
}


def analiza_wartosci_skrajnych(df: pd.DataFrame) -> None:
    naglowek("0.5  Wartosci skrajne: WBC, Na, K, KREA")

    for c, (lo, hi) in ZAKRESY_FIZJOLOGICZNE.items():
        if c not in df.columns:
            print(f"\n{c}: BRAK kolumny w zbiorze")
            continue
        s = df[c].dropna()
        ponizej = (s < lo).sum()
        powyzej = (s > hi).sum()
        print(f"\n{c}  (n={len(s):,}, braki: {df[c].isna().sum():,})")
        print(f"  min {s.min():10.2f} | p1 {s.quantile(0.01):8.2f} | mediana {s.median():8.2f} "
              f"| p99 {s.quantile(0.99):8.2f} | max {s.max():10.2f}")
        print(f"  poza orientacyjnym zakresem przezycia [{lo}, {hi}]: "
              f"{ponizej + powyzej} wartosci ({ponizej} ponizej, {powyzej} powyzej)")
        if ponizej + powyzej:
            poza = s[(s < lo) | (s > hi)]
            pac = df.loc[poza.index, GROUP_COL].nunique()
            print(f"    dotyczy {pac} pacjentow; przykladowe wartosci: "
                  f"{sorted(poza.unique())[:5]} ... {sorted(poza.unique())[-3:]}")


# ---------------------------------------------------------------------------

def main():
    print(f"Wczytywanie {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    meta = [GROUP_COL, "custom_id", DATE_COL, TARGET_COL]
    feat_cols = [c for c in df.columns if c not in meta]
    print(f"  {df.shape[0]:,} wierszy x {df.shape[1]} kolumn ({len(feat_cols)} cech)")

    analiza_pacjentow_mieszanych(df)
    analiza_chronologii(df)
    analiza_agregacji(df, feat_cols)
    analiza_wartosci_skrajnych(df)

    print()
    print("=" * 78)
    print("  Koniec diagnostyki. Wnioski -> 4-pu-setup/ETAP0_USTALENIA.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
