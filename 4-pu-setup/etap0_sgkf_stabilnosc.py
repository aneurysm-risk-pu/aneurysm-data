"""
etap0_sgkf_stabilnosc.py
========================

Punkt 0.6 z 02_PLAN_PRZED_MODELOWANIEM.md — siatka stabilnosci podzialu.

Odpowiada na trzy pytania, ktore trzeba rozstrzygnac przed zamrozeniem foldow:

  1. Czy wlaczenie shuffle zmienia podzial i czy wynik jest powtarzalny
     miedzy ziarnami?
  2. Czy podzial po rekordach (obecny) rozni sie od podzialu po pacjentach?
  3. Czy foldy sa zbalansowane pod wzgledem rozkladu lat — po diagnostyce
     rozjazdu czasowego kohort to jest osobny warunek poprawnosci.

Wariant etykiety pacjentowej (decyzja 0.1) jest parametrem, wiec siatka
liczy sie dla obu opcji i decyzja moze zapasc na liczbach.

Nic nie modyfikuje i nic nie zapisuje poza tabela wynikow w results/.

Uruchomienie:
    python 4-pu-setup/etap0_sgkf_stabilnosc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "aneurysm_concatted_cleaned.csv"
OUTPUT_DIR = Path(__file__).parent / "results"

TARGET_COL = "label"
GROUP_COL = "patient_id"
DATE_COL = "examination_date"

SEEDS = [42, 7, 13, 101, 2024]
N_SPLITS_GRID = [3, 5, 10]
ERA_YEARS = {2020, 2021}          # okno, w ktorym mieszka 99,5% rekordow KOR


def naglowek(tekst: str) -> None:
    print()
    print("=" * 78)
    print(f"  {tekst}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Etykieta pacjentowa — dwa warianty decyzji 0.1
# ---------------------------------------------------------------------------

def tabela_pacjentow(df: pd.DataFrame, wariant: str) -> pd.DataFrame:
    """Jeden wiersz na pacjenta: etykieta pacjentowa + kontekst do kontroli foldow.

    wariant 'pozytywni' : pacjent mieszany dostaje label=1 (byl w NEURO)
    wariant 'bez_mieszanych' : pacjenci mieszani wypadaja z analizy
    """
    g = df.groupby(GROUP_COL)
    pac = pd.DataFrame({
        "n_rek": g.size(),
        "n_etykiet": g[TARGET_COL].nunique(),
        "label_pac": g[TARGET_COL].max(),
        "udzial_ery": g["rok"].apply(lambda s: s.isin(ERA_YEARS).mean()),
    }).reset_index()
    pac["mieszany"] = pac["n_etykiet"] > 1

    if wariant == "bez_mieszanych":
        pac = pac[~pac["mieszany"]].copy()
    elif wariant != "pozytywni":
        raise ValueError(wariant)
    return pac


# ---------------------------------------------------------------------------
# Jeden przebieg podzialu -> statystyki foldow
# ---------------------------------------------------------------------------

def statystyki_foldow(df: pd.DataFrame, pac: pd.DataFrame, podzialy, poziom: str) -> dict:
    """Zbiera to, co musi byc stabilne: rozmiary, udzial klasy, rozklad lat."""
    mapa_label = dict(zip(pac[GROUP_COL], pac["label_pac"]))
    mapa_ery = dict(zip(pac[GROUP_COL], pac["udzial_ery"]))

    udzial_poz, rozmiar, udzial_ery, nakladanie = [], [], [], 0
    for pac_train, pac_test in podzialy:
        wsp = set(pac_train) & set(pac_test)
        nakladanie += len(wsp)
        etyk = np.array([mapa_label[p] for p in pac_test])
        udzial_poz.append(etyk.mean())
        rozmiar.append(len(pac_test))
        udzial_ery.append(float(np.mean([mapa_ery[p] for p in pac_test])))

    return {
        "poziom": poziom,
        "udzial_poz_min": min(udzial_poz),
        "udzial_poz_max": max(udzial_poz),
        "udzial_poz_rozstep": max(udzial_poz) - min(udzial_poz),
        "rozmiar_min": min(rozmiar),
        "rozmiar_max": max(rozmiar),
        "rozmiar_rozstep_wzgl": (max(rozmiar) - min(rozmiar)) / np.mean(rozmiar),
        "udzial_ery_min": min(udzial_ery),
        "udzial_ery_max": max(udzial_ery),
        "pacjenci_wspolni": nakladanie,
    }


def podzial_rekordowy(df: pd.DataFrame, pac: pd.DataFrame, n_splits: int,
                      shuffle: bool, seed):
    """Obecne podejscie: SGKF na rekordach, grupy = pacjent, stratyfikacja po rekordach."""
    d = df[df[GROUP_COL].isin(set(pac[GROUP_COL]))]
    kw = dict(n_splits=n_splits, shuffle=shuffle)
    if shuffle:
        kw["random_state"] = seed
    sgkf = StratifiedGroupKFold(**kw)
    for tr, te in sgkf.split(d, d[TARGET_COL], groups=d[GROUP_COL]):
        yield (d.iloc[tr][GROUP_COL].unique(), d.iloc[te][GROUP_COL].unique())


def podzial_pacjentowy(df: pd.DataFrame, pac: pd.DataFrame, n_splits: int,
                       shuffle: bool, seed):
    """Alternatywa: jeden wiersz na pacjenta, zwykly StratifiedKFold po etykiecie pacjentowej."""
    kw = dict(n_splits=n_splits, shuffle=shuffle)
    if shuffle:
        kw["random_state"] = seed
    skf = StratifiedKFold(**kw)
    idx = pac[GROUP_COL].to_numpy()
    for tr, te in skf.split(idx, pac["label_pac"]):
        yield (idx[tr], idx[te])



# ---------------------------------------------------------------------------
# Czy shuffle w ogole zmienia sklad foldow
# ---------------------------------------------------------------------------

def przypisanie_do_foldow(pac: pd.DataFrame, podzialy) -> pd.Series:
    """patient_id -> numer foldu testowego."""
    przypis = {}
    for i, (_, pac_test) in enumerate(podzialy):
        for p in pac_test:
            przypis[p] = i
    return pd.Series(przypis).sort_index()


def analiza_wplywu_shuffle(df: pd.DataFrame, pac: pd.DataFrame, n_splits: int = 5) -> None:
    naglowek(f"Czy shuffle zmienia sklad foldow (n_splits={n_splits}, wariant pozytywni)")
    print("\nAdjusted Rand Index: 1.00 = identyczny podzial, ~0.00 = podzial niezalezny\n")

    for poziom, fn in [("rekordowy", podzial_rekordowy), ("pacjentowy", podzial_pacjentowy)]:
        bez = przypisanie_do_foldow(pac, fn(df, pac, n_splits, False, None))
        zs = {s: przypisanie_do_foldow(pac, fn(df, pac, n_splits, True, s)) for s in SEEDS}

        vs_bez = [adjusted_rand_score(bez.loc[z.index], z) for z in zs.values()]
        pary = [adjusted_rand_score(zs[a].loc[zs[b].index], zs[b])
                for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]

        print(f"  {poziom:11s} shuffle=False vs ziarna : ARI srednio {np.mean(vs_bez):.4f}")
        print(f"  {poziom:11s} ziarno vs ziarno        : ARI srednio {np.mean(pary):.4f}")


# ---------------------------------------------------------------------------

def main():
    print(f"Wczytywanie {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    df["rok"] = pd.to_datetime(df[DATE_COL]).dt.year
    print(f"  {df.shape[0]:,} wierszy, {df[GROUP_COL].nunique():,} pacjentow")

    wyniki = []
    for wariant in ["pozytywni", "bez_mieszanych"]:
        pac = tabela_pacjentow(df, wariant)
        naglowek(f"Wariant etykiety pacjentowej: {wariant}")
        print(f"  pacjentow: {len(pac):,} | pozytywnych: {int(pac['label_pac'].sum()):,} "
              f"({pac['label_pac'].mean():.2%})")

        for n_splits in N_SPLITS_GRID:
            for poziom, fn in [("rekordowy", podzial_rekordowy),
                               ("pacjentowy", podzial_pacjentowy)]:
                # shuffle=False liczymy raz - seed nie ma wplywu
                przebiegi = [(False, None)] + [(True, s) for s in SEEDS]
                for shuffle, seed in przebiegi:
                    st = statystyki_foldow(df, pac, fn(df, pac, n_splits, shuffle, seed), poziom)
                    st.update(wariant=wariant, n_splits=n_splits,
                              shuffle=shuffle, seed=seed)
                    wyniki.append(st)

    out = pd.DataFrame(wyniki)
    OUTPUT_DIR.mkdir(exist_ok=True)
    sciezka = OUTPUT_DIR / "etap0_sgkf_stabilnosc.csv"
    out.to_csv(sciezka, index=False)

    naglowek("Wynik: rozstep udzialu klasy pozytywnej miedzy foldami")
    print("(im mniejszy, tym rowniej rozlozone sa pozytywne przypadki)\n")
    piw = out.pivot_table(index=["wariant", "n_splits", "poziom"],
                          columns="shuffle", values="udzial_poz_rozstep")
    print(piw.to_string(float_format=lambda v: f"{v:.4f}"))

    naglowek("Wynik: powtarzalnosc miedzy ziarnami (shuffle=True)")
    print("(odchylenie udzialu klasy pozytywnej miedzy 5 ziarnami)\n")
    z = out[out.shuffle].groupby(["wariant", "n_splits", "poziom"])["udzial_poz_rozstep"]
    print(z.agg(["mean", "std", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))

    naglowek("Kontrola poprawnosci i rozkladu lat")
    print(f"\nPacjenci wspolni miedzy train a test (musi byc 0): "
          f"{out['pacjenci_wspolni'].sum()}")
    print(f"Najwiekszy rozstep udzialu rekordow z lat 2020-2021 miedzy foldami: "
          f"{(out['udzial_ery_max'] - out['udzial_ery_min']).max():.4f}")
    print(f"Najwiekszy wzgledny rozstep rozmiaru foldu: "
          f"{out['rozmiar_rozstep_wzgl'].max():.4f}")
    analiza_wplywu_shuffle(df, tabela_pacjentow(df, "pozytywni"))

    print(f"\nPelna tabela -> {sciezka.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
