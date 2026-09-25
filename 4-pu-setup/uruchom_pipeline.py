"""
Uruchomienie pipeline'u przygotowania PU (punkty 1, 2, 4, 5).

Tryby:

  artefakty   — buduje i zapisuje to, co ma zostac ZAMROZONE przed treningiem:
                tabele patient_id -> fold, komplet masek ukrywania oraz zrzut
                konfiguracji protokolu. Sekundy, bez imputacji.

  fold        — przetwarza jeden fold zewnetrzny na prawdziwych danych
                (scaler + MICE + agregacja) i sprawdza asercje. Sluzy do
                weryfikacji, ze pipeline dziala end-to-end. Z opcja --podprobka
                liczy sie w minutach zamiast w godzinach.

  demo        — pelny przebieg na podprobce z losowym scoringiem, zeby pokazac,
                ze metryki i funkcja celu licza sie na realnych foldach.
                Wynik NIE jest wynikiem modelu — model powstaje dopiero w
                punkcie 6, w 03_PLAN_MODELOWANIE.md.

Przyklady:
    python 4-pu-setup/uruchom_pipeline.py artefakty
    python 4-pu-setup/uruchom_pipeline.py fold --podprobka 4000
    python 4-pu-setup/uruchom_pipeline.py demo --podprobka 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pu.cel import agreguj_po_foldach, wartosc_celu
from pu.config import DOMYSLNA, KonfiguracjaPU
from pu.dane import KOL_PACJENT, przygotuj
from pu.foldy import foldy_zewnetrzne, podzial_pacjentow, przygotuj_fold, tabela_pacjent_fold
from pu.metryki import komplet_metryk
from pu.ukrywanie import sprawdz_maske, wszystkie_maski

KATALOG_WYNIKOW = Path(__file__).parent / "results"


def naglowek(tekst: str) -> None:
    print()
    print("=" * 78)
    print(f"  {tekst}")
    print("=" * 78)


def zwez_do_podprobki(df: pd.DataFrame, pac: pd.DataFrame, n: int, seed: int = 0):
    """Losuje n pacjentow z zachowaniem proporcji klas — do szybkich przebiegow."""
    if not n or n >= len(pac):
        return df, pac
    rng = np.random.default_rng(seed)
    wybrani = []
    for etykieta, grupa in pac.groupby("true_label"):
        ile = max(2, round(n * len(grupa) / len(pac)))
        wybrani.append(grupa.iloc[rng.choice(len(grupa), min(ile, len(grupa)), replace=False)])
    pac_m = pd.concat(wybrani).sort_values(KOL_PACJENT).reset_index(drop=True)
    return df[df[KOL_PACJENT].isin(set(pac_m[KOL_PACJENT]))].copy(), pac_m


# ---------------------------------------------------------------------------

def tryb_artefakty(cfg: KonfiguracjaPU) -> None:
    df, pac, feat = przygotuj(cfg)
    naglowek("Artefakty do zamrozenia przed treningiem")
    print(f"\n  danych: {len(df):,} rekordow, {len(pac):,} pacjentow, {len(feat)} cech")
    print(f"  pozytywnych pacjentow: {int(pac['true_label'].sum()):,} "
          f"({pac['true_label'].mean():.2%})")

    KATALOG_WYNIKOW.mkdir(exist_ok=True)

    t = tabela_pacjent_fold(pac, cfg, df)
    t.to_csv(KATALOG_WYNIKOW / "pacjent_fold.csv", index=False)
    print(f"\n  [1/3] tabela pacjent -> fold: {len(t):,} wierszy")
    rozklad = t.groupby("fold")["true_label"].agg(["size", "sum", "mean"])
    rozklad.columns = ["pacjentow", "pozytywnych", "udzial"]
    print(rozklad.to_string(float_format=lambda v: f"{v:.4f}"))

    maski = wszystkie_maski(pac, cfg)
    for (u, s), m in maski.groupby(["udzial_ukrycia", "seed_ukrycia"]):
        sprawdz_maske(m)
    maski.to_csv(KATALOG_WYNIKOW / "maski_ukrycia.csv", index=False)
    print(f"\n  [2/3] maski ukrycia: {len(maski):,} wierszy, "
          f"{maski.groupby(['udzial_ukrycia', 'seed_ukrycia']).ngroups} kombinacji")
    podsum = maski.groupby("udzial_ukrycia")["is_hidden"].agg(["sum", "mean"])
    podsum.columns = ["ukrytych_lacznie", "udzial_wsrod_wszystkich"]
    print(podsum.to_string(float_format=lambda v: f"{v:.4f}"))

    (KATALOG_WYNIKOW / "konfiguracja.json").write_text(
        json.dumps(cfg.do_slownika(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  [3/3] konfiguracja protokolu zapisana")
    print(f"\n  wszystko w: {KATALOG_WYNIKOW}")
    print("\n  UWAGA: to sa artefakty ROBOCZE. Zamrozenie nastepuje dopiero po")
    print("  rozstrzygnieciu decyzji 0.1, 0.2 i 0.5 — patrz PYTANIA_DO_PROWADZACEGO.md")


def tryb_fold(cfg: KonfiguracjaPU, podprobka: int) -> None:
    df, pac, feat = przygotuj(cfg)
    df, pac = zwez_do_podprobki(df, pac, podprobka)
    naglowek(f"Jeden fold zewnetrzny na prawdziwych danych ({len(pac):,} pacjentow)")

    tr, te = next(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df))
    print(f"\n  train: {len(tr):,} pacjentow | test: {len(te):,} pacjentow")
    print(f"  braki przed imputacja: {df[feat].isna().mean().mean():.1%} komorek")
    print("\n  scaler + MICE + agregacja ...")
    start = time.time()
    fold = przygotuj_fold(df, feat, tr, te, cfg)
    print(f"  gotowe w {time.time() - start:.1f} s")
    print(f"\n  X_train: {fold.X_train.shape} | X_test: {fold.X_test.shape}")
    print("  asercje przeszly: brak wspolnych pacjentow, brak NaN, brak wartosci ujemnych")


def tryb_demo(cfg: KonfiguracjaPU, podprobka: int) -> None:
    df, pac, feat = przygotuj(cfg)
    df, pac = zwez_do_podprobki(df, pac, podprobka)
    maska = wszystkie_maski(pac, cfg)
    maska = maska[(maska["udzial_ukrycia"] == cfg.udzial_ukrycia_glowny)
                  & (maska["seed_ukrycia"] == cfg.seedy_ukrycia[0])]
    sprawdz_maske(maska)

    naglowek(f"Przebieg kontrolny na {len(pac):,} pacjentach")
    print("\n  Scoring jest LOSOWY — to test instalacji metryk, nie wynik modelu.")
    print(f"  Oczekiwanie: recall_hidden_at_q blisko q = {cfg.q}, lift blisko 1.0\n")

    rng = np.random.default_rng(0)
    wyniki, cele = [], []
    for fold in foldy_zewnetrzne(df, pac, feat, cfg):
        score = pd.Series(rng.random(len(fold.X_test)), index=fold.X_test.index)
        maska_foldu = maska[maska[KOL_PACJENT].isin(fold.X_test.index)]
        m = komplet_metryk(score, maska_foldu, cfg.q)
        m["fold"] = fold.numer
        wyniki.append(m)
        cele.append(wartosc_celu(score, maska_foldu, cfg))
        print(f"  fold {fold.numer}: pula U {m['pacjentow_w_puli_u']:5,} | "
              f"ukrytych {m['ukrytych_w_puli_u']:3d} | "
              f"recall_hidden@q {m['recall_hidden_at_q']:.3f} | lift {m['lift_at_q']:.2f}")

    w = pd.DataFrame(wyniki)
    KATALOG_WYNIKOW.mkdir(exist_ok=True)
    w.to_csv(KATALOG_WYNIKOW / "demo_metryki.csv", index=False)
    print(f"\n  funkcja celu po foldach: {agreguj_po_foldach(cele):.4f} "
          f"(losowy scoring, wiec powinno wyjsc okolo {cfg.q})")
    print(f"  metryki zapisane w {KATALOG_WYNIKOW / 'demo_metryki.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tryb", choices=["artefakty", "fold", "demo"])
    p.add_argument("--podprobka", type=int, default=0,
                   help="ogranicz do N pacjentow (0 = caly zbior)")
    p.add_argument("--cechy", choices=["38", "35"], default=None)
    p.add_argument("--szybkie-mice", action="store_true",
                   help="lzejszy MICE do weryfikacji dzialania, nie do wynikow")
    a = p.parse_args()

    cfg = DOMYSLNA
    if a.cechy:
        cfg = cfg.zmieniona(zestaw_cech=a.cechy)
    if a.szybkie_mice:
        cfg = cfg.zmieniona(mice_max_iter=2, mice_n_estimators=10, mice_max_depth=5)

    print(f"Konfiguracja: {cfg.zestaw_cech} cech | agregacja {cfg.regula_agregacji} | "
          f"podzial po {cfg.poziom_podzialu}, shuffle={cfg.shuffle}, seed={cfg.seed_podzialu}")

    if a.tryb == "artefakty":
        tryb_artefakty(cfg)
    elif a.tryb == "fold":
        tryb_fold(cfg, a.podprobka)
    else:
        tryb_demo(cfg, a.podprobka)


if __name__ == "__main__":
    main()
