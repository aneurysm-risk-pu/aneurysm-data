"""
Punkt 1: foldy i preprocessing wewnatrz foldu.

Kolejnosc jest nienaruszalna:

    podzial  ->  kontrola kohort  ->  scaler  ->  MICE  ->  agregacja

Wszystko po slowie "podzial" dopasowuje sie WYLACZNIE na czesci treningowej.
Czesc walidacyjna jest tylko transformowana. Dotyczy to rowniez skalera i
imputera — one tez sie ucza, wiec tez musza respektowac granice foldow.

Modul udostepnia podzial zagniezdzony: foldy zewnetrzne sluza do raportowania
wyniku, foldy wewnetrzne do strojenia. Tych samych foldow zewnetrznych nie
wolno uzyc do obu rzeczy naraz, bo to zawyza jakosc.

Gniazdo `hak_kontroli_kohort` jest puste, dopoki nie zapadnie decyzja 0.2.
Wariant C (dopasowanie albo wazenie kohort po czasie i zrodle) wchodzi
wlasnie tam, bez przebudowy pipeline'u.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from .config import KonfiguracjaPU
from .dane import KOL_ETYKIETA, KOL_PACJENT, agreguj_do_pacjenta

# Hak kontroli kohort: (df_train, df_test, cfg) -> (df_train, df_test, wagi_train)
HakKontroli = Callable[[pd.DataFrame, pd.DataFrame, KonfiguracjaPU],
                       tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]]


@dataclass
class Fold:
    """Jeden gotowy fold: dane treningowe i walidacyjne po preprocessingu."""
    numer: int
    X_train: pd.DataFrame          # indeks = patient_id (albo custom_id przy rekordach)
    X_test: pd.DataFrame
    pacjenci_train: np.ndarray
    pacjenci_test: np.ndarray
    wagi_train: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Podzial
# ---------------------------------------------------------------------------

def _splitter(cfg: KonfiguracjaPU, n_splits: int):
    kw = dict(n_splits=n_splits, shuffle=cfg.shuffle)
    if cfg.shuffle:
        kw["random_state"] = cfg.seed_podzialu
    return kw


def podzial_pacjentow(pac: pd.DataFrame, cfg: KonfiguracjaPU, n_splits: int,
                      df: Optional[pd.DataFrame] = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Zwraca pary (pacjenci_train, pacjenci_test).

    Poziom 'pacjent' to zwykly StratifiedKFold na jednym wierszu na osobe.
    Poziom 'rekord' to StratifiedGroupKFold z grupowaniem po patient_id —
    wariant zgodny z dotychczasowym pipeline'em, zachowany do porownania.

    Stratyfikujemy po `true_label`, bo foldy powstaja PRZED maskami ukrywania.
    Model nigdy nie widzi tej kolumny; sluzy wylacznie do konstrukcji benchmarku.
    """
    if cfg.poziom_podzialu == "pacjent":
        skf = StratifiedKFold(**_splitter(cfg, n_splits))
        idx = pac[KOL_PACJENT].to_numpy()
        for tr, te in skf.split(idx, pac["true_label"]):
            yield idx[tr], idx[te]
    else:
        if df is None:
            raise ValueError("poziom 'rekord' wymaga przekazania danych rekordowych")
        d = df[df[KOL_PACJENT].isin(set(pac[KOL_PACJENT]))]
        sgkf = StratifiedGroupKFold(**_splitter(cfg, n_splits))
        for tr, te in sgkf.split(d, d[KOL_ETYKIETA], groups=d[KOL_PACJENT]):
            yield d.iloc[tr][KOL_PACJENT].unique(), d.iloc[te][KOL_PACJENT].unique()


def tabela_pacjent_fold(pac: pd.DataFrame, cfg: KonfiguracjaPU,
                        df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Artefakt do raportu: ktory pacjent trafil do ktorego foldu zewnetrznego."""
    wiersze = []
    for i, (_, test) in enumerate(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df)):
        wiersze.append(pd.DataFrame({KOL_PACJENT: test, "fold": i}))
    t = pd.concat(wiersze, ignore_index=True)
    assert t[KOL_PACJENT].is_unique, "pacjent trafil do wiecej niz jednego foldu testowego"
    return t.merge(pac[[KOL_PACJENT, "true_label"]], on=KOL_PACJENT).sort_values(KOL_PACJENT)


# ---------------------------------------------------------------------------
# Preprocessing wewnatrz foldu
# ---------------------------------------------------------------------------

def zbuduj_imputer(cfg: KonfiguracjaPU) -> IterativeImputer:
    """MICE z estymatorem ExtraTrees — parametry z benchmarku imputacji.

    Technicznie jest to pojedyncza, deterministyczna imputacja iteracyjna
    (`sample_posterior=False`), a nie klasyczne multiple imputation.
    """
    return IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=cfg.mice_n_estimators,
            max_depth=cfg.mice_max_depth,
            random_state=cfg.seed_imputacji,
            n_jobs=-1),
        max_iter=cfg.mice_max_iter,
        random_state=cfg.seed_imputacji,
        min_value=0.0,
        max_value=1.0,
    )


def _dopasuj_scaler(train: pd.DataFrame, feat: list[str], cfg: KonfiguracjaPU) -> MinMaxScaler:
    sc = MinMaxScaler()
    if cfg.scaler_na_kompletnych:
        kompletne = train[feat].dropna()
        if len(kompletne) == 0:
            # Brak kompletnych wierszy zdarza sie przy waskich kohortach
            # (np. wariant wspolnego okna dat) — wtedy skalujemy kolumnowo.
            print("    [uwaga] brak kompletnych wierszy w treningu, "
                  "scaler dopasowany kolumnowo na dostepnych wartosciach")
            sc.fit(train[feat])
            return sc
        if len(kompletne) < 50:
            print(f"    [uwaga] tylko {len(kompletne)} kompletnych wierszy do dopasowania "
                  "skalera — skala opiera sie na bardzo selektywnej podprobie")
        sc.fit(kompletne)
    else:
        sc.fit(train[feat])
    return sc


def przygotuj_fold(df: pd.DataFrame, feat: list[str], pacjenci_train: np.ndarray,
                   pacjenci_test: np.ndarray, cfg: KonfiguracjaPU, numer: int = 0,
                   hak_kontroli_kohort: Optional[HakKontroli] = None) -> Fold:
    """Pelny preprocessing jednego foldu, z zachowaniem granicy train/test."""
    train = df[df[KOL_PACJENT].isin(set(pacjenci_train))].copy()
    test = df[df[KOL_PACJENT].isin(set(pacjenci_test))].copy()

    wagi = None
    if hak_kontroli_kohort is not None:
        train, test, wagi = hak_kontroli_kohort(train, test, cfg)

    sc = _dopasuj_scaler(train, feat, cfg)
    imp = zbuduj_imputer(cfg)
    imp.fit(sc.transform(train[feat]))

    def przetworz(d: pd.DataFrame) -> pd.DataFrame:
        wart = sc.inverse_transform(imp.transform(sc.transform(d[feat])))
        out = d[[KOL_PACJENT]].copy()
        out[feat] = wart
        return out

    train_p, test_p = przetworz(train), przetworz(test)

    if cfg.jednostka_treningu == "pacjent":
        X_train = agreguj_do_pacjenta(train_p, feat, cfg)
        X_test = agreguj_do_pacjenta(test_p, feat, cfg)
    else:
        X_train = train_p.set_index(KOL_PACJENT)[feat]
        X_test = test_p.set_index(KOL_PACJENT)[feat]
        if wagi is None:
            wagi = 1.0 / X_train.groupby(level=0)[feat[0]].transform("size")

    fold = Fold(numer, X_train, X_test, np.asarray(pacjenci_train),
                np.asarray(pacjenci_test), wagi)
    sprawdz_fold(fold, train[feat])
    return fold


def sprawdz_fold(fold: Fold, train_surowy: pd.DataFrame) -> None:
    """Asercje poprawnosci foldu.

    Zakresy sprawdzamy ostrzezeniem, nie bledem: wartosc walidacyjna MOZE
    wyjsc poza min-max treningu i nie jest to powod do odrzucenia foldu.
    """
    wspolni = set(fold.pacjenci_train) & set(fold.pacjenci_test)
    assert not wspolni, f"{len(wspolni)} pacjentow w train i test jednoczesnie"

    for nazwa, X in [("train", fold.X_train), ("test", fold.X_test)]:
        assert not X.isna().any().any(), f"NaN po imputacji w {nazwa}"
        assert np.isfinite(X.to_numpy()).all(), f"wartosc nieskonczona w {nazwa}"
        ujemne = int((X.to_numpy() < 0).sum())
        assert ujemne == 0, f"{ujemne} wartosci ujemnych w {nazwa}"

    lo, hi = train_surowy.min(), train_surowy.max()
    poza = ((fold.X_test < lo) | (fold.X_test > hi)).to_numpy().sum()
    if poza:
        udzial = poza / fold.X_test.size
        print(f"    [uwaga] fold {fold.numer}: {poza} wartosci walidacyjnych "
              f"({udzial:.2%}) poza zakresem treningu — to dopuszczalne")


# ---------------------------------------------------------------------------
# Walidacja zagniezdzona
# ---------------------------------------------------------------------------

def foldy_zewnetrzne(df: pd.DataFrame, pac: pd.DataFrame, feat: list[str],
                     cfg: KonfiguracjaPU,
                     hak_kontroli_kohort: Optional[HakKontroli] = None) -> Iterator[Fold]:
    """Foldy do raportowania wyniku. Nigdy nie sluza do strojenia."""
    for i, (tr, te) in enumerate(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df)):
        yield przygotuj_fold(df, feat, tr, te, cfg, i, hak_kontroli_kohort)


def foldy_wewnetrzne(df: pd.DataFrame, pac: pd.DataFrame, feat: list[str],
                     pacjenci_outer_train: np.ndarray, cfg: KonfiguracjaPU,
                     hak_kontroli_kohort: Optional[HakKontroli] = None) -> Iterator[Fold]:
    """Foldy do strojenia, budowane WEWNATRZ treningu foldu zewnetrznego.

    Preprocessing dopasowuje sie od nowa w kazdym foldzie wewnetrznym.
    Zaimputowanie calego outer-train raz i dopiero potem dzielenie go na foldy
    wewnetrzne byloby wyciekiem — statystyki z czesci walidacyjnej inner
    weszlyby do uzupelniania brakow treningowych.
    """
    pac_in = pac[pac[KOL_PACJENT].isin(set(pacjenci_outer_train))]
    df_in = df[df[KOL_PACJENT].isin(set(pacjenci_outer_train))]
    for i, (tr, te) in enumerate(podzial_pacjentow(pac_in, cfg, cfg.n_splits_inner, df_in)):
        yield przygotuj_fold(df_in, feat, tr, te, cfg, i, hak_kontroli_kohort)


# ---------------------------------------------------------------------------
# Cache — preprocessing nie zalezy od hiperparametrow klasyfikatora
# ---------------------------------------------------------------------------

def odcisk_konfiguracji(cfg: KonfiguracjaPU, dodatkowe: str = "") -> str:
    """Skrot konfiguracji, zeby cache nie przezyl zmiany decyzji protokolu."""
    tresc = json.dumps(cfg.do_slownika(), sort_keys=True, default=str) + dodatkowe
    return hashlib.sha256(tresc.encode()).hexdigest()[:12]


def zapisz_fold(fold: Fold, katalog: Path, cfg: KonfiguracjaPU) -> None:
    katalog.mkdir(parents=True, exist_ok=True)
    znak = odcisk_konfiguracji(cfg)
    fold.X_train.to_parquet(katalog / f"fold{fold.numer}_{znak}_train.parquet")
    fold.X_test.to_parquet(katalog / f"fold{fold.numer}_{znak}_test.parquet")
