"""
Testy pipeline'u PU (punkty 1, 2, 4, 5).

Metryki i maski testujemy na danych syntetycznych, gdzie wynik znamy z gory —
na prawdziwych danych nie da sie odroznic bledu w metryce od slabego modelu.

Uruchomienie:
    python -m pytest 4-pu-setup/tests/test_pu.py -q
    albo
    python 4-pu-setup/tests/test_pu.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pu.cel import agreguj_po_foldach, wartosc_celu                      # noqa: E402
from pu.config import KonfiguracjaPU                                     # noqa: E402
from pu.dane import KOL_PACJENT                                          # noqa: E402
from pu.foldy import (foldy_wewnetrzne, podzial_pacjentow,               # noqa: E402
                      przygotuj_fold, tabela_pacjent_fold)
from pu.metryki import (komplet_metryk, lift_at_q, recall_hidden_at_q,   # noqa: E402
                        recall_known, udzial_ukrytych_w_top, przygotuj_tabele)
from pu.ukrywanie import maska_ukrycia, sprawdz_maske, wszystkie_maski   # noqa: E402


# ---------------------------------------------------------------------------
# Dane syntetyczne
# ---------------------------------------------------------------------------

def maska_testowa(n_u=100, n_ukrytych=10, n_znanych=20) -> pd.DataFrame:
    """Pula: n_u nieoznaczonych (w tym n_ukrytych prawdziwie pozytywnych) + znani pozytywni."""
    pid = np.arange(1, n_u + n_znanych + 1)
    true_label = np.r_[np.zeros(n_u - n_ukrytych, int), np.ones(n_ukrytych, int),
                       np.ones(n_znanych, int)]
    is_hidden = np.r_[np.zeros(n_u - n_ukrytych, bool), np.ones(n_ukrytych, bool),
                      np.zeros(n_znanych, bool)]
    return pd.DataFrame({KOL_PACJENT: pid, "true_label": true_label,
                         "is_hidden": is_hidden,
                         "observed_label": np.where(is_hidden, 0, true_label)})


def dane_testowe(n_pac=60, n_cech=4, seed=0) -> tuple:
    """Rekordy z brakami; czesc pacjentow ma po kilka pomiarow."""
    rng = np.random.default_rng(seed)
    wiersze = []
    for p in range(1, n_pac + 1):
        for _ in range(rng.integers(1, 4)):
            wiersze.append({KOL_PACJENT: p, "label": int(p > n_pac * 0.8),
                            **{f"c{i}": rng.normal(10, 2) for i in range(n_cech)}})
    df = pd.DataFrame(wiersze)
    feat = [f"c{i}" for i in range(n_cech)]
    braki = rng.random(df[feat].shape) < 0.15
    df[feat] = df[feat].mask(braki)
    pac = pd.DataFrame({KOL_PACJENT: sorted(df[KOL_PACJENT].unique())})
    pac["true_label"] = pac[KOL_PACJENT].map(df.groupby(KOL_PACJENT)["label"].max())
    return df, pac, feat


# ---------------------------------------------------------------------------
# Punkt 4 — metryki
# ---------------------------------------------------------------------------

def test_model_idealny_odzyskuje_wszystkich_ukrytych():
    m = maska_testowa(n_u=100, n_ukrytych=10)
    # ukryci dostaja najwyzszy score
    score = pd.Series(np.where(m["is_hidden"], 1.0, 0.0), index=m[KOL_PACJENT])
    t = przygotuj_tabele(score, m)
    assert recall_hidden_at_q(t, q=0.10) == 1.0
    assert udzial_ukrytych_w_top(t, q=0.10) == 1.0
    assert lift_at_q(t, q=0.10) == 10.0        # baza to 10/100


def test_model_odwrotny_nie_odzyskuje_nikogo():
    m = maska_testowa(n_u=100, n_ukrytych=10)
    score = pd.Series(np.where(m["is_hidden"], 0.0, 1.0), index=m[KOL_PACJENT])
    assert recall_hidden_at_q(przygotuj_tabele(score, m), q=0.10) == 0.0


def test_metryka_nie_zalezy_od_kolejnosci_wierszy():
    """Remisy musza byc rozstrzygane deterministycznie, inaczej wynik jest losowy."""
    m = maska_testowa(n_u=50, n_ukrytych=5)
    score = pd.Series(np.ones(len(m)), index=m[KOL_PACJENT])     # wszystkie remisy
    a = recall_hidden_at_q(przygotuj_tabele(score, m), q=0.20)
    przetasowana = m.sample(frac=1.0, random_state=7)
    b = recall_hidden_at_q(przygotuj_tabele(score.loc[przetasowana[KOL_PACJENT]],
                                            przetasowana), q=0.20)
    assert a == b


def test_ukryci_nie_wchodza_do_puli_znanych():
    """Ukryty pozytywny jest dla modelu nieoznaczony i tak musi byc liczony."""
    m = maska_testowa(n_u=100, n_ukrytych=10, n_znanych=20)
    score = pd.Series(np.linspace(0, 1, len(m)), index=m[KOL_PACJENT])
    w = komplet_metryk(score, m, q=0.10)
    assert w["pacjentow_w_puli_u"] == 100
    assert w["ukrytych_w_puli_u"] == 10
    assert 0.0 <= w["recall_known"] <= 1.0


def test_brak_ukrytych_daje_nan_zamiast_zera():
    """Fold bez ukrytych pozytywnych to brak informacji, a nie wynik zerowy."""
    m = maska_testowa(n_u=50, n_ukrytych=0)
    score = pd.Series(np.random.default_rng(0).random(len(m)), index=m[KOL_PACJENT])
    assert np.isnan(recall_hidden_at_q(przygotuj_tabele(score, m), q=0.10))


# ---------------------------------------------------------------------------
# Punkt 5 — funkcja celu
# ---------------------------------------------------------------------------

def test_funkcja_celu_z_alfa_jeden_to_czysty_recall_hidden():
    m = maska_testowa()
    score = pd.Series(np.where(m["is_hidden"], 1.0, 0.0), index=m[KOL_PACJENT])
    cfg = KonfiguracjaPU(q=0.10, alfa=1.0)
    assert wartosc_celu(score, m, cfg) == recall_hidden_at_q(przygotuj_tabele(score, m), 0.10)


def test_funkcja_celu_uwzglednia_znanych_gdy_alfa_mniejsza():
    m = maska_testowa()
    score = pd.Series(np.where(m["is_hidden"], 1.0, 0.0), index=m[KOL_PACJENT])
    pelna = wartosc_celu(score, m, KonfiguracjaPU(q=0.10, alfa=1.0))
    mieszana = wartosc_celu(score, m, KonfiguracjaPU(q=0.10, alfa=0.5))
    assert mieszana < pelna          # znani pozytywni maja tu score 0, wiec ciagna wynik w dol


def test_agregacja_po_foldach_pomija_puste():
    assert agreguj_po_foldach([1.0, float("nan"), 0.0]) == 0.5


# ---------------------------------------------------------------------------
# Punkt 2 — ukrywanie
# ---------------------------------------------------------------------------

def test_maska_ukrywa_wlasciwy_odsetek_pozytywnych():
    _, pac, _ = dane_testowe()
    m = maska_ukrycia(pac, udzial=0.40, seed=1)
    sprawdz_maske(m)
    poz = int((pac["true_label"] == 1).sum())
    assert int(m["is_hidden"].sum()) == round(0.40 * poz)


def test_maska_jest_odtwarzalna_i_zalezy_od_ziarna():
    _, pac, _ = dane_testowe()
    a = maska_ukrycia(pac, 0.40, seed=1)
    b = maska_ukrycia(pac, 0.40, seed=1)
    c = maska_ukrycia(pac, 0.40, seed=2)
    assert a["is_hidden"].equals(b["is_hidden"])
    assert not a["is_hidden"].equals(c["is_hidden"])


def test_nigdy_nie_ukrywamy_negatywnego():
    _, pac, _ = dane_testowe()
    for seed in range(5):
        m = maska_ukrycia(pac, 0.60, seed)
        assert (m.loc[m["is_hidden"], "true_label"] == 1).all()


def test_komplet_masek_ma_wszystkie_scenariusze():
    _, pac, _ = dane_testowe()
    cfg = KonfiguracjaPU()
    m = wszystkie_maski(pac, cfg)
    assert set(m["udzial_ukrycia"]) == {cfg.udzial_ukrycia_glowny, *cfg.udzialy_ukrycia_wrazliwosc}
    assert set(m["seed_ukrycia"]) == set(cfg.seedy_ukrycia)
    assert (m[m["scenariusz"] == "glowny"]["udzial_ukrycia"] == cfg.udzial_ukrycia_glowny).all()


# ---------------------------------------------------------------------------
# Punkt 1 — foldy
# ---------------------------------------------------------------------------

def _cfg_szybki(**kw) -> KonfiguracjaPU:
    """Lekka konfiguracja: MICE z jedna iteracja i malym lasem, zeby testy trwaly sekundy."""
    baza = dict(mice_max_iter=1, mice_n_estimators=3, mice_max_depth=3,
                n_splits_outer=3, n_splits_inner=2)
    baza.update(kw)
    return KonfiguracjaPU(**baza)


def test_zaden_pacjent_nie_jest_w_dwoch_foldach():
    df, pac, _ = dane_testowe()
    t = tabela_pacjent_fold(pac, _cfg_szybki(), df)
    assert t[KOL_PACJENT].is_unique
    assert len(t) == len(pac)


def test_fold_nie_ma_brakow_po_imputacji():
    df, pac, feat = dane_testowe()
    cfg = _cfg_szybki()
    tr, te = next(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df))
    fold = przygotuj_fold(df, feat, tr, te, cfg)
    assert not fold.X_train.isna().any().any()
    assert not fold.X_test.isna().any().any()
    assert not set(fold.pacjenci_train) & set(fold.pacjenci_test)


def test_agregacja_daje_jeden_wiersz_na_pacjenta():
    df, pac, feat = dane_testowe()
    cfg = _cfg_szybki(jednostka_treningu="pacjent")
    tr, te = next(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df))
    fold = przygotuj_fold(df, feat, tr, te, cfg)
    assert len(fold.X_train) == len(set(tr))
    assert len(fold.X_test) == len(set(te))


def test_foldy_wewnetrzne_nie_siegaja_poza_trening_zewnetrzny():
    """Najwazniejszy test wycieku: inner CV nie moze zobaczyc outer-test."""
    df, pac, feat = dane_testowe()
    cfg = _cfg_szybki()
    outer_tr, outer_te = next(podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df))
    for fold in foldy_wewnetrzne(df, pac, feat, outer_tr, cfg):
        uzyci = set(fold.pacjenci_train) | set(fold.pacjenci_test)
        assert uzyci <= set(outer_tr), "fold wewnetrzny siegnal po pacjenta z outer-test"
        assert not uzyci & set(outer_te)


def test_poziom_rekordowy_tez_nie_dzieli_pacjenta():
    df, pac, feat = dane_testowe()
    cfg = _cfg_szybki(poziom_podzialu="rekord")
    for tr, te in podzial_pacjentow(pac, cfg, cfg.n_splits_outer, df):
        assert not set(tr) & set(te)


if __name__ == "__main__":
    lokalne = dict(globals())
    testy = [(n, f) for n, f in lokalne.items() if n.startswith("test_")]
    bledy = 0
    for nazwa, fn in testy:
        try:
            fn()
            print(f"  OK   {nazwa}")
        except AssertionError as e:
            bledy += 1
            print(f"  BLAD {nazwa}: {e}")
    print(f"\n{len(testy) - bledy}/{len(testy)} testow przeszlo")
    sys.exit(1 if bledy else 0)
