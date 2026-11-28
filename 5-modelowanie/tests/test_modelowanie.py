"""
Testy funkcji oceny (#33), układów foldów i list ukrytych na małych, ręcznie
policzonych przykładach. Nie potrzebują żadnych danych.

Uruchomienie: python modelowanie/tests/test_modelowanie.py   (albo pytest)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dane import uklady  # noqa: E402
from ocena import ocena_w_grupach, recall_hidden  # noqa: E402
from ukrywanie import listy_ukrytych  # noqa: E402

# Pula 10 pacjentów A..J, wynik malejący od A do J; ukryci: B i E.
PULA = pd.Series(np.linspace(1.0, 0.1, 10), index=list("ABCDEFGHIJ"))


def test_przyklad_reczny():
    # q = 20% -> górne 2 miejsca: A, B -> znaleziony 1 z 2 ukrytych
    assert recall_hidden(PULA, {"B", "E"}, 0.20) == 0.5
    # q = 50% -> górne 5: A..E -> obaj ukryci
    assert recall_hidden(PULA, {"B", "E"}, 0.50) == 1.0
    # q = 10% -> tylko A -> nikt
    assert recall_hidden(PULA, {"B", "E"}, 0.10) == 0.0


def test_remis_na_granicy_liczony_jako_wartosc_oczekiwana():
    # A = 0.9; B, C, D remis 0.5 (B ukryty); q = 20% -> 2 miejsca: A + 1 z 3 remisujących
    w = pd.Series([0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=list("ABCDEFGHIJ"))
    assert abs(recall_hidden(w, {"B"}, 0.20) - 1 / 3) < 1e-12


def test_staly_wynik_daje_q():
    # model bez informacji (wszyscy remisują) -> dokładnie k / N
    w = pd.Series(0.3, index=list("ABCDEFGHIJ"))
    assert abs(recall_hidden(w, {"B", "E"}, 0.20) - 0.2) < 1e-12


def test_kolejnosc_wierszy_nie_ma_znaczenia():
    w = PULA.sample(frac=1, random_state=1)
    assert recall_hidden(w, {"B", "E"}, 0.20) == recall_hidden(PULA, {"B", "E"}, 0.20)


def test_brak_ukrytych_to_brak_miary_nie_zero():
    assert np.isnan(recall_hidden(PULA, set(), 0.05))


def test_losowy_ranking_srednio_q():
    rng = np.random.default_rng(0)
    idx = np.arange(2000)
    ukryci = set(range(100))
    r = [recall_hidden(pd.Series(rng.random(2000), index=idx), ukryci, 0.05) for _ in range(200)]
    assert abs(np.mean(r) - 0.05) < 0.01


def test_ocena_w_grupach_usuwa_skrot_grupy():
    # wynik = numer grupy (model zna tylko grupę). Grupa 1 to 40 pacjentów, w tym
    # wszyscy 20 ukrytych. Globalnie: 20 miejsc (10%) przypada na 40 remisujących
    # z grupy 1 -> oczekiwanie 10 ukrytych -> 0.5, czyli 5 razy więcej niż losowo.
    # W grupach model nic nie wie -> dokładnie q.
    idx = np.arange(200)
    grupa = pd.Series(np.where(idx < 40, 1, 0), index=idx)
    w = grupa.astype(float)
    ukryci = set(range(20))
    assert abs(recall_hidden(w, ukryci, 0.10) - 0.5) < 1e-12
    assert abs(ocena_w_grupach(w, ukryci, grupa, q=0.10) - 0.10) < 1e-12


def _pacjenci():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"patient_id": np.arange(500), "fold": np.repeat(np.arange(5), 100),
                         "label": (rng.random(500) < 0.2).astype(int)})


def test_uklady_robocze_nie_dotykaja_foldu_4():
    pac = _pacjenci()
    f = pac.set_index("patient_id")["fold"]
    ukl = uklady(pac)
    assert len(ukl) == 4
    for _, tr, te in ukl:
        assert (f.loc[tr] != 4).all() and (f.loc[te] != 4).all()
        assert not set(tr) & set(te)
    (_, tr, te), = uklady(pac, koncowy=True)
    assert set(f.loc[te]) == {4} and set(f.loc[tr]) == {0, 1, 2, 3}


def test_listy_ukrytych_rowny_udzial_w_foldach_i_powtarzalne():
    pac = _pacjenci()
    listy = listy_ukrytych(pac)
    chorzy = pac[pac.label == 1].set_index("patient_id")["fold"]
    for z, ukryci in listy.items():
        assert ukryci <= set(chorzy.index), "ukryty może być tylko chory"
        udzial = chorzy.index.isin(list(ukryci)).astype(float)
        assert (pd.Series(udzial, index=chorzy.index).groupby(chorzy).mean().sub(0.4).abs() < 0.05).all()
    assert listy_ukrytych(pac)[0] == listy[0] and listy[0] != listy[1]


if __name__ == "__main__":
    testy = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in testy:
        f()
        print(f"  OK   {n}")
    print(f"\n{len(testy)}/{len(testy)} testow przeszlo")
