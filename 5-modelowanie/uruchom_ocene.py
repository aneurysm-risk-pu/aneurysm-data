"""
Modele kontrolne (#31) i model odniesienia, ocenione wspólną funkcją (#33).

Schemat: 4 układy robocze (foldy 0–3, fold 4 nietknięty) x 5 list ukrytych = 20 przebiegów.
W każdym przebiegu:
  trening  = pacjenci z 3 foldów; etykieta widziana przez model: 1 = widoczny chory, 0 = reszta (KOR i ukryci),
  ocena    = pula z czwartego foldu: KOR + ukryci chorzy.

Uruchomienie:
    python modelowanie/uruchom_ocene.py                                   # dane syntetyczne z data/
    python modelowanie/uruchom_ocene.py --dane data_brak/processed/aneurysm_sgkf_input.csv --nazwa brak
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dane import BASE, KOL_PACJENT, WEJSCIE, uklady, wczytaj
from kontrolne import MODEL_ODNIESIENIA, model, zestawy_cech
from ocena import ocena, ocena_w_grupach, recall_hidden
from ukrywanie import listy_ukrytych


def przebiegi(df, pac, cechy) -> pd.DataFrame:
    zestawy = zestawy_cech(df, pac, cechy)
    p = pac.set_index(KOL_PACJENT)
    wiersze = []
    for z, ukryci in listy_ukrytych(pac).items():
        for i, (nazwa_ukl, tr, te) in enumerate(uklady(pac)):
            y_tr = ((p.loc[tr, "label"] == 1) & ~tr.isin(ukryci)).astype(int)
            pula = te[(p.loc[te, "label"] == 0) | te.isin(ukryci)]
            rng = np.random.default_rng(1000 + 10 * z + i)
            wyniki = {"losowy": pd.Series(rng.random(len(pula)), index=pula)}
            for nazwa, X in zestawy.items():
                m = model().fit(X.loc[tr], y_tr)
                wyniki[nazwa] = pd.Series(m.predict_proba(X.loc[pula])[:, 1], index=pula)
            for nazwa, w in wyniki.items():
                r = ocena(w, ukryci)
                r["R@5% w grupach liczby badań"] = ocena_w_grupach(w, ukryci, p["klasa_n_rek"])
                r["R@5% tylko okno wspólne"] = recall_hidden(w[p.loc[w.index, "w_oknie"].to_numpy()], ukryci, 0.05)
                wiersze.append({"model": nazwa, "ziarno": z, "uklad": nazwa_ukl, **r})
        print(f"  ziarno {z}: gotowe", flush=True)
    return pd.DataFrame(wiersze)


def podsumowanie(t: pd.DataFrame) -> pd.DataFrame:
    kol = ["R@1%", "R@5%", "R@10%", "AUC", "R@5% w grupach liczby badań", "R@5% tylko okno wspólne"]
    s = t.groupby("model", sort=False)[kol].mean()
    s.insert(2, "R@5% odch.", t.groupby("model", sort=False)["R@5%"].std())
    return s.round(3)


def przewaga(t: pd.DataFrame) -> pd.DataFrame:
    """R@5% modelu odniesienia minus R@5% każdej kontroli, w tych samych przebiegach."""
    w = t.pivot_table(index=["ziarno", "uklad"], columns="model", values="R@5%")
    d = w.drop(columns=MODEL_ODNIESIENIA).rsub(w[MODEL_ODNIESIENIA], axis=0)
    return pd.DataFrame({"średnio": d.mean(), "odch.": d.std(), "najmniej": d.min()}).round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dane", type=Path, default=WEJSCIE)
    ap.add_argument("--nazwa", default="realny")
    a = ap.parse_args()
    df, pac, cechy = wczytaj(BASE / a.dane if not a.dane.is_absolute() else a.dane)
    print(f"Dane: {a.nazwa}, pacjenci {len(pac)}, chorzy {int(pac.label.sum())}")
    t = przebiegi(df, pac, cechy)
    wyj = BASE / "results"
    wyj.mkdir(exist_ok=True)
    t.to_csv(wyj / f"ocena_{a.nazwa}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nŚrednie z {t.groupby('model').size().iloc[0]} przebiegów (q = 5%, losowy ranking daje ok. 0.05):")
    print(podsumowanie(t).to_string())
    print("\nPrzewaga modelu odniesienia nad każdą kontrolą (R@5%, te same przebiegi):")
    print(przewaga(t).to_string())
    print(f"\nZapisano -> results/ocena_{a.nazwa}.csv")


if __name__ == "__main__":
    main()
