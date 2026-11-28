"""
Zadanie #33: wspólna funkcja oceny.

Pula oceny w układzie = pacjenci z części oceny, których model widzi jako
nieoznaczonych: wszyscy KOR + ukryci chorzy. Widoczni chorzy z części oceny
nie wchodzą do puli.

RecallHidden@q: jaka część ukrytych chorych trafia do górnych q% puli.
Losowy ranking daje średnio q (np. 5% przy q = 5%).

Remisy: jeśli na granicy górnych q% kilku pacjentów ma ten sam wynik, liczymy
wartość oczekiwaną przy losowym rozstrzygnięciu remisu. Rozstrzyganie po
numerze pacjenta (jak w kontrola_skrotow.py) mogłoby faworyzować jedną
kohortę, a modele kontrolne, np. „liczba badań”, mają bardzo dużo remisów.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Q = (0.01, 0.05, 0.10)


def recall_hidden(wynik: pd.Series, ukryci: set, q: float) -> float:
    """wynik: patient_id -> wynik modelu, dla wszystkich pacjentów puli."""
    s = wynik.to_numpy(dtype=float)
    assert not np.isnan(s).any(), "wynik modelu zawiera NaN"
    h = wynik.index.isin(list(ukryci))
    if h.sum() == 0:
        return float("nan")          # brak ukrytych w puli: miary nie ma (to nie jest zero)
    k = int(np.ceil(q * len(s)))
    prog = np.sort(s)[::-1][k - 1]   # wynik pacjenta na ostatnim miejscu w górnych q%
    powyzej, rowne = s > prog, s == prog
    wolne = k - powyzej.sum()        # miejsca w górnych q% dla pacjentów z wynikiem równym progowi
    trafieni = (h & powyzej).sum() + (h & rowne).sum() * wolne / rowne.sum()
    return float(trafieni / h.sum())


def ocena(wynik: pd.Series, ukryci: set, qs=Q) -> dict:
    """Komplet miar dla jednej puli: RecallHidden@q dla każdego q i AUC (ukryci vs reszta puli)."""
    h = wynik.index.isin(list(ukryci))
    out = {f"R@{int(round(q * 100))}%": recall_hidden(wynik, ukryci, q) for q in qs}
    out["AUC"] = float(roc_auc_score(h, wynik)) if 0 < h.sum() < len(h) else float("nan")
    out["n_puli"], out["n_ukrytych"] = int(len(h)), int(h.sum())
    return out


def ocena_w_grupach(wynik: pd.Series, ukryci: set, grupa: pd.Series, q: float = 0.05,
                    min_ukrytych: int = 5) -> float:
    """RecallHidden@q liczony osobno w każdej grupie (np. ta sama liczba badań)
    i uśredniony z wagą liczby ukrytych.

    Ukrytych porównujemy tylko z pacjentami KOR z tej samej grupy, więc cecha,
    według której grupujemy, przestaje pomagać. Losowy ranking nadal daje około q.
    Grupy z mniej niż `min_ukrytych` ukrytymi pomijamy.
    """
    g = grupa.reindex(wynik.index)
    wyniki, wagi = [], []
    for _, idx in wynik.groupby(g).groups.items():
        w = wynik.loc[idx]
        n_h = w.index.isin(list(ukryci)).sum()
        if n_h >= min_ukrytych:
            wyniki.append(recall_hidden(w, ukryci, q))
            wagi.append(n_h)
    return float(np.average(wyniki, weights=wagi)) if wagi else float("nan")
