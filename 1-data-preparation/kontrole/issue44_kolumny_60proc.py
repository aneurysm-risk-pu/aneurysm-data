"""
#44 Skąd mamy 38 cech: reguła 60% braków w preprocessingu Liwii (final_preprocessing.py).

Krok 9 (drop_high_missingness) działa OSOBNO w każdej grupie: usuwa kolumny z ponad 60% braków,
a potem rekordy z ponad 60% braków wśród pozostałych cech. Krok 10 zostawia tylko kolumny wspólne,
więc kolumna usunięta w jednej grupie znika z obu („ponad 60% w którejkolwiek grupie”).
Skrypt liczy braki na wejściu kroku 9 i porównuje regułę z kolumnami w plikach końcowych.

Uruchomienie: python 1-data-preparation/kontrole/issue44_kolumny_60proc.py
"""

import pandas as pd

from wspolne import KOHORTY, cechy_lab, przebieg, zapisz

PROG = 60.0

kroki = dict(przebieg())
przed = kroki["8 walidacja wartości"]
koniec = kroki["10 tylko kolumny wspólne"]

lab = cechy_lab(przed["kor"])
braki = pd.DataFrame({k: (przed[k][lab].isna().mean() * 100).round(1) for k in KOHORTY})
osobno = {k: sorted(braki.index[braki[k] > PROG]) for k in KOHORTY}
ktorakolwiek = sorted(braki.index[(braki > PROG).any(axis=1)])
usuniete = sorted(c for c in lab if c not in koniec["kor"].columns)
rekordy = {k: len(przed[k]) - len(koniec[k]) for k in KOHORTY}

pokaz = sorted(set(usuniete) | set(braki.index[((braki > 50) & (braki < 70)).any(axis=1)]))
print("Braki [%] na wejściu kroku 9: kolumny usunięte i bliskie progu (50–70%):")
print(braki.loc[pokaz].to_string())
print(f"\nKrok 9 osobno w grupach: KOR usuwa {osobno['kor']}, NEURO usuwa {osobno['neuro']}")
print(f"Po kroku 10 (kolumny wspólne) = ponad 60% w którejkolwiek grupie: {ktorakolwiek}")
print(f"Faktycznie brak w plikach końcowych: {usuniete} -> zgodne: {ktorakolwiek == usuniete}")
print(f"Zostaje {len(lab) - len(usuniete)} kolumn badań + wiek i płeć = {len(lab) - len(usuniete) + 2} cech")
print(f"Ta sama reguła usuwa też rekordy z ponad 60% braków: KOR {rekordy['kor']} "
      f"({rekordy['kor'] / len(przed['kor']):.1%}), NEURO {rekordy['neuro']} ({rekordy['neuro'] / len(przed['neuro']):.1%})")

zapisz("issue44_kolumny_60proc", {
    "skrypt_sprawdzany": "1-data-preparation/scripts-lf/final_preprocessing.py (kroki 9–10)",
    "prog_proc": PROG,
    "braki_proc_na_wejsciu_kroku_9": braki.loc[pokaz].to_dict(orient="index"),
    "krok9_osobno_usuwa": osobno,
    "ponad_60_w_ktorejkolwiek_grupie": ktorakolwiek,
    "brak_w_plikach_koncowych": usuniete,
    "zgodne": ktorakolwiek == usuniete,
    "liczba_cech": len(lab) - len(usuniete) + 2,
    "usuniete_rekordy_ponad_60proc": {k: {"rekordy": rekordy[k], "udzial": round(rekordy[k] / len(przed[k]), 4)}
                                      for k in KOHORTY},
})
