"""
#46 Czyszczenie z etapu 1 a różnice między grupami.

Rozkłada spadek liczby rekordów z kroku 1 i 2 na powody, osobno dla KOR i NEURO,
na samych plikach (notebooka kroku 2 nie da się uruchomić od nowa: Colab, ręczne progi):
  krok 1  dane surowe -> *_cleaning.csv   powtórzony custom_id, komórki '<x' zamienione na brak
  krok 2  *_cleaning.csv -> *_shortend.csv, w kolejności notebooka:
          a) wiek poza 18–100 lub brak płci                      (liczone)
          b) tygodnie obecne w obu plikach (ten sam custom_id)   (liczone)
          c) wartości odstające, kwantyle liczone osobno w grupach (liczba z wydruku notebooka,
             3-sgkf-split/przygotowanie/results/analiza_decyzji.json, sekcja F)
          d) limit historii 11 KOR / 38 NEURO                    (reszta: wydruk notebooka po c minus *_shortend.csv)
Sprawdza też, czy działał filtr „pacjenci z ponad 60% braków” (dokumenty podają 1 793 pacjentów KOR).

Uruchomienie: python 1-data-preparation/kontrole/issue46_czyszczenie_a_grupy.py
"""

import json

import pandas as pd

from wspolne import (BASE, ETYKIETA, KOHORTY, LIMIT_HISTORII, cechy_lab, po_kroku1, po_kroku2,
                     po_kroku4, surowe, zapisz)

ODSTAJACE = json.loads((BASE / "3-sgkf-split/przygotowanie/results/analiza_decyzji.json")
                       .read_text(encoding="utf-8"))["F_czyszczenie_kwantylowe_etap1"]["rekordy_przed_po"]

s0 = {k: surowe(k) for k in KOHORTY}
s1 = {k: po_kroku1(k) for k in KOHORTY}
s2 = {k: po_kroku2(k) for k in KOHORTY}
lab = cechy_lab(s0["kor"])
cechy38 = [c for c in s2["kor"].columns if c not in ("custom_id", "patient_id", "examination_date")]
wspolne_tygodnie = set(s1["kor"]["custom_id"]) & set(s1["neuro"]["custom_id"])

wynik, wiersze = {}, []
for k in KOHORTY:
    r0, r1, r2 = s0[k], s1[k], s2[k]
    # krok 1: komórki z wartością w surowych, puste po kroku 1 (wpisy typu '<0.4')
    sur = r0.drop_duplicates("custom_id").set_index("custom_id").loc[r1["custom_id"]]
    zgubione = {c: int((sur[c].notna().to_numpy() & r1[c].isna().to_numpy()).sum()) for c in lab}
    zgubione = {c: n for c, n in sorted(zgubione.items(), key=lambda x: -x[1]) if n}
    # krok 2, kroki a i b
    wiek = pd.to_numeric(r1["patient_age"], errors="coerce")
    a = r1[wiek.between(18, 100) & pd.to_numeric(r1["patient_sex"], errors="coerce").notna()]
    b = a[~a["custom_id"].isin(wspolne_tygodnie)]
    # kontrola: po krokach a i b zostaje dokładnie tyle rekordów, ile notebook wypisał przed usuwaniem odstających
    assert ODSTAJACE[k.upper()]["przed"] == len(b), "liczba przed odstającymi nie zgadza się z notebookiem"
    c_n = ODSTAJACE[k.upper()]["usuniete"]
    d_n = ODSTAJACE[k.upper()]["po"] - len(r2)
    assert (r2.groupby("patient_id").size() <= LIMIT_HISTORII[k]).all()

    n0 = len(r0)
    kroki = {"krok 1: powtórzony custom_id": n0 - len(r1),
             "krok 2a: wiek poza 18–100 / brak płci": len(r1) - len(a),
             "krok 2b: tydzień obecny w obu plikach": len(a) - len(b),
             "krok 2c: wartości odstające": c_n,
             "krok 2d: limit historii (11 KOR / 38 NEURO)": d_n}
    for nazwa, n in kroki.items():
        wiersze.append({"grupa": k.upper(), "krok": nazwa, "rekordy": n, "% rekordów surowych": round(100 * n / n0, 2)})
    wiersze.append({"grupa": k.upper(), "krok": "RAZEM kroki 1–2", "rekordy": n0 - len(r2),
                    "% rekordów surowych": round(100 * (n0 - len(r2)) / n0, 2)})

    # filtr „ponad 60% braków”: czy rzadko badani pacjenci zniknęli?
    prof1 = r1[cechy38].notna().groupby(r1["patient_id"]).any().mean(axis=1)
    rzadcy = set(prof1.index[prof1 < 0.4])
    wynik[k] = {
        "rekordy": {"surowe": n0, "po_kroku_1": len(r1), "po_kroku_2": len(r2)},
        "pacjenci": {"surowe": int(r0["patient_id"].nunique()), "po_kroku_2": int(r2["patient_id"].nunique())},
        "kroki": kroki,
        "krok1_komorki_zamienione_na_brak": {"razem": sum(zgubione.values()), "najwiecej": dict(list(zgubione.items())[:4])},
        "rekordy_ponad_60proc_brakow": {"po_kroku_1": int((r1[cechy38].isna().mean(axis=1) > 0.6).sum()),
                                        "po_kroku_2": int((r2[cechy38].isna().mean(axis=1) > 0.6).sum())},
        "pacjenci_ponizej_40proc_zmierzonych_cech": {"po_kroku_1": len(rzadcy),
                                                     "z_nich_po_kroku_2": int(len(rzadcy & set(r2["patient_id"])))},
    }

# udział rzadko badanych pacjentów w danych do modelowania
d4 = po_kroku4()
prof4 = d4[cechy38].notna().groupby([d4["label"], d4["patient_id"]]).any().mean(axis=1)
for k in KOHORTY:
    p = prof4.loc[ETYKIETA[k]]
    wynik[k]["wejscie_sgkf_odsetek_pacjentow_ponizej_40proc"] = round(float((p < 0.4).mean()), 4)

t = pd.DataFrame(wiersze)
print(t.pivot_table(index="krok", columns="grupa", values=["rekordy", "% rekordów surowych"], sort=False).to_string())
for k in KOHORTY:
    w = wynik[k]
    print(f"\n[{k.upper()}] pacjenci {w['pacjenci']['surowe']} -> {w['pacjenci']['po_kroku_2']}")
    print(f"   krok 1, komórki '<x' -> brak: {w['krok1_komorki_zamienione_na_brak']}")
    print(f"   rekordy z >60% braków: {w['rekordy_ponad_60proc_brakow']}; "
          f"pacjenci <40% zmierzonych cech: {w['pacjenci_ponizej_40proc_zmierzonych_cech']}; "
          f"w danych do modelowania: {w['wejscie_sgkf_odsetek_pacjentow_ponizej_40proc']:.2%}")

zapisz("issue46_czyszczenie_a_grupy", {"tabela": wiersze, "szczegoly": wynik})
