"""
#46 Preprocessing a różnice między grupami (final_preprocessing.py Liwii).

Dla każdego kroku jej skryptu: ile rekordów i pacjentów ubyło w KOR i w NEURO.
Do tego trzy sprawdzenia kroków, które najmocniej różnicują grupy:
  - krok 6: czy pacjenci obecni w obu grupach to te same osoby (płeć i rok urodzenia),
  - krok 9: czy rekordy z ponad 60% braków odpadają równo w czasie (NEURO przed i od 2016),
  - brak limitu historii: ile rekordów ma pacjent (wcześniej limit 11 KOR / 38 NEURO).
Porównanie z poprzednim plikiem wejściowym etapu 3: data/imputation-inputs/aneurysm_concatted.csv.

Uruchomienie: python 1-data-preparation/kontrole/issue46_czyszczenie_a_grupy.py
"""

import pandas as pd

from wspolne import DANE, KOHORTY, cechy_lab, przebieg, surowe, zapisz

kroki = przebieg()
stan = dict(kroki)
n0 = {k: len(kroki[0][1][k]) for k in KOHORTY}

# 1. ubytek rekordów i pacjentów krok po kroku
wiersze = []
for (nazwa, d), (_, poprz) in zip(kroki[1:], kroki[:-1]):
    if all(len(poprz[k]) == len(d[k]) for k in KOHORTY):
        continue                                    # krok nie usuwa rekordów
    for k in KOHORTY:
        ubylo = len(poprz[k]) - len(d[k])
        wiersze.append({"krok": nazwa, "grupa": k.upper(), "rekordy": ubylo,
                        "% rekordów surowych": round(100 * ubylo / n0[k], 2),
                        "pacjenci": poprz[k]["patient_id"].nunique() - d[k]["patient_id"].nunique()})
koniec = stan["10 tylko kolumny wspólne"]
for k in KOHORTY:
    wiersze.append({"krok": "RAZEM", "grupa": k.upper(), "rekordy": n0[k] - len(koniec[k]),
                    "% rekordów surowych": round(100 * (n0[k] - len(koniec[k])) / n0[k], 2),
                    "pacjenci": kroki[0][1][k]["patient_id"].nunique() - koniec[k]["patient_id"].nunique()})
t = pd.DataFrame(wiersze)
print(t.pivot_table(index="krok", columns="grupa", values=["rekordy", "% rekordów surowych", "pacjenci"],
                    sort=False).to_string())


# 2. krok 6: pacjenci obecni w obu grupach — te same osoby?
def osoby(k):
    d = surowe(k)[["patient_id", "examination_date", "patient_age", "patient_sex"]].copy()
    data = pd.to_datetime(d["examination_date"])
    d["rok_urodzenia"] = data.dt.year + data.dt.dayofyear / 365.25 - pd.to_numeric(d["patient_age"], errors="coerce")
    d["plec"] = pd.to_numeric(d["patient_sex"], errors="coerce")
    return d.groupby("patient_id").agg(plec=("plec", "first"), rok_urodzenia=("rok_urodzenia", "median"))


o = osoby("kor").join(osoby("neuro"), how="inner", lsuffix="_kor", rsuffix="_neuro")
zgodni = int(((o["plec_kor"] == o["plec_neuro"]) & ((o["rok_urodzenia_kor"] - o["rok_urodzenia_neuro"]).abs() <= 1)).sum())
neuro_pac = kroki[0][1]["neuro"]["patient_id"].nunique()
print(f"\nKrok 6: {len(o)} pacjentów w obu grupach; ta sama płeć i rok urodzenia (±1): {zgodni} "
      f"-> to te same osoby. Usunięcie zabiera {len(o) / neuro_pac:.1%} pacjentów NEURO.")

# 3. krok 9: rekordy z ponad 60% braków w NEURO przed i od 2016 (NEUT, NRBC mierzone od 2016)
przed9, po9 = stan["8 walidacja wartości"]["neuro"], stan["9 ponad 60% braków (kolumny, potem rekordy)"]["neuro"]
okres = pd.to_datetime(przed9["examination_date"]).dt.year.ge(2016).map({False: "przed 2016", True: "od 2016"})
usun = (~przed9["custom_id"].isin(po9["custom_id"])).groupby(okres).mean().round(3).to_dict()
print(f"Krok 9 w NEURO: odsetek usuniętych rekordów {usun}")

# 4. limit historii i rzadko badani pacjenci: nowe pliki vs poprzedni plik wejściowy etapu 3
stary = pd.read_csv(DANE / "imputation-inputs" / "aneurysm_concatted.csv")
porownanie = {}
for k, etyk in (("kor", 0), ("neuro", 1)):
    n, s = koniec[k], stary[stary["label"] == etyk]
    lab = cechy_lab(n)
    rzadcy = n[lab].notna().groupby(n["patient_id"]).any().mean(axis=1).lt(0.4).mean()
    porownanie[k] = {
        "rekordy_nowe": len(n), "rekordy_poprzednie": len(s),
        "pacjenci_nowe": int(n["patient_id"].nunique()), "pacjenci_poprzednie": int(s["patient_id"].nunique()),
        "maks_rekordow_na_pacjenta_nowe": int(n.groupby("patient_id").size().max()),
        "maks_rekordow_na_pacjenta_poprzednie": int(s.groupby("patient_id").size().max()),
        "odsetek_pacjentow_ponizej_40proc_zmierzonych_cech": round(float(rzadcy), 4),
    }
print("\nNowe pliki a poprzedni plik wejściowy etapu 3:")
print(pd.DataFrame(porownanie).to_string())

zapisz("issue46_czyszczenie_a_grupy", {
    "skrypt_sprawdzany": "1-data-preparation/scripts-lf/final_preprocessing.py",
    "ubytek_krok_po_kroku": wiersze,
    "krok6_te_same_osoby": {"pacjenci_w_obu_grupach": len(o), "zgodna_plec_i_rok_urodzenia": zgodni,
                            "udzial_pacjentow_neuro": round(len(o) / neuro_pac, 4)},
    "krok9_neuro_odsetek_usunietych_wg_okresu": usun,
    "porownanie_z_poprzednim_plikiem": porownanie,
})
