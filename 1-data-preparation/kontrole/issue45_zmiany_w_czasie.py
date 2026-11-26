"""
#45 Zmiany parametrów w czasie (NEURO, lata 2010–2024), na danych z preprocessingu Liwii
(data/processed/neuro_preprocessed_final.csv).

Pomijamy pacjentów z foldu 4 (zbiór testowy, przydział: 3-sgkf-split/sgkf/results/pacjent_fold.csv).
KOR ma tylko lata 2019–2021, więc trend liczymy w NEURO. Dla każdej z 36 cech badań:
  - korelacja Spearmana wartości z rokiem badania,
  - różnica median 2010–2014 vs 2020–2024 (w odchyleniach standardowych cechy),
  - skok wokół 2015 (mediana 2013–2014 vs 2016–2017; zmiana norm morfologii od 17.04.2015),
  - zmiana odsetka rekordów z pomiarem (czy badanie zaczęto/przestano zlecać).
Tabele rok po roku: results/issue45_mediany_wg_roku.csv, results/issue45_pomiar_wg_roku.csv.

Uruchomienie: python 1-data-preparation/kontrole/issue45_zmiany_w_czasie.py
"""

import warnings

import pandas as pd
from scipy.stats import spearmanr

from wspolne import BASE, WYNIKI, cechy_lab, wynik_liwii, zapisz

PROG_RHO, PROG_SD, PROG_POMIAR = 0.10, 0.20, 20.0
warnings.filterwarnings("ignore", "Mean of empty slice")   # cechy niemierzone w części lat -> mediana NaN

d = wynik_liwii("neuro")
fold = pd.read_csv(BASE / "3-sgkf-split/sgkf/results/pacjent_fold.csv").set_index("patient_id")["fold"]
w_foldzie4 = d["patient_id"].map(fold).eq(4)
d = d[~w_foldzie4].copy()
d["rok"] = pd.to_datetime(d["examination_date"]).dt.year
d = d[d["rok"] >= 2010]                  # przed 2010 jest pojedynczy rekord
cechy = cechy_lab(d)
print(f"NEURO bez foldu 4: {d['patient_id'].nunique()} pacjentów, {len(d)} rekordów "
      f"(pominięto {int(w_foldzie4.sum())} rekordów z foldu 4)")

lata = d.groupby("rok").agg(rekordy=("custom_id", "size"), pacjenci=("patient_id", "nunique"),
                            wiek_mediana=("patient_age", "median"), odsetek_mezczyzn=("patient_sex", "mean")).round(2)
print(lata.T.to_string())


def okres(od, do):
    return d[d["rok"].between(od, do)]


wiersze = []
for c in cechy:
    x = d[["rok", c]].dropna()
    sd = d[c].std()
    wiersze.append({
        "cecha": c, "n": len(x), "rho_z_rokiem": round(spearmanr(x["rok"], x[c])[0], 3),
        "zmiana_2010_14_vs_2020_24_sd": round((okres(2020, 2024)[c].median() - okres(2010, 2014)[c].median()) / sd, 2),
        "skok_2015_sd": round((okres(2016, 2017)[c].median() - okres(2013, 2014)[c].median()) / sd, 2),
        "pomiar_2010_14_proc": round(100 * okres(2010, 2014)[c].notna().mean(), 1),
        "pomiar_2020_24_proc": round(100 * okres(2020, 2024)[c].notna().mean(), 1),
    })
t = pd.DataFrame(wiersze).set_index("cecha")
t["zmiana_pomiaru_pp"] = t["pomiar_2020_24_proc"] - t["pomiar_2010_14_proc"]
pomiar_rok = d[cechy].notna().groupby(d["rok"]).mean() * 100

wartosci = t[(t["rho_z_rokiem"].abs() >= PROG_RHO) | (t["zmiana_2010_14_vs_2020_24_sd"].abs() >= PROG_SD)]
pomiar = t[t["zmiana_pomiaru_pp"].abs() >= PROG_POMIAR]
pomiar = pomiar.assign(pierwszy_rok_ponad_50proc=[int(pomiar_rok.index[pomiar_rok[c] > 50][0]) for c in pomiar.index])
print(f"\nCechy, których WARTOŚCI przesuwają się w czasie (|rho| >= {PROG_RHO} lub zmiana >= {PROG_SD} SD):")
print(wartosci.sort_values("rho_z_rokiem", key=abs, ascending=False).to_string())
print(f"\nCechy, które zaczęto albo przestano MIERZYĆ (zmiana odsetka pomiaru >= {PROG_POMIAR:.0f} pp):")
kol = ["pomiar_2010_14_proc", "pomiar_2020_24_proc", "zmiana_pomiaru_pp", "pierwszy_rok_ponad_50proc"]
print(pomiar[kol].to_string())

WYNIKI.mkdir(exist_ok=True)
d.groupby("rok")[cechy].median().T.round(3).to_csv(WYNIKI / "issue45_mediany_wg_roku.csv")
pomiar_rok.T.round(1).to_csv(WYNIKI / "issue45_pomiar_wg_roku.csv")
zapisz("issue45_zmiany_w_czasie", {
    "dane": "data/processed/neuro_preprocessed_final.csv, bez pacjentów z foldu 4, lata 2010–2024",
    "rok_po_roku": lata.to_dict(orient="index"),
    "progi": {"rho": PROG_RHO, "zmiana_sd": PROG_SD, "zmiana_pomiaru_pp": PROG_POMIAR},
    "przesuwajace_sie_wartosci": wartosci.to_dict(orient="index"),
    "zmiana_zlecania": pomiar[kol].to_dict(orient="index"),
    "wszystkie_cechy": t.to_dict(orient="index"),
})
