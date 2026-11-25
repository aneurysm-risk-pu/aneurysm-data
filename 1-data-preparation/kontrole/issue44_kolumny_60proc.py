"""
#44 Skąd mamy 38 cech: która reguła usunęła ALT, AST, BUN i P-LCR.

W notebooku kroku 2 komórka 19 usuwa kolumny z ponad 60% braków OSOBNO
w KOR i w NEURO, a komórka 20 zostawia tylko kolumny wspólne dla obu grup.
Razem daje to regułę „ponad 60% braków w którejkolwiek grupie”. Skrypt liczy
braki każdej z 40 kolumn badań (pliki po kroku 1, rekordy z wiekiem 18–100,
tak jak w notebooku przed tą regułą) i sprawdza, która reguła daje kolumny
faktycznie usunięte (brakujące w *_shortend.csv):
  a) „osobno”: tylko komórka 19,
  b) „w którejkolwiek grupie”: komórki 19 + 20.

Uruchomienie: python 1-data-preparation/kontrole/issue44_kolumny_60proc.py
"""

import pandas as pd

from wspolne import KOHORTY, cechy_lab, po_kroku1, po_kroku2, surowe, zapisz

PROG = 60.0

lab = cechy_lab(surowe("kor").head(1))
braki = {}
for k in KOHORTY:
    d = po_kroku1(k)
    d = d[pd.to_numeric(d["patient_age"], errors="coerce").between(18, 100)]
    braki[k] = (d[lab].isna().mean() * 100).round(1)
braki = pd.DataFrame(braki)

osobno = {k: sorted(braki.index[braki[k] > PROG]) for k in KOHORTY}
ktorakolwiek = sorted(braki.index[(braki > PROG).any(axis=1)])
w_danych = set(po_kroku2("kor").columns) & set(po_kroku2("neuro").columns)
usuniete = sorted(c for c in lab if c not in w_danych)

print("Braki [%] kolumn, które usunięto, i kolumn blisko progu 60% (50–70%):")
blisko = braki[(braki > 50).any(axis=1) & (braki < 70).any(axis=1)]
print(braki.loc[sorted(set(usuniete) | set(blisko.index))].to_string())
print(f"\nReguła „osobno” (sama komórka 19): KOR usuwa {osobno['kor']}, NEURO usuwa {osobno['neuro']}")
print(f"Reguła „w którejkolwiek grupie” (komórki 19 + 20): {ktorakolwiek}")
print(f"Faktycznie usunięte (brak w obu *_shortend.csv): {usuniete}")
zgodna = "w którejkolwiek grupie" if ktorakolwiek == usuniete else ("osobno" if osobno["kor"] == osobno["neuro"] == usuniete else "żadna")
print(f"-> z danymi zgadza się reguła: {zgodna}; zostaje {len(lab) - len(usuniete)} kolumn badań + wiek i płeć = {len(lab) - len(usuniete) + 2} cech")

zapisz("issue44_kolumny_60proc", {
    "prog_proc": PROG,
    "braki_proc": braki.loc[sorted(set(usuniete) | set(blisko.index))].to_dict(orient="index"),
    "regula_osobno_usuwa": osobno,
    "regula_w_ktorejkolwiek_grupie_usuwa": ktorakolwiek,
    "faktycznie_usuniete": usuniete,
    "zgodna_regula": zgodna,
    "liczba_cech": len(lab) - len(usuniete) + 2,
})
