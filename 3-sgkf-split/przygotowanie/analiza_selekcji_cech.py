"""
analiza_selekcji_cech.py
========================

CZĘŚĆ A etapu 3 — czy usunięcie CRP, MONO i %MONO (etap 1, 08.06.2026)
było słuszne? Podkładka pod decyzję o jednym zestawie cech.

  1. Odtworzenie procedury z notebooka
     1-data-preparation/scripts-lf/aneurysm_data_merge.ipynb na danych
     źródłowych: korelacja każdej cechy z etykietą (Pearson, Spearman, Kendall,
     Phi-K), 10 najsłabiej skorelowanych w każdej mierze, zliczenie wystąpień,
     usunięcie 3 najczęstszych (remisy rozstrzygane alfabetycznie, jak
     w notebooku: sort_values(['liczba', 'Parametr'], ascending=[False, True])).
  2. Porównanie z wynikami wydrukowanymi w notebooku.
  3. Remis na granicy wyboru — które cechy miały tyle samo „głosów”.
  4. Ta sama procedura na danych po przygotowaniu (04.10.2026) — czy wybór
     jest stabilny.
  5. Własności samych trzech cech niezależne od etykiety: redundancja MONO
     i %MONO, braki i rozkład CRP w kohortach.

Wynik: przygotowanie/results/analiza_selekcji_cech.json

Uruchomienie:
    python 3-sgkf-split/przygotowanie/analiza_selekcji_cech.py
"""

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from przygotuj_dane import BASE_DIR, KOL_ETYKIETA, WYJSCIE, WYNIKI, ZRODLO  # noqa: E402

warnings.filterwarnings("ignore")
NOTEBOOK = BASE_DIR / "1-data-preparation" / "scripts-lf" / "aneurysm_data_merge.ipynb"
USUNIETE = ["CRP", "MONO", "%MONO"]
POMIN = ["patient_id"]


def _r(x, n=6):
    return None if x is None or pd.isna(x) else round(float(x), n)


def korelacje_z_etykieta(df: pd.DataFrame) -> dict:
    """Dokładnie jak w notebooku: kolumny liczbowe bez patient_id, korelacja z 'label'."""
    import phik  # noqa: F401  (rejestruje DataFrame.phik_matrix)
    num = df.select_dtypes(include=["float64", "int64"]).drop(columns=[c for c in POMIN if c in df.columns])
    return {
        "pearson": num.corr(method="pearson")[KOL_ETYKIETA].drop(KOL_ETYKIETA),
        "spearman": num.corr(method="spearman")[KOL_ETYKIETA].drop(KOL_ETYKIETA),
        "kendall": num.corr(method="kendall")[KOL_ETYKIETA].drop(KOL_ETYKIETA),
        "phik": num.phik_matrix(verbose=False)[KOL_ETYKIETA].drop(KOL_ETYKIETA),
    }


def selekcja_jak_w_notebooku(kor: dict, n: int = 10, ile: int = 3) -> dict:
    najslabsze = {m: (s.sort_values(ascending=True) if m == "phik" else s.abs().sort_values(ascending=True)).head(n)
                  for m, s in kor.items()}
    licznik = Counter(c for s in najslabsze.values() for c in s.index)
    tab = (pd.DataFrame.from_dict(licznik, orient="index", columns=["liczba"])
           .rename_axis("Parametr").reset_index()
           .sort_values(["liczba", "Parametr"], ascending=[False, True]))
    wybrane = tab.head(ile)["Parametr"].tolist()
    prog = int(tab.iloc[ile - 1]["liczba"])
    return {
        "najslabsze_10": {m: {c: _r(v) for c, v in s.items()} for m, s in najslabsze.items()},
        "glosy": dict(zip(tab["Parametr"], tab["liczba"].astype(int))),
        "wybrane": wybrane,
        "remis_na_granicy": tab.loc[tab["liczba"] == prog, "Parametr"].tolist(),
        "glosy_na_granicy": prog,
    }


def wydruki_notebooka() -> dict:
    """Wartości 'TOP 10 NAJMNIEJ SKORELOWANYCH' wydrukowane w notebooku."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tekst = ""
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", []))
            if "TOP 10 NAJMNIEJ SKORELOWANYCH" in t:
                tekst = t
    wynik = {}
    for nazwa, klucz in (("PEARSON", "pearson"), ("SPEARMAN", "spearman"), ("KENDALL", "kendall"), ("PHI-K", "phik")):
        blok = re.search(rf"--- {re.escape(nazwa)} ---\n(.*?)Name: label", tekst, re.S)
        if blok:
            wynik[klucz] = {m.group(1): float(m.group(2))
                            for m in re.finditer(r"^(\S+)\s+([\d.eE-]+)\s*$", blok.group(1), re.M)}
    return wynik


def zgodnosc_z_notebookiem(wydruk: dict, policzone: dict) -> dict:
    out = {}
    for m, wart in wydruk.items():
        # notebook drukuje |r| dla Pearsona, Spearmana i Kendalla, a Phi-K bez zmian (zawsze >= 0)
        rozn = [abs(v - abs(policzone[m][c])) for c, v in wart.items() if c in policzone[m]]
        out[m] = {"cechy": len(rozn), "maks_roznica": _r(max(rozn)) if rozn else None}
    return out


def wlasnosci_trzech_cech(df: pd.DataFrame) -> dict:
    d = df.copy()
    rozmaz = d[["%NEUT", "%LYMPH", "%EO", "%BAZO"]].sum(axis=1, min_count=4)
    m1 = d["%MONO"].notna() & rozmaz.notna()
    pelne = d[["MONO", "%MONO", "WBC"]].dropna()
    przybl = pelne["%MONO"] * pelne["WBC"] / 100
    crp = d["CRP"]
    k, n = d[KOL_ETYKIETA] == 0, d[KOL_ETYKIETA] == 1
    return {
        "procentMONO_z_100_minus_reszta_rozmazu": {
            "korelacja": _r(np.corrcoef(d.loc[m1, "%MONO"], 100 - rozmaz[m1])[0, 1], 4),
            "mediana_bezwzgl_roznicy_pp": _r((d.loc[m1, "%MONO"] - (100 - rozmaz[m1])).abs().median(), 3)},
        "MONO_z_procentMONO_razy_WBC": {
            "korelacja": _r(np.corrcoef(pelne["MONO"], przybl)[0, 1], 4),
            "mediana_bledu_wzglednego": _r(((pelne["MONO"] - przybl).abs() / pelne["MONO"].clip(lower=1e-6)).median(), 4)},
        "CRP": {
            "braki_kor": _r(crp[k].isna().mean(), 4), "braki_neuro": _r(crp[n].isna().mean(), 4),
            "mediana_kor": _r(crp[k].median(), 2), "mediana_neuro": _r(crp[n].median(), 2),
            "p75_kor": _r(crp[k].quantile(0.75), 2), "p75_neuro": _r(crp[n].quantile(0.75), 2),
            "spearman_z_etykieta": _r(d[["CRP", KOL_ETYKIETA]].corr(method="spearman").iloc[0, 1], 4)},
    }


def main() -> None:
    zrodlo = pd.read_csv(ZRODLO)
    print("[1] Odtworzenie procedury z notebooka na danych źródłowych ...")
    kor_zr = korelacje_z_etykieta(zrodlo)
    sel_zr = selekcja_jak_w_notebooku(kor_zr)
    print(f"    wybrane: {sel_zr['wybrane']}   (w notebooku: {USUNIETE})")
    print(f"    remis na granicy ({sel_zr['glosy_na_granicy']} głosy): {sel_zr['remis_na_granicy']}")

    print("[2] Zgodność z wartościami wydrukowanymi w notebooku ...")
    zg = zgodnosc_z_notebookiem(wydruki_notebooka(), kor_zr)
    print(f"    {zg}")

    print("[3] Ta sama procedura na danych po przygotowaniu (04.10.2026) ...")
    przyg = pd.read_csv(WYJSCIE)
    sel_pr = selekcja_jak_w_notebooku(korelacje_z_etykieta(przyg))
    print(f"    wybrane: {sel_pr['wybrane']}; remis na granicy: {sel_pr['remis_na_granicy']}")

    print("[4] Własności CRP, MONO, %MONO niezależne od etykiety ...")
    wl = wlasnosci_trzech_cech(przyg)
    print(json.dumps(wl, indent=2, ensure_ascii=False))

    wynik = {
        "procedura_notebook": {"plik": str(NOTEBOOK.relative_to(BASE_DIR)), "usuniete": USUNIETE},
        "odtworzenie_na_zrodle": sel_zr,
        "korelacje_usunietych_ze_zrodla": {c: {m: _r(kor_zr[m][c]) for m in kor_zr} for c in USUNIETE},
        "zgodnosc_z_wydrukiem_notebooka": zg,
        "ta_sama_procedura_po_przygotowaniu": {k: sel_pr[k] for k in ("wybrane", "remis_na_granicy", "glosy_na_granicy")},
        "wlasnosci_trzech_cech": wl,
    }
    (WYNIKI / "analiza_selekcji_cech.json").write_text(json.dumps(wynik, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
    print(f"\nZapisano -> {(WYNIKI / 'analiza_selekcji_cech.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
