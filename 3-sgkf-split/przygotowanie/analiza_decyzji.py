"""
analiza_decyzji.py
==================

CZĘŚĆ A etapu 3 — podkładka liczbowa pod decyzje z 04.10.2026
(1026_REALIZACJA_DO_SGKF.md, sekcja 6; 3-sgkf-split/RAPORT_SGKF_MICE.md, sekcja 2).

Każda liczba cytowana w dokumentach przy tych decyzjach pochodzi z tego
skryptu. Nic nie modyfikuje — czyta dane źródłowe, przygotowuje je tą samą
funkcją co przygotuj_dane.py i zapisuje wyniki do
przygotowanie/results/analiza_decyzji.json.

  A. dane przed i po czyszczeniu
  B. rozjazd czasowy kohort                        (decyzja: ograniczenie, modelujemy na całości)
  C. 63 pacjentów obecnych w obu kohortach         (decyzja: pozytywni, rekordy KOR usunięte)
  D. KREA a eGFR — błąd jednostek                  (decyzja: KREA > 50 albo sprzeczne z eGFR -> brak)
  E. wartości niezgodne z życiem i skrajne-możliwe (decyzja: K > 15, Na < 80 -> brak; reszta zostaje)
  F. wcześniejsze czyszczenie kwantylowe           (stan zastany, notebook z etapu 1)
  G. asymetria braków między kohortami             (decyzja: ograniczenie, raportowane per fold)
  H. zakres skalera dopasowanego na kompletnych wierszach
  I. redundancja MONO / %MONO                      (decyzja 0.4: 38 cech, 35 jako wrażliwość)
  J. cenzurowanie eGFR
  K. kontrole jakości danych po czyszczeniu

Uruchomienie:
    python 3-sgkf-split/przygotowanie/analiza_decyzji.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from przygotuj_dane import (BASE_DIR, KOL_ETYKIETA, KOL_MIESZANY, KOL_PACJENT,  # noqa: E402
                            WYNIKI, ZRODLO, pacjenci_mieszani, przygotuj)

NOTEBOOK_ETAP1 = BASE_DIR / "1-data-preparation" / "scripts-lf" / "aneurysm_data_analysis (1).ipynb"
KOL_DATA = "examination_date"
META_KOLUMNY = ["patient_id", "custom_id", "examination_date", "label", KOL_MIESZANY]
CECHY_POZA_35 = ["CRP", "MONO", "%MONO"]   # różnica między wariantem 38 i 35 cech


def kolumny_cech(df: pd.DataFrame, zestaw: str = "38") -> list:
    pomin = set(META_KOLUMNY) | (set(CECHY_POZA_35) if zestaw == "35" else set())
    return [c for c in df.columns if c not in pomin]


def wspolne_okno_dat(df: pd.DataFrame):
    kor = df.loc[df[KOL_ETYKIETA] == 0, KOL_DATA]
    neuro = df.loc[df[KOL_ETYKIETA] == 1, KOL_DATA]
    return max(kor.min(), neuro.min()), min(kor.max(), neuro.max())


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def naglowek(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


# ---------------------------------------------------------------------------

def a_dane(zrodlo: pd.DataFrame, czyste: pd.DataFrame) -> dict:
    def opis(d):
        p = d.groupby(KOL_PACJENT)[KOL_ETYKIETA].max()
        return {"rekordy": int(len(d)), "rekordy_kor": int((d[KOL_ETYKIETA] == 0).sum()),
                "rekordy_neuro": int((d[KOL_ETYKIETA] == 1).sum()), "pacjenci": int(len(p)),
                "pacjenci_pozytywni": int(p.sum()),
                "pacjenci_w_obu_kohortach": int((d.groupby(KOL_PACJENT)[KOL_ETYKIETA].nunique() > 1).sum())}
    return {"zrodlo": opis(zrodlo), "po_czyszczeniu": opis(czyste)}


def b_rozjazd_czasowy(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["rok"] = d[KOL_DATA].dt.year
    lata = d.groupby(["rok", KOL_ETYKIETA]).size().unstack(fill_value=0)
    kor, neuro = d[d[KOL_ETYKIETA] == 0], d[d[KOL_ETYKIETA] == 1]
    od, do = wspolne_okno_dat(d)
    neuro_okno = neuro[neuro[KOL_DATA].between(od, do)]
    przedzialy = {"do 2009": (0, 2009), "2010–2018": (2010, 2018), "2019": (2019, 2019),
                  "2020": (2020, 2020), "2021": (2021, 2021), "2022–2024": (2022, 2100)}
    tabela = {k: {"kor": int(lata.loc[(lata.index >= a) & (lata.index <= b), 0].sum()),
                  "neuro": int(lata.loc[(lata.index >= a) & (lata.index <= b), 1].sum())}
              for k, (a, b) in przedzialy.items()}
    return {
        "lata": tabela,
        "kor_udzial_2020_2021": _r(kor["rok"].isin([2020, 2021]).mean()),
        "neuro_zakres_lat": [int(neuro["rok"].min()), int(neuro["rok"].max())],
        "wspolne_okno": [str(od.date()), str(do.date())],
        "neuro_rekordy_w_oknie": int(len(neuro_okno)), "neuro_rekordy": int(len(neuro)),
        "neuro_pacjenci_w_oknie": int(neuro_okno[KOL_PACJENT].nunique()),
        "neuro_pacjenci": int(neuro[KOL_PACJENT].nunique()),
    }


def c_mieszani(zrodlo: pd.DataFrame) -> dict:
    d = zrodlo.copy()
    d[KOL_DATA] = pd.to_datetime(d[KOL_DATA])
    mieszani = pacjenci_mieszani(d)
    m = d[d[KOL_PACJENT].isin(mieszani)]
    uklad, odstep = {"kor_przed_neuro": 0, "neuro_przed_kor": 0, "przeplatane": 0}, []
    for _, g in m.groupby(KOL_PACJENT):
        k, n = g.loc[g[KOL_ETYKIETA] == 0, KOL_DATA], g.loc[g[KOL_ETYKIETA] == 1, KOL_DATA]
        if k.max() < n.min():
            uklad["kor_przed_neuro"] += 1
            odstep.append((n.min() - k.max()).days)
        elif n.max() < k.min():
            uklad["neuro_przed_kor"] += 1
        else:
            uklad["przeplatane"] += 1
    return {"pacjenci": len(mieszani), "rekordy": int(len(m)),
            "rekordy_kor": int((m[KOL_ETYKIETA] == 0).sum()), "rekordy_neuro": int((m[KOL_ETYKIETA] == 1).sum()),
            "uklad_czasowy": uklad,
            "odstep_kor_neuro_dni": {"mediana": int(np.median(odstep)), "min": int(min(odstep)), "max": int(max(odstep))}}


def d_krea(zrodlo: pd.DataFrame) -> dict:
    k, e = zrodlo["KREA"], zrodlo["eGFR-MDRD"]
    przedzialy = [(0, 1.2), (1.2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf)]
    tabela = {}
    for a, b in przedzialy:
        m = (k > a) & (k <= b)
        tabela[f"({a}, {b}]"] = {
            "wartosci": int(m.sum()),
            "egfr_mediana": _r(e[m].median(), 1),
            "udzial_egfr_ge_60": _r((e[m] >= 60).sum() / m.sum()),
            "udzial_egfr_ge_30": _r((e[m] >= 30).sum() / m.sum()),
            "udzial_brak_egfr": _r(e[m].isna().mean()),
        }
    r_jednostka = k > 50
    r_sprzecznosc = (k > 10) & (k <= 50) & (e >= 30)
    usuniete = r_jednostka | r_sprzecznosc
    zostaja_ponad_10 = (k > 10) & ~usuniete
    przeliczone = k[r_sprzecznosc & (e >= 60)] / 88.4
    return {
        "wartosci_krea": int(k.notna().sum()),
        "przedzialy": tabela,
        "regula_ponad_50": int(r_jednostka.sum()),
        "regula_sprzecznosc_z_egfr": int(r_sprzecznosc.sum()),
        "usuniete_razem": int(usuniete.sum()),
        "usuniete_udzial": _r(usuniete.sum() / k.notna().sum()),
        "usuniete_pacjenci": int(zrodlo.loc[usuniete, KOL_PACJENT].nunique()),
        "usuniete_neuro": int((usuniete & (zrodlo[KOL_ETYKIETA] == 1)).sum()),
        "zostaja_krea_ponad_10": int(zostaja_ponad_10.sum()),
        "zostaja_krea_ponad_10_egfr_ponizej_30": int((zostaja_ponad_10 & (e < 30)).sum()),
        "zostaja_krea_ponad_10_brak_egfr": int((zostaja_ponad_10 & e.isna()).sum()),
        "prosty_prog_20_usunalby": int((k > 20).sum()),
        "prosty_prog_20_usunalby_przy_egfr_ponizej_30": int(((k > 20) & (e < 30)).sum()),
        "sprzeczne_z_egfr_ge_60_po_przeliczeniu_mg_dl_mediana": _r(przeliczone.median(), 2),
    }


def e_wartosci_graniczne(zrodlo: pd.DataFrame, czyste: pd.DataFrame) -> dict:
    def ile(m):
        return {"wartosci": int(m.sum()), "kor": int((m & (zrodlo[KOL_ETYKIETA] == 0)).sum()),
                "neuro": int((m & (zrodlo[KOL_ETYKIETA] == 1)).sum())}
    feat = kolumny_cech(czyste)
    return {
        "K_ponad_15_usuniete": {**ile(zrodlo["K"] > 15), "max": _r(zrodlo["K"].max(), 2)},
        "K_9_15_zostaja": ile((zrodlo["K"] > 9) & (zrodlo["K"] <= 15)),
        "Na_ponizej_80_usuniete": {**ile(zrodlo["Na"] < 80), "min": _r(zrodlo["Na"].min(), 2)},
        "Na_80_100_zostaja": ile((zrodlo["Na"] >= 80) & (zrodlo["Na"] < 100)),
        "WBC_ponad_200_zostaja": {**ile(zrodlo["WBC"] > 200), "max": _r(zrodlo["WBC"].max(), 2)},
        "zakresy_po_czyszczeniu": {c: [_r(czyste[c].min(), 3), _r(czyste[c].max(), 3)] for c in feat},
    }


def f_czyszczenie_kwantylowe() -> dict:
    """Odczyt wyników z notebooka etapu 1 (Liwia): usuwanie rekordów po kwantylach."""
    if not NOTEBOOK_ETAP1.exists():
        return {"pominiete": f"brak {NOTEBOOK_ETAP1.name}"}
    nb = json.loads(NOTEBOOK_ETAP1.read_text(encoding="utf-8"))
    kroki, ksztalty = [], {}
    wzor_wyw = re.compile(r"clean_single_parameter\(\s*\w+\s*,\s*['\"]([^'\"]+)['\"]\s*,\s*['\"](KOR|NEURO)['\"]\s*,\s*q_level\s*=\s*([\d.]+)")
    wzor_wyn = re.compile(r"Usunięto rekordów: (\d+) \| Granice: \[([-\d.]+), ([-\d.]+)\]")
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        zrodlo = "".join(c["source"])
        wyjscie = "".join("".join(o.get("text", "")) for o in c.get("outputs", []))
        aktywne = [ln for ln in zrodlo.splitlines() if "clean_single_parameter(" in ln and not ln.strip().startswith("#")
                   and "def " not in ln]
        for ln, wyn in zip(aktywne, wzor_wyn.findall(wyjscie)):
            w = wzor_wyw.search(ln)
            if w:
                kroki.append({"kohorta": w.group(2), "cecha": w.group(1), "q": float(w.group(3)),
                              "usuniete": int(wyn[0]), "dolna": float(wyn[1]), "gorna": float(wyn[2])})
        m = re.search(r"Shape PRZED usunięciem outliers: \((\d+), \d+\)\s*Shape PO usunięciu outliers: \((\d+), \d+\)", wyjscie)
        if m:
            klucz = "KOR" if "df_work_kor" in zrodlo else "NEURO"
            ksztalty[klucz] = {"przed": int(m.group(1)), "po": int(m.group(2)),
                               "usuniete": int(m.group(1)) - int(m.group(2)),
                               "udzial": _r((int(m.group(1)) - int(m.group(2))) / int(m.group(1)))}
    t = pd.DataFrame(kroki)
    q = {k: [float(g["q"].min()), float(g["q"].max())] for k, g in t.groupby("kohorta")} if len(t) else {}
    granice = {f"{r.kohorta}_{r.cecha}": [r.dolna, r.gorna] for r in t.itertuples()
               if r.cecha in ("K", "KREA", "Na", "WBC")}
    return {"rekordy_przed_po": ksztalty, "zakres_kwantyli": q, "kroki": int(len(t)),
            "granice_wybranych_cech": granice}


def g_braki(czyste: pd.DataFrame) -> dict:
    wyn = {}
    for zestaw in ("38", "35"):
        feat = kolumny_cech(czyste, zestaw)
        b = czyste[feat].isna()
        udz = b.groupby(czyste[KOL_ETYKIETA]).apply(lambda x: x.to_numpy().mean())
        kompl = czyste[feat].notna().all(axis=1).groupby(czyste[KOL_ETYKIETA]).sum()
        wyn[zestaw] = {"braki_razem": int(b.to_numpy().sum()), "udzial_razem": _r(b.to_numpy().mean()),
                       "udzial_kor": _r(udz[0]), "udzial_neuro": _r(udz[1]),
                       "kompletne_kor": int(kompl[0]), "kompletne_neuro": int(kompl[1]),
                       "kompletne_udzial_kor_wsrod_kompletnych": _r(kompl[0] / kompl.sum())}
    feat = kolumny_cech(czyste, "38")
    per = czyste[feat].isna().groupby(czyste[KOL_ETYKIETA]).mean().T
    per["roznica"] = per[1] - per[0]
    top = per.reindex(per["roznica"].abs().sort_values(ascending=False).index).head(8)
    wyn["najwieksze_roznice"] = {c: {"kor": _r(r[0]), "neuro": _r(r[1])} for c, r in top.iterrows()}
    return wyn


def h_zakres_skalera(czyste: pd.DataFrame) -> dict:
    wyn = {}
    for zestaw in ("38", "35"):
        X = czyste[kolumny_cech(czyste, zestaw)]
        k = X.dropna()
        poza = ((X < k.min()) | (X > k.max())).to_numpy().sum()
        wyn[zestaw] = {"poza_zakresem": int(poza), "udzial": _r(poza / X.notna().to_numpy().sum(), 6)}
    return wyn


def i_redundancja_mono(czyste: pd.DataFrame) -> dict:
    s = czyste[["%NEUT", "%LYMPH", "%EO", "%BAZO", "%MONO"]].sum(axis=1, min_count=5).dropna()
    pelne = czyste[["MONO", "%MONO", "WBC"]].dropna()
    przyblizone = pelne["%MONO"] * pelne["WBC"] / 100
    blad_wzgl = ((pelne["MONO"] - przyblizone).abs() / pelne["MONO"].clip(lower=1e-6))
    return {"suma_rozmazu_z_mono": {"srednia": _r(s.mean(), 2), "odch_std": _r(s.std(), 2),
                                    "min": _r(s.min(), 1), "max": _r(s.max(), 1), "n": int(len(s))},
            "korelacja_MONO_z_procentMONO_x_WBC": _r(np.corrcoef(pelne["MONO"], przyblizone)[0, 1]),
            "mediana_bledu_wzglednego": _r(blad_wzgl.median()),
            "cechy_tylko_w_38": CECHY_POZA_35}


def j_egfr(czyste: pd.DataFrame) -> dict:
    return {"eGFR-MDRD_max": _r(czyste["eGFR-MDRD"].max(), 1),
            "eGFR-MDRD_udzial_rownych_max": _r((czyste["eGFR-MDRD"] == czyste["eGFR-MDRD"].max()).mean()),
            "eGFRCKD_max": _r(czyste["eGFRCKD"].max(), 1),
            "eGFRCKD_udzial_rownych_max": _r((czyste["eGFRCKD"] == czyste["eGFRCKD"].max()).mean())}


def k_kontrole(czyste: pd.DataFrame) -> dict:
    feat = kolumny_cech(czyste)
    proc = [c for c in feat if c.startswith("%")]
    return {"custom_id_unikalny": bool(czyste["custom_id"].is_unique),
            "pacjent_w_jednej_kohorcie": bool((czyste.groupby(KOL_PACJENT)[KOL_ETYKIETA].nunique() == 1).all()),
            "wartosci_ujemne": int((czyste[feat] < 0).to_numpy().sum()),
            "procenty_poza_0_100": int(((czyste[proc] < 0) | (czyste[proc] > 100)).to_numpy().sum()),
            "braki_plec": int(czyste["patient_sex"].isna().sum()),
            "braki_wiek": int(czyste["patient_age"].isna().sum()),
            "wiek_zakres": [_r(czyste["patient_age"].min(), 2), _r(czyste["patient_age"].max(), 2)],
            "wiersze_bez_zadnej_cechy": int(czyste[feat].isna().all(axis=1).sum())}


# ---------------------------------------------------------------------------

def main() -> None:
    zrodlo = pd.read_csv(ZRODLO)
    czyste, _, _ = przygotuj(zrodlo)
    zrodlo_dt = zrodlo.copy()
    zrodlo_dt[KOL_DATA] = pd.to_datetime(zrodlo_dt[KOL_DATA])
    czyste[KOL_DATA] = pd.to_datetime(czyste[KOL_DATA])

    wyniki = {
        "A_dane": a_dane(zrodlo, czyste),
        "B_rozjazd_czasowy": b_rozjazd_czasowy(czyste),
        "C_pacjenci_mieszani": c_mieszani(zrodlo),
        "D_krea_egfr": d_krea(zrodlo),
        "E_wartosci_graniczne": e_wartosci_graniczne(zrodlo, czyste),
        "F_czyszczenie_kwantylowe_etap1": f_czyszczenie_kwantylowe(),
        "G_asymetria_brakow": g_braki(czyste),
        "H_zakres_skalera": h_zakres_skalera(czyste),
        "I_redundancja_mono": i_redundancja_mono(czyste),
        "J_cenzurowanie_egfr": j_egfr(czyste),
        "K_kontrole_jakosci": k_kontrole(czyste),
    }

    for klucz, w in wyniki.items():
        naglowek(klucz)
        if klucz == "E_wartosci_graniczne":
            w = {k: v for k, v in w.items() if k != "zakresy_po_czyszczeniu"}
        print(json.dumps(w, indent=2, ensure_ascii=False))

    WYNIKI.mkdir(parents=True, exist_ok=True)
    (WYNIKI / "analiza_decyzji.json").write_text(json.dumps(wyniki, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"\nZapisano -> {WYNIKI.relative_to(BASE_DIR)}/analiza_decyzji.json")


if __name__ == "__main__":
    main()
