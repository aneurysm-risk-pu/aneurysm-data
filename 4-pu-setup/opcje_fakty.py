"""
opcje_fakty.py
==============

Fakty o wykonalnosci opcji modelowania opisanych w 4-pu-setup/OPCJE_MODELOWANIA.md.

Kazda liczba z tego skryptu odpowiada na pytanie "czy dana opcja jest w ogole
wykonalna na naszych danych", a nie "ktora opcja jest lepsza". Skrypt nie trenuje
modeli i nie korzysta z etykiet do zadnego wyboru; etykieta sluzy tylko do
rozbicia statystyk na kohorty.

  A. Ile rekordow ma pacjent (opcje jednostki: profil, pierwszy rekord, trajektoria).
  B. Udzial brakow przy roznych reprezentacjach pacjenta (wszystkie rekordy,
     pierwszy rekord, mediana pacjenta przed imputacja).
  C. Bloki cech brakujacych razem i wzorce brakow: czy wzorce NEURO wystepuja
     tez w KOR (opcje dopasowania, wazenia i oceny w warstwach wzorca).
  D. Pokrycie kandydackich "wspolnych paneli" na poziomie pacjenta (opcja modelu
     tylko na wartosciach zmierzonych, bez imputacji).
  E. Pacjenci NEURO wedlug okresu (opcje walidacji czasowej i wspolnego okna).
  F. Ilu ukrytych chorych trafia do treningowej puli U przy roznych udzialach
     ukrywania (opcje metod PU i estymacji class prior).
  G. Ktore cechy sa wyliczalne z innych (opcje zestawu cech i interpretacji)
     oraz czy przeliczenie PT -> INR / WPT zalezy od okresu (slad kalibracji).

Nic nie modyfikuje. Wynik: results/opcje_fakty.json.

Uruchomienie:
    python 4-pu-setup/opcje_fakty.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "aneurysm_sgkf_input.csv"
FOLDS_CSV = BASE_DIR / "3-sgkf-split" / "sgkf" / "results" / "pacjent_fold.csv"
OUTPUT_JSON = Path(__file__).parent / "results" / "opcje_fakty.json"

TARGET_COL = "label"
GROUP_COL = "patient_id"
DATE_COL = "examination_date"
META = [GROUP_COL, "custom_id", DATE_COL, TARGET_COL]
DEMO = ["patient_age", "patient_sex"]

# Wspolne okno dat kohort - to samo co w etapie 3 (1026_REALIZACJA_DO_SGKF.md, sekcja 6.1)
OKNO = (pd.Timestamp("2019-12-29"), pd.Timestamp("2021-12-26"))

# Bloki cech, ktorych braki wystepuja razem. Wyznaczone z samej macierzy brakow
# (identyczne lub prawie identyczne kolumny brakow, Jaccard >= 0,86 w bloku),
# bez etykiety. Rozni sie od grup w README: WBC i MPV brakuje razem z morfologia,
# a NEUT, NRBC i IG tworza osobny blok "rozmazu rozszerzonego".
BLOKI = {
    "morfologia": ["HGB", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW", "PLT", "WBC", "MPV"],
    "rozmaz_podstawowy": ["LYMPH", "%LYMPH", "MONO", "%MONO", "EO", "%EO", "BAZO", "%BAZO"],
    "rozmaz_rozszerzony": ["NEUT", "%NEUT", "NRBC", "%NRBC", "IG", "%IG"],
    "nerki": ["KREA", "eGFR-MDRD", "eGFRCKD"],
    "elektrolity": ["Na", "K"],
    "GLU": ["GLU"],
    "CRP": ["CRP"],
    "koagulologia": ["PT", "INR", "WPT", "APTT", "WAPTT"],
}

# Kandydackie wspolne panele (kumulatywnie) dla opcji "tylko wartosci zmierzone"
PANELE = [
    ("morfologia", ["morfologia"]),
    ("+ nerki i elektrolity", ["morfologia", "nerki", "elektrolity"]),
    ("+ rozmaz podstawowy", ["morfologia", "nerki", "elektrolity", "rozmaz_podstawowy"]),
    ("+ CRP", ["morfologia", "nerki", "elektrolity", "rozmaz_podstawowy", "CRP"]),
    ("+ GLU", ["morfologia", "nerki", "elektrolity", "rozmaz_podstawowy", "CRP", "GLU"]),
    ("wszystkie 36 cech laboratoryjnych", list(BLOKI)),
]

UDZIALY_UKRYWANIA = [0.2, 0.4, 0.6]


def naglowek(tekst: str) -> None:
    print()
    print("=" * 78)
    print(f"  {tekst}")
    print("=" * 78)


def wzorzec(maska: pd.DataFrame) -> pd.Series:
    """Ciag 0/1 dla kazdego wiersza: 1 = blok zmierzony."""
    return maska.astype(int).astype(str).agg("".join, axis=1)


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[DATE_COL])
    feat = [c for c in df.columns if c not in META]
    lab = [c for c in feat if c not in DEMO]
    assert sorted(sum(BLOKI.values(), [])) == sorted(lab), "bloki nie pokrywaja cech laboratoryjnych"
    print(f"{INPUT_CSV.name}: {len(df):,} rekordow, {df[GROUP_COL].nunique():,} pacjentow, "
          f"{len(feat)} cech ({len(lab)} laboratoryjnych)")
    kohorta = {0: "KOR", 1: "NEURO"}
    wynik = {"wejscie": INPUT_CSV.name, "bloki_brakow": BLOKI}

    etykieta = df.groupby(GROUP_COL)[TARGET_COL].max()

    # ---------------------------------------------------------------
    naglowek("A. Liczba rekordow na pacjenta")
    n_rek = df.groupby(GROUP_COL).size()
    a = {}
    for l, nazwa in kohorta.items():
        s = n_rek[etykieta == l]
        a[nazwa] = {
            "pacjenci": int(len(s)),
            "udzial_1_rekord": round(float((s == 1).mean()), 4),
            "udzial_2_3_rekordy": round(float(s.between(2, 3).mean()), 4),
            "udzial_4_i_wiecej": round(float((s >= 4).mean()), 4),
            "mediana": float(s.median()),
            "maksimum": int(s.max()),
        }
        print(f"  {nazwa:5s}: 1 rekord {a[nazwa]['udzial_1_rekord']:.1%}, 2-3 "
              f"{a[nazwa]['udzial_2_3_rekordy']:.1%}, >=4 {a[nazwa]['udzial_4_i_wiecej']:.1%}, "
              f"mediana {a[nazwa]['mediana']:.0f}, maks. {a[nazwa]['maksimum']}")
    wynik["A_rekordy_na_pacjenta"] = a

    # ---------------------------------------------------------------
    naglowek("B. Udzial brakujacych komorek cech laboratoryjnych wedlug reprezentacji")
    pierwszy = df.sort_values([DATE_COL, "custom_id"]).groupby(GROUP_COL).head(1)
    mediana = df.groupby(GROUP_COL)[lab].median()
    reprezentacje = {
        "wszystkie_rekordy": (df[lab], df[TARGET_COL]),
        "pierwszy_rekord_pacjenta": (pierwszy[lab], pierwszy[TARGET_COL]),
        "mediana_pacjenta_przed_imputacja": (mediana, etykieta.loc[mediana.index]),
    }
    b = {}
    for nazwa, (x, y) in reprezentacje.items():
        b[nazwa] = {kohorta[l]: round(float(x[y == l].isna().values.mean()), 4) for l in kohorta}
        print(f"  {nazwa:34s} KOR {b[nazwa]['KOR']:.1%}  NEURO {b[nazwa]['NEURO']:.1%}")
    wynik["B_braki_wedlug_reprezentacji"] = b

    # ---------------------------------------------------------------
    naglowek("C. Wzorce brakow: czy wzorce NEURO wystepuja w KOR")
    c = {}
    # C1: dokladny wzorzec 36 cech na rekordach
    w36 = wzorzec(df[lab].notna())
    licz36 = pd.crosstab(w36, df[TARGET_COL])
    w_neuro = w36[df[TARGET_COL] == 1]
    c["rekordy_dokladny_wzorzec_36"] = {
        "liczba_wzorcow": int(len(licz36)),
        "rekordy_NEURO_ze_wzorcem_nieobecnym_w_KOR": round(float((w_neuro.map(licz36[0]) == 0).mean()), 4),
    }
    print(f"  rekordy, dokladny wzorzec 36 cech: {len(licz36)} wzorcow; rekordy NEURO ze wzorcem "
          f"nieobecnym w KOR: {c['rekordy_dokladny_wzorzec_36']['rekordy_NEURO_ze_wzorcem_nieobecnym_w_KOR']:.1%}")

    # C2: wzorzec blokow (blok zmierzony = co najmniej jedna jego cecha zmierzona)
    def bloki_zmierzone(x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({k: x[v].notna().any(axis=1) for k, v in BLOKI.items()}, index=x.index)

    for poziom, (x, y) in {
        "rekordy": (df, df[TARGET_COL]),
        "pacjenci": (df.groupby(GROUP_COL)[lab].max(), etykieta),  # blok zmierzony w ktorymkolwiek rekordzie
    }.items():
        y = y.loc[x.index]
        wb = wzorzec(bloki_zmierzone(x))
        licz = pd.crosstab(wb, y)
        wb_neuro = wb[y == 1]
        n_kor_dla_neuro = wb_neuro.map(licz[0])
        # Wazenie KOR do rozkladu wzorcow NEURO: waga = p_NEURO(wzorzec) / p_KOR(wzorzec)
        p_neuro = licz[1] / licz[1].sum()
        p_kor = licz[0] / licz[0].sum()
        wspolne = licz[0] > 0
        waga = (p_neuro[wspolne] / p_kor[wspolne]).reindex(licz.index).fillna(0.0)
        w_kor = wb[y == 0].map(waga)
        ess = float(w_kor.sum() ** 2 / (w_kor ** 2).sum())
        top = licz.assign(KOR_udzial=p_kor, NEURO_udzial=p_neuro).sort_values(1, ascending=False).head(6)
        c[f"{poziom}_wzorzec_blokow"] = {
            "liczba_wzorcow": int(len(licz)),
            "NEURO_ze_wzorcem_w_co_najmniej_50_KOR": round(float((n_kor_dla_neuro >= 50).mean()), 4),
            "NEURO_ze_wzorcem_nieobecnym_w_KOR": round(float((n_kor_dla_neuro == 0).mean()), 4),
            "efektywna_liczebnosc_KOR_po_wazeniu": round(ess),
            "liczebnosc_KOR": int((y == 0).sum()),
            "najczestsze_wzorce_NEURO": {
                k: {"KOR": round(float(r["KOR_udzial"]), 4), "NEURO": round(float(r["NEURO_udzial"]), 4)}
                for k, r in top.iterrows()
            },
        }
        cc = c[f"{poziom}_wzorzec_blokow"]
        print(f"  {poziom}, wzorzec {len(BLOKI)} blokow: {cc['liczba_wzorcow']} wzorcow; NEURO ze wzorcem "
              f"obecnym u >= 50 KOR: {cc['NEURO_ze_wzorcem_w_co_najmniej_50_KOR']:.1%}; nieobecnym w KOR: "
              f"{cc['NEURO_ze_wzorcem_nieobecnym_w_KOR']:.1%}")
        print(f"    wazenie KOR do rozkladu wzorcow NEURO: efektywna liczebnosc "
              f"{cc['efektywna_liczebnosc_KOR_po_wazeniu']:,} z {cc['liczebnosc_KOR']:,}")
        print(f"    kolejnosc blokow we wzorcu: {', '.join(BLOKI)}")
        for k, r in cc["najczestsze_wzorce_NEURO"].items():
            print(f"    {k}: KOR {r['KOR']:.1%}  NEURO {r['NEURO']:.1%}")
    wynik["C_wzorce_brakow"] = c

    # ---------------------------------------------------------------
    naglowek("D. Pacjenci z kompletnym panelem (mediana pacjenta, bez imputacji)")
    d = {}
    for nazwa, bloki in PANELE:
        kol = sum((BLOKI[k] for k in bloki), [])
        kompletny = mediana[kol].notna().all(axis=1)
        e = etykieta.loc[mediana.index]
        d[nazwa] = {"liczba_cech": len(kol)} | {
            kohorta[l]: round(float(kompletny[e == l].mean()), 4) for l in kohorta}
        print(f"  {nazwa:36s} ({len(kol):2d} cech): KOR {d[nazwa]['KOR']:.1%}  NEURO {d[nazwa]['NEURO']:.1%}")
    wynik["D_kompletne_panele_pacjenci"] = d

    # ---------------------------------------------------------------
    naglowek("E. Pacjenci NEURO wedlug okresu badan")
    neuro = df[df[TARGET_COL] == 1]
    pierwsza = neuro.groupby(GROUP_COL)[DATE_COL].min()
    w_oknie = neuro[neuro[DATE_COL].between(*OKNO)].groupby(GROUP_COL).size()
    e = {
        "pierwszy_rekord_przed_oknem": int((pierwsza < OKNO[0]).sum()),
        "pierwszy_rekord_w_oknie": int(pierwsza.between(*OKNO).sum()),
        "pierwszy_rekord_po_oknie": int((pierwsza > OKNO[1]).sum()),
        "co_najmniej_jeden_rekord_w_oknie": int(len(w_oknie)),
        "okno": [str(OKNO[0].date()), str(OKNO[1].date())],
    }
    for k, v in e.items():
        print(f"  {k}: {v}")
    wynik["E_NEURO_wedlug_okresu"] = e

    # ---------------------------------------------------------------
    naglowek("F. Ukryci chorzy w treningowej puli U (wartosci oczekiwane, srednio na fold)")
    foldy = pd.read_csv(FOLDS_CSV)
    f = {}
    for h in UDZIALY_UKRYWANIA:
        wiersze = []
        for k in sorted(foldy["fold"].unique()):
            tr = foldy[foldy["fold"] != k]
            n_p = int((tr["label"] == 1).sum())
            n_u = int((tr["label"] == 0).sum())
            ukryci = h * n_p
            wiersze.append((n_p, n_u, ukryci, ukryci / (n_u + ukryci)))
        n_p, n_u, ukryci, udzial = np.mean(wiersze, axis=0)
        f[f"{h:.0%}"] = {"P_w_treningu": round(n_p), "KOR_w_treningu": round(n_u),
                         "ukryci_w_treningu": round(ukryci), "widoczni_P": round(n_p - ukryci),
                         "udzial_ukrytych_w_puli_U": round(float(udzial), 4)}
        print(f"  ukrywanie {h:.0%}: widoczni P {n_p - ukryci:,.0f}, ukryci w puli U {ukryci:,.0f} "
              f"wsrod {n_u + ukryci:,.0f} (udzial {udzial:.2%})")
    wynik["F_ukryci_w_treningu"] = f

    # ---------------------------------------------------------------
    naglowek("G. Cechy wyliczalne z innych cech i slad kalibracji PT -> INR")

    def blad_wzgledny(wyliczona: pd.Series, zapisana: pd.Series) -> dict:
        m = wyliczona.notna() & zapisana.notna() & (zapisana != 0)
        b = (wyliczona[m] - zapisana[m]).abs() / zapisana[m].abs()
        return {"mediana_bledu_wzglednego": round(float(b.median()), 4), "rekordy": int(m.sum())}

    g = {"wzory": {}}
    wzory = {
        "MCV = 10 * HCT / RBC": (10 * df["HCT"] / df["RBC"], df["MCV"]),
        "MCH = 10 * HGB / RBC": (10 * df["HGB"] / df["RBC"], df["MCH"]),
        "MCHC = 100 * HGB / HCT": (100 * df["HGB"] / df["HCT"], df["MCHC"]),
    }
    for x in ["NEUT", "LYMPH", "MONO", "EO", "BAZO", "IG", "NRBC"]:
        wzory[f"{x} = WBC * %{x} / 100"] = (df["WBC"] * df[f"%{x}"] / 100, df[x])
    # eGFR-MDRD (wersja 175, IDMS) z KREA, wieku i plci; plec 1 sprawdzana jako kobieta.
    # Porownanie tylko ponizej progu cenzurowania (zapisane wartosci koncza sie na 60).
    ponizej = df["eGFR-MDRD"] < 59
    mdrd = 175 * df["KREA"] ** -1.154 * df["patient_age"] ** -0.203 * np.where(df["patient_sex"] == 1, 0.742, 1.0)
    wzory["eGFR-MDRD = MDRD(KREA, wiek, plec), ponizej 59"] = (mdrd.where(ponizej), df["eGFR-MDRD"].where(ponizej))
    for nazwa, (wyl, zap) in wzory.items():
        g["wzory"][nazwa] = blad_wzgledny(wyl, zap)
        print(f"  {nazwa:48s} mediana bledu wzglednego {g['wzory'][nazwa]['mediana_bledu_wzglednego']:.2%}")
    suma = df[["%NEUT", "%LYMPH", "%MONO", "%EO", "%BAZO"]].sum(axis=1, min_count=5)
    g["suma_5_procentow_rozmazu"] = {"srednia": round(float(suma.mean()), 3), "odch_std": round(float(suma.std()), 3)}
    iloraz = (df["APTT"] / df["WAPTT"]).dropna()
    g["APTT_przez_WAPTT"] = {"mediana": round(float(iloraz.median()), 2),
                             "kwantyle_5_95": [round(float(iloraz.quantile(q)), 2) for q in (0.05, 0.95)]}
    g["cenzurowanie_eGFR"] = {
        "eGFR-MDRD_rowne_60": round(float((df["eGFR-MDRD"].dropna() == 60).mean()), 4),
        "eGFRCKD_rowne_90": round(float((df["eGFRCKD"].dropna() == 90).mean()), 4),
    }
    print(f"  suma 5 procentow rozmazu: srednia {g['suma_5_procentow_rozmazu']['srednia']}, "
          f"odch. std. {g['suma_5_procentow_rozmazu']['odch_std']}")
    print(f"  APTT / WAPTT: mediana {g['APTT_przez_WAPTT']['mediana']}, 5-95%: {g['APTT_przez_WAPTT']['kwantyle_5_95']}")
    print(f"  cenzurowanie: eGFR-MDRD = 60 w {g['cenzurowanie_eGFR']['eGFR-MDRD_rowne_60']:.1%} zmierzonych, "
          f"eGFRCKD = 90 w {g['cenzurowanie_eGFR']['eGFRCKD_rowne_90']:.1%}")

    # Przy stalym PT mediana INR i WPT powinna byc taka sama, jesli przeliczenie (odczynnik, ISI)
    # sie nie zmienia. Grupy jak w kontroli epoki: KOR (okno) i NEURO w oknie / przed / po.
    k = df.dropna(subset=["PT", "INR", "WPT"])
    grupa = np.select(
        [k[TARGET_COL] == 0, k[DATE_COL] < OKNO[0], k[DATE_COL] > OKNO[1]],
        ["KOR", "NEURO przed oknem", "NEURO po oknie"], "NEURO w oknie")
    pasmo = k["PT"].between(11.5, 12.5)
    g["INR_i_WPT_przy_PT_11.5_12.5"] = {}
    for nazwa in ["KOR", "NEURO w oknie", "NEURO przed oknem", "NEURO po oknie"]:
        x = k[pasmo & (grupa == nazwa)]
        g["INR_i_WPT_przy_PT_11.5_12.5"][nazwa] = {
            "rekordy": int(len(x)), "mediana_INR": round(float(x["INR"].median()), 3),
            "mediana_WPT": round(float(x["WPT"].median()), 1)}
        print(f"  PT 11,5-12,5 s, {nazwa:18s} (n={len(x):5d}): mediana INR {x['INR'].median():.2f}, "
              f"mediana WPT {x['WPT'].median():.0f}")
    wynik["G_cechy_wyliczalne"] = g

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(wynik, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano {OUTPUT_JSON.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
