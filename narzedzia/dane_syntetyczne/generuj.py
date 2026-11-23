"""
generuj.py
==========

Sztuczne (syntetyczne) dane o tej samej budowie co prawdziwe, do sprawdzania
kodu bez dostępu do danych pacjentów. Generator czyta wyłącznie zbiorcze
statystyki z parametry.json (zob. wyciagnij_parametry.py) i ziarno losowania.

Powstają trzy poziomy, w takim samym układzie folderów jak data/:

  <wyjscie>/raw/kor_merged_aggregated_1W_mean.csv       dane surowe (128 kolumn,
  <wyjscie>/raw/neuro_merged_aggregated_1W_mean.csv      słowniki {'values': [...]}, jednostki, normy, nazwy badań)
  <wyjscie>/imputation-inputs/aneurysm_concatted.csv    po UPROSZCZONYM etapie 1 (42 kolumny)
  <wyjscie>/processed/aneurysm_sgkf_input.csv           po prawdziwym przygotuj_dane.py (wejście SGKF)
  <wyjscie>/OPIS.json                                   tryb, ziarno, liczby i "klucz odpowiedzi"

Tryby (--tryb):
  realny     każda kohorta ma własne rozkłady wartości, braków, dat i liczby badań,
             więc są te same skróty co w prawdziwych danych (domyślny);
  brak       etykieta jest niezależna od wszystkiego: dobry model musi dać wynik
             losowy (np. RecallHidden@5% około 5%);
  wbudowany  jak "brak", ale chorzy mają znane przesunięcie kilku cech
             (SYGNAL poniżej): kod oceny powinien je odnaleźć.

Uproszczony etap 1 NIE odtwarza notebooków z 1-data-preparation/: zamienia
słowniki na średnie, usuwa kolumny tekstowe, jednostki i normy, filtruje wiek
18–100, usuwa stałą listę 4 cech (ALT, AST, BUN, P-LCR), pacjentów KOR, którym
brakuje ponad 60% cech, przycina historię do 11 (KOR) / 38 (NEURO) ostatnich
rekordów i usuwa powtórzone custom_id. Wartości odstających nie usuwa.

Celowo zasiane przypadki (opisane w OPIS.json):
  - pacjenci obecni w obu kohortach (ten sam patient_id),
  - wartości niemożliwe z reguł przygotuj_dane.py (KREA > 50, KREA > 10 przy eGFR >= 30, K > 15, Na < 80),
  - kolizja custom_id (2017-01-01 i 2017-12-31 dają ten sam klucz rok + tydzień ISO),
  - wiek poza 18–100 (w tym 222 lata), pojedyncze braki wieku i płci,
  - wpisy nieliczbowe ('<0.4', '>160' itp.) i pusta kolumna diagnosis_id.
Wartości ujemnych NIE ma (prawdziwe dane też ich nie mają).

Uruchomienie:
    python narzedzia/dane_syntetyczne/generuj.py                       # pełny rozmiar, tryb realny
    python narzedzia/dane_syntetyczne/generuj.py --skala 0.1           # mały zbiór do szybkich testów
    python narzedzia/dane_syntetyczne/generuj.py --tryb brak --wyjscie data_syntetyczne_brak
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parents[2]
PARAMETRY = Path(__file__).resolve().parent / "parametry.json"
DOMYSLNE_WYJSCIE = BASE_DIR / "data_syntetyczne"

KOHORTY = ("kor", "neuro")
ETYKIETA = {"kor": 0, "neuro": 1}
PLIK_SUROWY = {"kor": "kor_merged_aggregated_1W_mean.csv", "neuro": "neuro_merged_aggregated_1W_mean.csv"}

# Kolejność kolumn plików wynikowych etapu 1 (taka sama jak w prawdziwych plikach).
KOLUMNY_CONCATTED = ["HGB", "PT", "MCV", "GLU", "%IG", "CRP", "MONO", "patient_age", "EO", "%EO", "APTT",
                     "%BAZO", "MPV", "RBC", "WBC", "BAZO", "patient_id", "IG", "patient_sex", "KREA",
                     "eGFRCKD", "examination_date", "PLT", "HCT", "%NRBC", "NRBC", "MCHC", "Na", "WPT",
                     "%NEUT", "eGFR-MDRD", "MCH", "RDW", "K", "NEUT", "LYMPH", "INR", "%LYMPH", "WAPTT",
                     "custom_id", "%MONO", "label"]
CECHY_USUWANE_ETAP1 = ["ALT", "AST", "BUN", "P-LCR"]
LIMIT_HISTORII = {"kor": 11, "neuro": 38}
PROG_BRAKOW_PACJENTA = 0.60

# Tryb "wbudowany": przesunięcie chorych w skali wyniku normalnego (odchylenia standardowe).
SYGNAL = {"WBC": 0.35, "NEUT": 0.35, "%NEUT": 0.30, "%LYMPH": -0.30, "GLU": 0.30, "CRP": 0.25}

# Liczba cyfr po przecinku pojedynczego pomiaru.
CYFRY = {"MCH": 1, "HGB": 1, "MCV": 1, "RBC": 2, "MCHC": 1, "PLT": 0, "HCT": 1, "WBC": 2, "RDW": 1,
         "%BAZO": 1, "BAZO": 2, "EO": 2, "%EO": 1, "LYMPH": 2, "%LYMPH": 1, "%MONO": 1, "MONO": 2,
         "MPV": 1, "KREA": 2, "Na": 0, "K": 1, "%IG": 1, "IG": 2, "eGFR-MDRD": 0, "eGFRCKD": 0,
         "NEUT": 2, "%NEUT": 1, "%NRBC": 1, "NRBC": 2, "WPT": 0, "GLU": 0, "INR": 2, "PT": 1, "CRP": 2,
         "WAPTT": 2, "APTT": 1, "ALT": 0, "BUN": 0, "AST": 0, "P-LCR": 1}

# Przybliżone zakresy referencyjne (do kolumn *-norm: -1 poniżej, 0 w normie, 1 powyżej).
NORMY = {"MCH": (27, 34), "HGB": (12, 16), "MCV": (80, 99), "RBC": (4.0, 5.5), "MCHC": (32, 36),
         "PLT": (150, 400), "HCT": (37, 47), "WBC": (4, 10), "RDW": (11.5, 14.5), "%BAZO": (0, 1),
         "BAZO": (0, 0.1), "EO": (0.02, 0.5), "%EO": (1, 5), "LYMPH": (1, 4.5), "%LYMPH": (20, 45),
         "%MONO": (2, 10), "MONO": (0.2, 0.95), "MPV": (7.5, 11.5), "KREA": (0.6, 1.2), "Na": (136, 145),
         "K": (3.5, 5.1), "%IG": (0, 0.5), "IG": (0, 0.03), "eGFR-MDRD": (60, 1000), "eGFRCKD": (60, 1000),
         "NEUT": (1.8, 7.7), "%NEUT": (40, 70), "%NRBC": (0, 0.2), "NRBC": (0, 0.01), "WPT": (80, 120),
         "GLU": (70, 99), "INR": (0.8, 1.2), "PT": (9.4, 12.5), "CRP": (0, 5), "WAPTT": (0.8, 1.2),
         "APTT": (26, 36), "ALT": (0, 35), "BUN": (7, 20), "AST": (0, 35), "P-LCR": (13, 43)}

# Która grupa nazw badań odpowiada których cechom (nazwa trafia do rekordu, gdy jest choć jedna cecha).
CECHY_GRUPY_NAZW = {
    "morfologia": ["MCH", "HGB", "MCV", "RBC", "MCHC", "PLT", "HCT", "WBC", "RDW", "%BAZO", "BAZO", "EO",
                   "%EO", "LYMPH", "%LYMPH", "%MONO", "MONO", "MPV", "%IG", "IG", "NEUT", "%NEUT",
                   "%NRBC", "NRBC", "P-LCR"],
    "kreatynina": ["KREA"], "mocznik": ["BUN"], "crp": ["CRP"], "elektrolity": ["Na", "K"],
    "glukoza": ["GLU"], "egfr_mdrd": ["eGFR-MDRD"], "egfr_ckd": ["eGFRCKD"], "pt": ["PT"],
    "wpt": ["WPT"], "inr": ["INR"], "aptt": ["APTT"], "waptt": ["WAPTT"], "alt": ["ALT"], "ast": ["AST"],
}

OPISY = ["Hemoliza próbki.", "Próbka lipemiczna.", "Wynik potwierdzony powtórnym oznaczeniem.",
         "Skrzep w próbce.", "Hemoliza próbki. Wynik potasu może być zawyżony."]

# Dodatkowo zasiane wartości niemożliwe (na 100 000 rekordów) i udział KOR w trybie realnym.
# Wysokie KREA pojawiają się też same, bo górne kwantyle prawdziwej kreatyniny zawierają
# wartości w złych jednostkach; zasiew gwarantuje, że każda reguła ma przypadki.
NIEMOZLIWE = {"krea_jednostka": (30, 0.93), "krea_egfr": (30, 0.93), "k": (30, 1.0), "na": (2, 0.0)}


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def macierz_dodatnia(k: np.ndarray) -> np.ndarray:
    """Najbliższa macierz korelacji dodatnio określona (przycięcie wartości własnych)."""
    k = (np.asarray(k) + np.asarray(k).T) / 2
    w, v = np.linalg.eigh(k)
    w = np.clip(w, 1e-3, None)
    with np.errstate(all="ignore"):   # BLAS na macOS zgłasza fałszywe ostrzeżenia przy matmul
        k = (v * w) @ v.T
    d = np.sqrt(np.diag(k))
    return k / np.outer(d, d)


def z_kwantyli(u: np.ndarray, poziomy: np.ndarray, wartosci: list) -> np.ndarray:
    """Odwrotna dystrybuanta z zapisanych kwantyli (interpolacja liniowa, obcięta do 0,1–99,9%)."""
    if not wartosci:
        return np.full(len(u), np.nan)
    return np.interp(u, poziomy, np.asarray(wartosci, dtype=float))


def losuj_z_histogramu(rng, slownik: dict, n: int) -> np.ndarray:
    klucze = list(slownik.keys())
    p = np.asarray(list(slownik.values()), dtype=float)
    return np.asarray(klucze)[rng.choice(len(klucze), size=n, p=p / p.sum())]


def niedziela(d: date) -> date:
    return d + timedelta(days=(6 - d.weekday()) % 7)


def custom_id(pid: int, d: date) -> str:
    return f"{pid}-{d.year}-W{d.isocalendar()[1]}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self, parametry: dict, tryb: str, skala: float, ziarno: int):
        self.p = parametry
        self.tryb = tryb
        self.skala = skala
        self.rng = np.random.default_rng(ziarno)
        self.cechy = parametry["cechy_lab"]
        self.poziomy = np.asarray(parametry["poziomy_kwantyli"])
        self.klucz = {"zasiane": {}}
        self.chol = {k: np.linalg.cholesky(macierz_dodatnia(parametry["kohorty"][k]["korelacja"]))
                     for k in KOHORTY}
        self.chol["wspolne"] = np.linalg.cholesky(macierz_dodatnia(parametry["wspolne"]["korelacja"]))

    # --- pacjenci -----------------------------------------------------------

    def pacjenci(self) -> pd.DataFrame:
        rng, P = self.rng, self.p["kohorty"]
        n = {k: max(20, int(round(P[k]["n_pacjentow"] * self.skala))) for k in KOHORTY}
        max_id = max(P[k]["max_patient_id"] for k in KOHORTY)
        wszystkie_id = rng.choice(np.arange(1, int(max_id * 1.1)), size=n["kor"] + n["neuro"], replace=False)
        tab = pd.DataFrame({
            "patient_id": wszystkie_id,
            "kohorta": ["kor"] * n["kor"] + ["neuro"] * n["neuro"],
        })
        # pacjenci obecni w obu kohortach: część pacjentów NEURO dostaje id pacjenta KOR
        n_wsp = max(1, int(round(self.p["pacjenci_w_obu_kohortach"] * self.skala)))
        neuro_idx = tab.index[tab["kohorta"] == "neuro"]
        kor_ids = tab.loc[tab["kohorta"] == "kor", "patient_id"].to_numpy()
        wybrani = rng.choice(neuro_idx, size=min(n_wsp, len(neuro_idx)), replace=False)
        tab.loc[wybrani, "patient_id"] = rng.choice(kor_ids, size=len(wybrani), replace=False)
        self.klucz["zasiane"]["pacjenci_w_obu_kohortach"] = sorted(int(x) for x in tab.loc[wybrani, "patient_id"])

        # profil źródłowy: skąd biorą się rozkłady (w trybie realnym = własna kohorta)
        if self.tryb == "realny":
            tab["zrodlo"] = tab["kohorta"]
        else:
            udz = P["neuro"]["n_pacjentow"] / (P["neuro"]["n_pacjentow"] + P["kor"]["n_pacjentow"])
            tab["zrodlo"] = np.where(rng.random(len(tab)) < udz, "neuro", "kor")

        # liczba rekordów
        n_rek = np.zeros(len(tab), dtype=int)
        for k in KOHORTY:
            m = (tab["zrodlo"] == k).to_numpy()
            hist = {int(a): b for a, b in P[k]["rek_na_pacjenta_hist"].items()}
            ogon = P[k]["rek_na_pacjenta_ogon"]
            n_rek[m] = losuj_z_histogramu(rng, hist, m.sum()).astype(int)
            if ogon["n"]:
                w_ogonie = m & (rng.random(len(tab)) < ogon["n"] / P[k]["n_pacjentow"])
                n_rek[w_ogonie] = rng.choice(ogon["kwantyle"], size=w_ogonie.sum())
        tab["n_rek"] = n_rek

        # wiek przy pierwszym badaniu i płeć
        wiek = np.zeros(len(tab))
        plec = np.zeros(len(tab), dtype=int)
        for k in KOHORTY:
            m = (tab["zrodlo"] == k).to_numpy()
            wiek[m] = z_kwantyli(rng.uniform(0.001, 0.999, m.sum()), self.poziomy, P[k]["wiek_pierwszy_kwantyle"])
            plec[m] = (rng.random(m.sum()) < P[k]["plec_odsetek_1"]).astype(int)
        tab["wiek0"] = wiek
        tab["plec"] = plec
        return tab

    # --- rekordy: daty, wzorce braków ---------------------------------------

    def rekordy(self, pac: pd.DataFrame) -> pd.DataFrame:
        rng, P = self.rng, self.p["kohorty"]
        wiersze = []
        zakres = {}
        for k in KOHORTY:
            miesiace = list(P[k]["pierwsze_badanie_miesiac"].keys())
            pierwszy = datetime.strptime(miesiace[0], "%Y-%m").date()
            ostatni = datetime.strptime(miesiace[-1], "%Y-%m").date()
            ostatni = (ostatni.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            zakres[k] = (pierwszy, ostatni)
        for r in pac.itertuples(index=False):
            P_k = P[r.zrodlo]
            mies = losuj_z_histogramu(rng, P_k["pierwsze_badanie_miesiac"], 1)[0]
            start = datetime.strptime(mies, "%Y-%m").date() + timedelta(days=int(rng.integers(0, 28)))
            odstepy = np.maximum(1, np.round(z_kwantyli(rng.uniform(0.001, 0.999, r.n_rek - 1),
                                                         self.poziomy, P_k["odstep_tygodnie_kwantyle"])))
            tyg = np.concatenate([[0], np.cumsum(odstepy)]).astype(int)
            daty = [niedziela(start + timedelta(weeks=int(t))) for t in tyg]
            koniec = zakres[r.zrodlo][1]
            if daty[-1] > koniec:                       # przesuń całą historię, żeby zmieściła się w okresie
                przesun = (daty[-1] - koniec).days // 7 + 1
                daty = [d - timedelta(weeks=przesun) for d in daty]
                daty = [d for d in daty if d >= zakres[r.zrodlo][0]] or [daty[-1]]
            for d in daty:
                wiersze.append((r.patient_id, r.kohorta, r.zrodlo, d, r.wiek0 + (d - daty[0]).days / 365.25, r.plec))
        rek = pd.DataFrame(wiersze, columns=["patient_id", "kohorta", "zrodlo", "data", "wiek", "plec"])

        # kolizja custom_id: jeden pacjent NEURO z badaniami 2017-01-01 i 2017-12-31
        kandydaci = rek[rek["kohorta"] == "neuro"]["patient_id"].unique()
        pid = int(rng.choice(kandydaci))
        baza = rek[rek["patient_id"] == pid].iloc[0]
        dodatkowe = pd.DataFrame([{**baza.to_dict(), "data": d,
                                   "wiek": baza["wiek"] + (d - baza["data"]).days / 365.25}
                                  for d in (date(2017, 1, 1), date(2017, 12, 31))])
        rek = pd.concat([rek, dodatkowe], ignore_index=True)
        self.klucz["zasiane"]["kolizja_custom_id"] = {"patient_id": pid, "custom_id": f"{pid}-2017-W52",
                                                      "daty": ["2017-01-01", "2017-12-31"]}
        rek = rek.drop_duplicates(["patient_id", "kohorta", "data"]).sort_values(["kohorta", "patient_id", "data"])
        return rek.reset_index(drop=True)

    def wzorce(self, rek: pd.DataFrame) -> np.ndarray:
        """Macierz obecności (rekordy x 40 cech) z częstych wzorców braków + rzadkie odmiany."""
        rng, P = self.rng, self.p["kohorty"]
        obecne = np.zeros((len(rek), len(self.cechy)), dtype=bool)
        for k in KOHORTY:
            m = (rek["zrodlo"] == k).to_numpy()
            wz = P[k]["wzorce_brakow"]
            wybrane = losuj_z_histogramu(rng, wz, m.sum())
            macierz = np.array([[c == "1" for c in w] for w in wybrane], dtype=bool)
            rzadkie = rng.random(m.sum()) > P[k]["wzorce_pokrycie"]
            obecnosc = np.array([P[k]["cechy"][c]["odsetek_obecnych"] for c in self.cechy])
            for i in np.flatnonzero(rzadkie):          # rzadki wzorzec: częsty z kilkoma cechami wylosowanymi na nowo
                zmiany = rng.choice(len(self.cechy), size=1 + rng.poisson(3), replace=False)
                macierz[i, zmiany] = rng.random(len(zmiany)) < obecnosc[zmiany]
            obecne[m] = macierz
        return obecne

    # --- wartości -----------------------------------------------------------

    def wartosci(self, rek: pd.DataFrame) -> np.ndarray:
        rng, P = self.rng, self.p["kohorty"]
        nc = len(self.cechy)
        wynik = np.zeros((len(rek), nc))
        pacjenci, odwr = np.unique(rek["patient_id"].astype(str) + "_" + rek["kohorta"], return_inverse=True)
        for k in KOHORTY:
            m = (rek["zrodlo"] == k).to_numpy()
            chol = self.chol[k] if self.tryb == "realny" else self.chol["wspolne"]
            icc = np.array([P[k]["udzial_miedzy_pacjentami"][c] for c in self.cechy])
            with np.errstate(all="ignore"):   # BLAS na macOS zgłasza fałszywe ostrzeżenia przy matmul
                u = rng.standard_normal((len(pacjenci), nc)) @ chol.T  # efekt pacjenta
                e = rng.standard_normal((m.sum(), nc)) @ chol.T        # zmienność rekord do rekordu
            assert np.isfinite(u).all() and np.isfinite(e).all()
            z = np.sqrt(icc) * u[odwr[m]] + np.sqrt(1 - icc) * e
            if self.tryb == "wbudowany":
                chory = (rek.loc[m, "kohorta"] == "neuro").to_numpy()
                for c, s in SYGNAL.items():
                    z[chory, self.cechy.index(c)] += s
            uu = np.clip(norm.cdf(z), self.poziomy[0], self.poziomy[-1])
            for j, c in enumerate(self.cechy):
                kw = P[k]["cechy"][c]["kwantyle"] if self.tryb == "realny" else self.p["wspolne"]["kwantyle"][c]
                wynik[m, j] = z_kwantyli(uu[:, j], self.poziomy, kw)
        if self.tryb == "wbudowany":
            self.klucz["sygnal_wbudowany"] = {"cechy": SYGNAL, "jednostka": "odchylenia standardowe wyniku normalnego",
                                              "kogo_dotyczy": "wszyscy pacjenci NEURO (label = 1)"}
        return wynik

    def zasiej_niemozliwe(self, rek: pd.DataFrame, wart: np.ndarray, obecne: np.ndarray) -> None:
        rng = self.rng
        j = {c: self.cechy.index(c) for c in ("KREA", "eGFR-MDRD", "K", "Na")}
        zasiane = {}
        for regula, (na_100k, udzial_kor) in NIEMOZLIWE.items():
            n = max(1, int(round(na_100k * len(rek) / 100_000)))
            if self.tryb == "realny":
                n_kor = int(round(n * udzial_kor))
                kandydaci = [np.flatnonzero((rek["kohorta"] == "kor").to_numpy()),
                             np.flatnonzero((rek["kohorta"] == "neuro").to_numpy())]
                idx = np.concatenate([rng.choice(kandydaci[0], n_kor, replace=False),
                                      rng.choice(kandydaci[1], n - n_kor, replace=False)])
            else:
                idx = rng.choice(len(rek), n, replace=False)
            if regula == "krea_jednostka":
                wart[idx, j["KREA"]] = rng.uniform(55, 160, n).round(1)
                obecne[idx, j["KREA"]] = True
            elif regula == "krea_egfr":
                wart[idx, j["KREA"]] = rng.uniform(11, 45, n).round(1)
                obecne[idx, j["KREA"]] = True
                wart[idx, j["eGFR-MDRD"]] = np.maximum(wart[idx, j["eGFR-MDRD"]], 30)
                obecne[idx, j["eGFR-MDRD"]] = True
            elif regula == "k":
                wart[idx, j["K"]] = rng.uniform(16, 39, n).round(1)
                obecne[idx, j["K"]] = True
            elif regula == "na":
                wart[idx, j["Na"]] = rng.uniform(20, 79, n).round(0)
                obecne[idx, j["Na"]] = True
            zasiane[regula] = int(n)
        self.klucz["zasiane"]["wartosci_niemozliwe_dodatkowo"] = zasiane

    # --- zapis surowy ---------------------------------------------------------

    def nazwy_badan(self, kohorta_zrodla: str, rok: int, obecne_wiersz: np.ndarray, dl: int) -> str:
        P = self.p["kohorty"][kohorta_zrodla]
        wg_roku = P["nazwy_wg_roku"].get(str(rok)) or {}
        nazwy = []
        for grupa, cechy in CECHY_GRUPY_NAZW.items():
            if not any(obecne_wiersz[self.cechy.index(c)] for c in cechy):
                continue
            warianty = self.p["grupy_nazw"][grupa]
            wagi = np.array([wg_roku.get(w, 0) for w in warianty], dtype=float)
            if wagi.sum() == 0:
                wagi = np.array([sum(r.get(w, 0) for r in P["nazwy_wg_roku"].values()) for w in warianty], float)
            if wagi.sum() == 0:
                wagi = np.ones(len(warianty))
            nazwy.append(warianty[self.rng.choice(len(warianty), p=wagi / wagi.sum())])
        for n, odsetek in P["nazwy_dodatkowe_odsetek"].items():
            if self.rng.random() < odsetek:
                nazwy.append(n)
        lista = nazwy * dl if nazwy else ["Morfologia"]
        return str({"values": lista})

    def zapisz_surowe(self, rek: pd.DataFrame, wart: np.ndarray, obecne: np.ndarray, katalog: Path) -> dict:
        rng, P = self.rng, self.p["kohorty"]
        katalog.mkdir(parents=True, exist_ok=True)
        pliki = {}
        for k in KOHORTY:
            m = np.flatnonzero((rek["kohorta"] == k).to_numpy())
            r = rek.iloc[m].reset_index(drop=True)
            W, O = wart[m], obecne[m]
            fmt = P[k]["cechy"]                          # format zapisu to cecha pliku (kohorty)
            dl_rek = losuj_z_histogramu(rng, {int(a): b for a, b in P[k]["dlugosc_listy"].items()}, len(r)).astype(int)
            kol = {
                "custom_id": [custom_id(int(p), d) for p, d in zip(r["patient_id"], r["data"])],
                "patient_id": r["patient_id"].astype(int).astype(str).tolist(),
                "examination_date": [d.isoformat() for d in r["data"]],
                "patient_age": [f"{w:.2f}" for w in r["wiek"]],
                "patient_sex": [(f"{p}.0" if k == "kor" else str(p)) for p in r["plec"]],
                "diagnosis_id": [""] * len(r),
                "descriptive_result": [str({"values": [OPISY[rng.integers(len(OPISY))]]})
                                       if rng.random() < P[k]["odsetek_opisow"] else "" for _ in range(len(r))],
                "examination_type": [self.nazwy_badan(z, d.year, O[i], dl_rek[i])
                                     for i, (z, d) in enumerate(zip(r["zrodlo"], r["data"]))],
            }
            for j, c in enumerate(self.cechy):
                v_kol, j_kol, n_kol = [], [], []
                slownik = fmt[c]["odsetek_slownik"] >= 0.5
                tokeny = fmt[c].get("nieliczbowe_odsetek", {})
                lo, hi = NORMY[c]
                for i in range(len(r)):
                    if not O[i, j]:
                        v_kol.append(""); j_kol.append(""); n_kol.append("")
                        continue
                    ile = dl_rek[i] if rng.random() < 0.7 else int(rng.integers(1, dl_rek[i] + 1))
                    pomiary = np.round(W[i, j] * (1 + rng.normal(0, 0.03, ile)), CYFRY[c])
                    pomiary = np.maximum(pomiary, 0.0)
                    teksty = [str(float(x)) for x in pomiary]
                    for t, o in tokeny.items():          # wpisy nieliczbowe, np. '<0.4'
                        if rng.random() < o:
                            teksty[int(rng.integers(len(teksty)))] = t
                    if slownik:
                        v_kol.append(str({"values": teksty}))
                    else:
                        liczby = [float(t) for t in teksty if not t[0] in "<>"]
                        v_kol.append(repr(float(np.mean(liczby))) if liczby else teksty[0])
                    srednia = float(np.mean(pomiary))
                    j_kol.append(fmt[c]["jednostka"])
                    n_kol.append("-1.0" if srednia < lo else ("1.0" if srednia > hi else "0.0"))
                kol[c], kol[f"{c}-unit"], kol[f"{c}-norm"] = v_kol, j_kol, n_kol
            df = pd.DataFrame(kol)

            # wiek poza 18–100, pojedyncze braki wieku i płci
            n_wiek = max(1, int(round(P[k]["wiek_poza_18_100"] * self.skala)))
            idx = rng.choice(len(df), n_wiek, replace=False)
            df.loc[idx, "patient_age"] = [f"{w:.2f}" for w in rng.uniform(15, 17.9, n_wiek)]
            df.loc[idx[0], "patient_age"] = "222.00"
            braki = {}
            for kol_, ile in (("patient_age", P[k]["braki_wiek"]), ("patient_sex", P[k]["braki_plec"])):
                if ile:
                    ii = rng.choice(len(df), max(1, int(round(ile * self.skala))), replace=False)
                    df.loc[ii, kol_] = ""
                    braki[kol_] = int(len(ii))
            self.klucz["zasiane"][f"wiek_poza_18_100_{k}"] = int(n_wiek)
            self.klucz["zasiane"][f"braki_metadanych_{k}"] = braki

            kolejnosc = ["custom_id", "patient_id", "examination_date", "patient_age", "patient_sex",
                         "diagnosis_id", "descriptive_result", "examination_type"]
            for c in self.cechy:
                kolejnosc += [c, f"{c}-unit", f"{c}-norm"]
            sciezka = katalog / PLIK_SUROWY[k]
            df[kolejnosc].to_csv(sciezka, index=False, encoding="utf-8-sig", lineterminator="\n")
            pliki[k] = df[kolejnosc]
        return pliki


# ---------------------------------------------------------------------------
# Uproszczony etap 1 i etap 3A
# ---------------------------------------------------------------------------

def _srednia(v: str) -> float:
    if not isinstance(v, str) or not v:
        return np.nan
    if v.startswith("{"):
        liczby = []
        for x in eval(v)["values"]:  # noqa: S307 - format zapisany przez ten generator
            try:
                liczby.append(float(x))
            except ValueError:
                pass
        return float(np.mean(liczby)) if liczby else np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def etap1_uproszczony(surowe: dict, cechy: list) -> tuple[pd.DataFrame, dict]:
    czesci, licz = [], {}
    for k, df in surowe.items():
        d = pd.DataFrame({
            "patient_id": df["patient_id"].astype(int),
            "custom_id": df["custom_id"],
            "examination_date": df["examination_date"],
            "patient_age": pd.to_numeric(df["patient_age"], errors="coerce"),
            "patient_sex": pd.to_numeric(df["patient_sex"], errors="coerce"),
        })
        for c in cechy:
            d[c] = df[c].map(_srednia)
        n0 = len(d)
        d = d.dropna(subset=["patient_age", "patient_sex"])
        d = d[(d["patient_age"] >= 18) & (d["patient_age"] <= 100)]
        n_wiek = n0 - len(d)
        d = d.drop(columns=CECHY_USUWANE_ETAP1)
        cechy38 = [c for c in cechy if c not in CECHY_USUWANE_ETAP1]
        n_pac_usun = 0
        if k == "kor":   # pacjenci KOR, którym brakuje ponad 60% cech (w całej historii)
            zmierzone = d.groupby("patient_id")[cechy38].apply(lambda g: g.notna().any().mean())
            zli = set(zmierzone[zmierzone < 1 - PROG_BRAKOW_PACJENTA].index)
            n_pac_usun = len(zli)
            d = d[~d["patient_id"].isin(zli)]
        d = d.sort_values(["patient_id", "examination_date"])
        d = d.groupby("patient_id").tail(LIMIT_HISTORII[k])
        n_przed = len(d)
        d = d.drop_duplicates("custom_id", keep="last")
        d["patient_sex"] = d["patient_sex"].astype(int)
        d["label"] = ETYKIETA[k]
        czesci.append(d)
        licz[k] = {"rekordy_surowe": int(n0), "usuniete_wiek_lub_braki": int(n_wiek),
                   "pacjenci_kor_ponad_60proc_brakow": int(n_pac_usun),
                   "usuniete_powtorzone_custom_id": int(n_przed - len(d)), "rekordy_po": int(len(d))}
    wynik = pd.concat(czesci, ignore_index=True)[KOLUMNY_CONCATTED]
    return wynik, licz


def etap3a(concatted: pd.DataFrame):
    sys.path.insert(0, str(BASE_DIR / "3-sgkf-split" / "przygotowanie"))
    from przygotuj_dane import przygotuj  # noqa: E402
    return przygotuj(concatted)


# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tryb", choices=["realny", "brak", "wbudowany"], default="realny")
    ap.add_argument("--skala", type=float, default=1.0, help="ułamek liczby pacjentów (1.0 = jak w prawdziwych danych)")
    ap.add_argument("--ziarno", type=int, default=42)
    ap.add_argument("--wyjscie", type=Path, default=DOMYSLNE_WYJSCIE,
                    help="folder wynikowy (podfoldery raw/, imputation-inputs/, processed/)")
    a = ap.parse_args(argv)
    wyjscie = a.wyjscie if a.wyjscie.is_absolute() else (Path.cwd() / a.wyjscie)

    parametry = json.loads(PARAMETRY.read_text(encoding="utf-8"))
    g = Generator(parametry, a.tryb, a.skala, a.ziarno)
    print(f"Tryb {a.tryb}, skala {a.skala}, ziarno {a.ziarno} -> {wyjscie}", flush=True)

    pac = g.pacjenci()
    rek = g.rekordy(pac)
    obecne = g.wzorce(rek)
    wart = g.wartosci(rek)
    g.zasiej_niemozliwe(rek, wart, obecne)
    print(f"  pacjenci {len(pac)}, rekordy {len(rek)}; zapis danych surowych ...", flush=True)
    surowe = g.zapisz_surowe(rek, wart, obecne, wyjscie / "raw")

    concatted, licz1 = etap1_uproszczony(surowe, g.cechy)
    (wyjscie / "imputation-inputs").mkdir(parents=True, exist_ok=True)
    concatted.to_csv(wyjscie / "imputation-inputs" / "aneurysm_concatted.csv", index=False, lineterminator="\n")

    sgkf, usuniete_rek, usuniete_wart = etap3a(concatted)
    (wyjscie / "processed").mkdir(parents=True, exist_ok=True)
    sgkf.to_csv(wyjscie / "processed" / "aneurysm_sgkf_input.csv", index=False, lineterminator="\n")

    opis = {
        "UWAGA": "Dane SYNTETYCZNE. Nie opisują żadnego pacjenta. Wyników policzonych na nich nie zapisujemy "
                 "w folderach results/ i nie commitujemy.",
        "tryb": a.tryb, "skala": a.skala, "ziarno": a.ziarno,
        "wygenerowano": datetime.now().isoformat(timespec="seconds"),
        "parametry_z_dnia": parametry["wygenerowano"],
        "liczby": {
            "surowe": {k: {"rekordy": int(len(v)), "pacjenci": int(v["patient_id"].nunique())} for k, v in surowe.items()},
            "etap1_uproszczony": licz1,
            "aneurysm_concatted": {"rekordy": int(len(concatted)), "pacjenci": int(concatted["patient_id"].nunique()),
                                   "pacjenci_neuro": int(concatted.loc[concatted["label"] == 1, "patient_id"].nunique())},
            "aneurysm_sgkf_input": {"rekordy": int(len(sgkf)), "pacjenci": int(sgkf["patient_id"].nunique()),
                                    "pacjenci_neuro": int(sgkf.loc[sgkf["label"] == 1, "patient_id"].nunique()),
                                    "usuniete_rekordy_kor_mieszanych": int(len(usuniete_rek)),
                                    "wartosci_na_brak": usuniete_wart.groupby("regula").size().astype(int).to_dict()},
        },
        "klucz_odpowiedzi": g.klucz,
        "czego_nie_ma": ["wartości ujemnych", "informacji o pęknięciu tętniaka (diagnosis_id puste)",
                         "usuwania wartości odstających w uproszczonym etapie 1"],
    }
    (wyjscie / "OPIS.json").write_text(json.dumps(opis, ensure_ascii=False, indent=1), encoding="utf-8")
    l = opis["liczby"]["aneurysm_sgkf_input"]
    print(f"  wejście SGKF: {l['rekordy']} rekordów, {l['pacjenci']} pacjentów, chorych {l['pacjenci_neuro']}")
    print(f"Zapisano -> {wyjscie} (opis w OPIS.json)")


if __name__ == "__main__":
    main()
