"""
wyciagnij_parametry.py
======================

Liczy ZBIORCZE statystyki prawdziwych danych surowych, z których generator
(generuj.py) buduje dane syntetyczne. Wynik: parametry.json obok skryptu.

Do pliku trafiają wyłącznie wielkości zbiorcze:
  - liczby pacjentów i rekordów, rozkład liczby rekordów na pacjenta,
  - rozkład dat pierwszego badania (miesiące) i odstępów między badaniami,
  - kwantyle wieku i wartości badań (bez minimum i maksimum: 0,1–99,9%),
  - korelacje między cechami i udział zmienności "między pacjentami",
  - wzorce braków danych występujące w co najmniej MIN_WZORZEC rekordach,
  - nazwy badań (`examination_type`) zliczone według kohorty i roku,
  - format zapisu wartości (liczba czy słownik {'values': [...]}) i jednostki.

Żaden wiersz danych, identyfikator pacjenta ani data konkretnego badania
nie jest zapisywany.

Uruchomienie (wymaga prawdziwych danych w data/raw/):
    python narzedzia/dane_syntetyczne/wyciagnij_parametry.py
"""

import ast
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

BASE_DIR = Path(__file__).resolve().parents[2]
SUROWE = {
    "kor": BASE_DIR / "data" / "raw" / "kor_merged_aggregated_1W_mean.csv",
    "neuro": BASE_DIR / "data" / "raw" / "neuro_merged_aggregated_1W_mean.csv",
}
WYJSCIE = Path(__file__).resolve().parent / "parametry.json"

CECHY_LAB = ["MCH", "HGB", "MCV", "RBC", "MCHC", "PLT", "HCT", "WBC", "RDW", "%BAZO", "BAZO",
             "EO", "%EO", "LYMPH", "%LYMPH", "%MONO", "MONO", "MPV", "KREA", "Na", "K", "%IG",
             "IG", "eGFR-MDRD", "eGFRCKD", "NEUT", "%NEUT", "%NRBC", "NRBC", "WPT", "GLU", "INR",
             "PT", "CRP", "WAPTT", "APTT", "ALT", "BUN", "AST", "P-LCR"]

# Poziomy kwantyli zapisywanych dla każdej cechy (bez 0 i 100: skrajne wartości
# pojedynczych pacjentów nie trafiają do pliku).
POZIOMY = [0.001, 0.005] + [i / 100 for i in range(1, 100)] + [0.995, 0.999]
MIN_WZORZEC = 20        # wzorzec braków zapisujemy, jeśli ma co najmniej tyle rekordów
MAX_REK_HIST = 30       # liczba rekordów na pacjenta: histogram do 30, dalej kwantyle

# Grupy nazw badań: nazwy alternatywne dla tej samej grupy cech.
GRUPY_NAZW = {
    "morfologia": ["Morfologia", "Morfologia z retikulocytami"],
    "kreatynina": ["Kreatynina we krwi", "Kreatynina"],
    "mocznik": ["Azot mocznika we krwi", "Azot Mocznika"],
    "crp": ["Białko C-reaktywne"],
    "elektrolity": ["Elektrolity (potas, sód)", "Elektrolity"],
    "glukoza": ["Glukoza"],
    "egfr_mdrd": ["Wskaźnik filtracji kłębuszkowej (wg MDRD)"],
    "egfr_ckd": ["Wskaźnik filtracji kłębuszkowej (wg CKD-EPI)"],
    "pt": ["Czas protrombinowy"],
    "wpt": ["Wskaźnik protrombinowy"],
    "inr": ["INR"],
    "aptt": ["Czas częściowej tromboplastyny po aktywacji"],
    "waptt": ["Współczynnik APTT"],
    "alt": ["Transaminaza alaninowa"],
    "ast": ["Transaminaza asparaginianowa"],
}
MIN_NAZWA_DODATKOWA = 30   # nazwy spoza grup zapisujemy, jeśli wystąpiły w co najmniej tylu rekordach


def _liczba(x: str):
    try:
        return float(x)
    except ValueError:
        return None


def wartosc_liczbowa(s: pd.Series) -> tuple[pd.Series, pd.Series, list, Counter, int]:
    """(średnia liczb, czy zapis słownikowy, długości list, wpisy nieliczbowe, liczba wpisów).

    Wpisy nieliczbowe (np. '<0.1') nie wchodzą do średniej; liczymy tylko, jak często
    i jakie są, żeby generator mógł je odtworzyć.
    """
    slownik = s.str.startswith("{", na=False)
    wynik = pd.to_numeric(s.where(~slownik), errors="coerce")
    dlugosci, nieliczbowe, wpisy = [], Counter(), 0
    for i, v in s[slownik].items():
        lista = ast.literal_eval(v)["values"]
        dlugosci.append(len(lista))
        wpisy += len(lista)
        liczby = []
        for x in lista:
            f = _liczba(str(x))
            if f is None:
                nieliczbowe[str(x)] += 1
            else:
                liczby.append(f)
        wynik.at[i] = float(np.mean(liczby)) if liczby else np.nan
    zwykle = s[~slownik].dropna()
    wpisy += len(zwykle)
    for x in zwykle[pd.to_numeric(zwykle, errors="coerce").isna()]:
        nieliczbowe[str(x)] += 1
    return wynik, slownik, dlugosci, nieliczbowe, wpisy


def kwantyle(x: np.ndarray) -> list:
    x = x[~np.isnan(x)]
    return [round(float(v), 6) for v in np.quantile(x, POZIOMY)] if len(x) else []


def wyniki_normalne(df: pd.DataFrame) -> pd.DataFrame:
    """Każda cecha zamieniona na wynik normalny przez rangi (braki zostają brakami)."""
    out = pd.DataFrame(index=df.index)
    for c in CECHY_LAB:
        x = df[c].to_numpy(dtype=float)
        m = ~np.isnan(x)
        z = np.full(len(x), np.nan)
        if m.sum() > 1:
            z[m] = norm.ppf((rankdata(x[m]) - 0.5) / m.sum())
        out[c] = z
    return out


def korelacja(z: pd.DataFrame) -> list:
    k = z.corr(min_periods=200).fillna(0.0).to_numpy()
    np.fill_diagonal(k, 1.0)
    return [[round(float(v), 4) for v in w] for w in k]


def udzial_miedzy_pacjentami(z: pd.DataFrame, pacjent: pd.Series) -> dict:
    """Jaka część zmienności cechy to różnice między pacjentami (pacjenci z >= 2 pomiarami)."""
    wynik = {}
    for c in CECHY_LAB:
        d = pd.DataFrame({"p": pacjent, "z": z[c]}).dropna()
        n = d.groupby("p")["z"].transform("size")
        d = d[n >= 2]
        if len(d) < 200:
            wynik[c] = 0.5
            continue
        wewn = d.groupby("p")["z"].var(ddof=1).mean()
        calk = d["z"].var(ddof=1)
        wynik[c] = round(float(np.clip(1 - wewn / calk, 0.0, 0.95)), 3) if calk > 0 else 0.5
    return wynik


def main() -> None:
    parametry = {
        "opis": "Zbiorcze statystyki danych surowych (bez wierszy, identyfikatorów i dat badań). "
                "Plik dla generuj.py.",
        "wygenerowano": date.today().isoformat(),
        "cechy_lab": CECHY_LAB,
        "poziomy_kwantyli": POZIOMY,
        "grupy_nazw": GRUPY_NAZW,
        "kohorty": {},
    }
    numeryczne = {}
    for kohorta, sciezka in SUROWE.items():
        print(f"[{kohorta}] wczytywanie ...", flush=True)
        df = pd.read_csv(sciezka, dtype=str)
        daty = pd.to_datetime(df["examination_date"])
        p = {"n_rekordow": int(len(df)), "n_pacjentow": int(df["patient_id"].nunique())}

        # liczba rekordów na pacjenta
        rp = df.groupby("patient_id").size()
        hist = rp[rp <= MAX_REK_HIST].value_counts().sort_index()
        p["rek_na_pacjenta_hist"] = {str(int(k)): int(v) for k, v in hist.items()}
        ogon = rp[rp > MAX_REK_HIST].to_numpy()
        p["rek_na_pacjenta_ogon"] = {"n": int(len(ogon)),
                                     "kwantyle": [int(v) for v in np.quantile(ogon, np.linspace(0.05, 0.95, 10))]
                                     if len(ogon) else []}

        # daty: pierwsze badanie pacjenta (miesiąc) i odstępy między kolejnymi badaniami
        tmp = pd.DataFrame({"p": df["patient_id"], "d": daty}).sort_values(["p", "d"])
        pierwsze = tmp.groupby("p")["d"].min()
        p["pierwsze_badanie_miesiac"] = {k: int(v) for k, v in
                                         pierwsze.dt.strftime("%Y-%m").value_counts().sort_index().items()}
        odstepy = (tmp.groupby("p")["d"].diff().dt.days.dropna() / 7).to_numpy()
        p["odstep_tygodnie_kwantyle"] = [round(float(v), 2) for v in np.quantile(odstepy, POZIOMY)]
        p["zakres_lat"] = [int(daty.dt.year.min()), int(daty.dt.year.max())]
        p["rekordy_wg_roku"] = {str(k): int(v) for k, v in daty.dt.year.value_counts().sort_index().items()}

        # wiek przy pierwszym badaniu, płeć, braki metadanych
        wiek = pd.to_numeric(df["patient_age"], errors="coerce")
        pierwszy_idx = tmp.groupby("p").head(1).index
        p["wiek_pierwszy_kwantyle"] = kwantyle(wiek.loc[pierwszy_idx].to_numpy(dtype=float))
        plec = pd.to_numeric(df["patient_sex"], errors="coerce")
        p["plec_odsetek_1"] = round(float(plec.groupby(df["patient_id"]).first().mean()), 4)
        p["plec_zapis"] = sorted(df["patient_sex"].dropna().unique().tolist())
        p["braki_wiek"] = int(wiek.isna().sum())
        p["braki_plec"] = int(plec.isna().sum())
        p["wiek_poza_18_100"] = int(((wiek < 18) | (wiek > 100)).sum())
        p["odsetek_opisow"] = round(float(df["descriptive_result"].notna().mean()), 4)
        p["diagnosis_id_wypelnione"] = int(df["diagnosis_id"].notna().sum())
        p["max_patient_id"] = int(pd.to_numeric(df["patient_id"]).max())

        # wartości badań: format, jednostki, kwantyle
        num = pd.DataFrame(index=df.index)
        format_ = {}
        dl_wszystkie = Counter()
        for c in CECHY_LAB:
            wart, slownik, dl, nieliczbowe, wpisy = wartosc_liczbowa(df[c])
            num[c] = wart
            n_war = int(df[c].notna().sum())
            jedn = df[f"{c}-unit"].dropna()
            format_[c] = {
                "odsetek_obecnych": round(n_war / len(df), 4),
                "odsetek_slownik": round(float(slownik.sum() / n_war), 4) if n_war else 0.0,
                "jednostka": jedn.mode().iloc[0] if len(jedn) else "",
                "kwantyle": kwantyle(wart.to_numpy(dtype=float)),
                "ujemne": int((wart < 0).sum()),
                # wpisy nieliczbowe (np. '<0.1'): tylko te, które wystąpiły co najmniej 20 razy
                "nieliczbowe_odsetek": {k: round(v / wpisy, 5) for k, v in nieliczbowe.most_common(5)
                                        if v >= 20} if wpisy else {},
            }
            dl_wszystkie.update(min(x, 6) for x in dl)
        p["cechy"] = format_
        suma = sum(dl_wszystkie.values()) or 1
        p["dlugosc_listy"] = {str(k): round(v / suma, 4) for k, v in sorted(dl_wszystkie.items())}

        # wzorce braków (40 cech): tylko wzorce z co najmniej MIN_WZORZEC rekordami
        bity = num[CECHY_LAB].notna().astype(int).astype(str).agg("".join, axis=1)
        licz = bity.value_counts()
        czeste = licz[licz >= MIN_WZORZEC]
        p["wzorce_brakow"] = {k: int(v) for k, v in czeste.items()}
        p["wzorce_pokrycie"] = round(float(czeste.sum() / len(df)), 4)

        # nazwy badań wg roku
        rok = daty.dt.year.astype(str).to_numpy()
        wg_roku = defaultdict(Counter)
        wszystkie = Counter()
        for r, v in zip(rok, df["examination_type"]):
            if isinstance(v, str):
                nazwy = set(ast.literal_eval(v)["values"])
                wg_roku[r].update(nazwy)
                wszystkie.update(nazwy)
        w_grupach = {n for g in GRUPY_NAZW.values() for n in g}
        p["nazwy_wg_roku"] = {r: {n: int(k) for n, k in c.items() if n in w_grupach}
                              for r, c in sorted(wg_roku.items())}
        p["nazwy_dodatkowe_odsetek"] = {n: round(k / len(df), 5) for n, k in wszystkie.most_common()
                                        if n not in w_grupach and k >= MIN_NAZWA_DODATKOWA}

        # korelacje i zmienność między pacjentami
        z = wyniki_normalne(num)
        p["korelacja"] = korelacja(z)
        p["udzial_miedzy_pacjentami"] = udzial_miedzy_pacjentami(z, df["patient_id"])

        parametry["kohorty"][kohorta] = p
        num["patient_id"] = df["patient_id"].to_numpy()
        numeryczne[kohorta] = num

    # pacjenci obecni w obu kohortach (tylko liczba)
    wspolni = set(numeryczne["kor"]["patient_id"]) & set(numeryczne["neuro"]["patient_id"])
    parametry["pacjenci_w_obu_kohortach"] = len(wspolni)

    # parametry wspólne (obie kohorty razem): do trybów bez różnic między kohortami
    razem = pd.concat([numeryczne["kor"], numeryczne["neuro"]], ignore_index=True)
    parametry["wspolne"] = {
        "kwantyle": {c: kwantyle(razem[c].to_numpy(dtype=float)) for c in CECHY_LAB},
        "korelacja": korelacja(wyniki_normalne(razem)),
    }

    WYJSCIE.write_text(json.dumps(parametry, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, p in parametry["kohorty"].items():
        print(f"[{k}] pacjenci {p['n_pacjentow']}, rekordy {p['n_rekordow']}, "
              f"wzorce braków {len(p['wzorce_brakow'])} (pokrycie {p['wzorce_pokrycie']:.1%})")
    print(f"pacjenci w obu kohortach: {parametry['pacjenci_w_obu_kohortach']}")
    print(f"Zapisano -> {WYJSCIE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
