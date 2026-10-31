"""
podzial.py
==========

CZESC B etapu 3 — zamrozony podzial na foldy metoda StratifiedGroupKFold (SGKF).

Co robi, krok po kroku:

  1. wczytaj_rekordy()   czyta data/processed/aneurysm_sgkf_input.csv: polaczone
                         KOR i NEURO po przygotowaniu z czesci A, przed imputacja.
  2. tabela_pacjentow()  dla kazdego pacjenta wyznacza warstwe: etykieta x okno
                         czasowe x klasa liczby rekordow (12 warstw). Warstwa
                         mowi, w jakim pacjent jest "typie"; nikogo nie usuwa.
  3. zbuduj_podzial()    uruchamia StratifiedGroupKFold na rekordach:
                         y = warstwa pacjenta, groups = pacjent. SGKF trzyma
                         wszystkie rekordy pacjenta w jednym foldzie i wyrownuje
                         udzial kazdej warstwy miedzy foldami.
  4. zapisz_podzial()    zapisuje przydzial pacjent -> fold razem z odciskiem
                         pliku danych i wersjami bibliotek.

Od tej pory zrodlem prawdy jest plik `results/pacjent_fold.csv`, a nie przepis na
podzial. Kazdy kolejny etap czyta go przez `wczytaj_podzial()`.

Co zmienia wzgledem wersji 1 z czerwca (to tez byl SGKF, z y = etykieta):

  a. y to warstwa zamiast samej etykiety. Sama etykieta zostawiala miedzy foldami
     okolo 4 pp rozjazdu w udziale NEURO z okna czasowego KOR, a to znane zrodlo
     falszywego sygnalu (1026_REALIZACJA_DO_SGKF.md, sekcja 6).

  b. Losowa kolejnosc pacjentow, sterowana ziarnem. SGKF bez tasowania przetwarza
     pacjentow wedlug numerow: nie zalezy od kolejnosci wierszy w CSV, ale ziarno
     nie ma na niego wplywu, wiec nie da sie sprawdzic, czy wynik nie jest
     przypadkowy.

  c. Przydzial jest zapisany z odciskiem danych. Zmiana pliku danych uniewaznia
     zapisany podzial, zamiast po cichu liczyc foldy od nowa.

Dlaczego kolejnosc mieszamy sami, a nie przez StratifiedGroupKFold(shuffle=True):
w sklearn 1.6 shuffle=True miesza tabele liczebnosci, ale gubi informacje, ktory
wiersz nalezy do ktorego pacjenta, wiec wyrownanie sie psuje. W sklearn 1.8 jest
to naprawione, a wiec ta sama konfiguracja daje na tych wersjach inne foldy.
Przemieszanie numerow pacjentow przed wywolaniem daje poprawne tasowanie bez tej
opcji, wiec wynik nie zalezy od tego bledu.

Wariant testowany i odrzucony: StratifiedKFold na tabeli pacjentow (jeden wiersz
= jeden pacjent). Na pacjentach wyrownuje foldy praktycznie tak samo, na rekordach
gorzej (porownaj_strategie, results/podzial_porownanie_strategii.csv).

Uruchomienie:
    python 3-sgkf-split/sgkf/podzial.py               # buduje, sprawdza i zapisuje podzial
    python 3-sgkf-split/sgkf/podzial.py --porownanie  # dodatkowo porownuje strategie na 5 ziarnach
    python 3-sgkf-split/sgkf/podzial.py --sprawdz     # weryfikuje zapisany podzial wzgledem danych
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

BASE_DIR = Path(__file__).resolve().parents[2]
WYNIKI = Path(__file__).resolve().parent / "results"
PODSUMOWANIE_PRZYGOTOWANIA = BASE_DIR / "3-sgkf-split" / "przygotowanie" / "results" / "podsumowanie.json"

# Wejscie: wynik czesci A (przygotowanie danych), 38 cech, dane PRZED imputacja.
# Wariant 35 cech (bez CRP, MONO, %MONO) sluzy wylacznie analizie porownawczej
# (porownanie_cech.py); decyzja o 38 cechach: przygotowanie/analiza_selekcji_cech.py.
WEJSCIE = BASE_DIR / "data" / "processed" / "aneurysm_sgkf_input.csv"
CECHY_POZA_35 = ["CRP", "MONO", "%MONO"]

KOL_PACJENT = "patient_id"
KOL_ETYKIETA = "label"
KOL_DATA = "examination_date"
META_KOLUMNY = ["patient_id", "custom_id", "examination_date", "label"]

# Granice klas liczby rekordow: 1 | 2 | 3-4 | 5+
GRANICE_N_REK = [0, 1, 2, 4, np.inf]
ETYKIETY_N_REK = ["1", "2", "3-4", "5+"]


@dataclass
class KonfiguracjaPodzialu:
    """Komplet decyzji podzialu. Zapisywany razem z przydzialem."""


    # 0.6 — konfiguracja podzialu
    n_splits: int = 5
    shuffle: bool = True     # True: losowa kolejnosc pacjentow sterowana ziarnem (patrz _przydzial_sgkf)
    seed: int = 42

    # Zmienne stratyfikujace, w tej kolejnosci laczone w jeden klucz.
    # 'etykieta' jest obowiazkowa, pozostale mozna wylaczyc.
    warstwy: tuple = ("etykieta", "okno", "n_rek")

    # Warstwa mniejsza niz `min_warstwa` jest doklejana do warstwy
    # "<etykieta>_inne", zeby kazda warstwa dala sie rozlozyc na wszystkie foldy.
    min_warstwa: int = 20

    @property
    def sciezka_danych(self) -> Path:
        return WEJSCIE


DOMYSLNA = KonfiguracjaPodzialu()


# ---------------------------------------------------------------------------
# Dane i tabela pacjentow
# ---------------------------------------------------------------------------

def wczytaj_rekordy(cfg: KonfiguracjaPodzialu = None) -> pd.DataFrame:
    """Dane przygotowane w czesci A, PRZED imputacja (imputacja dzieje sie w foldzie)."""
    cfg = cfg or DOMYSLNA
    if not cfg.sciezka_danych.exists():
        raise FileNotFoundError(f"brak {cfg.sciezka_danych.relative_to(BASE_DIR)} — uruchom najpierw "
                                "3-sgkf-split/przygotowanie/przygotuj_dane.py")
    df = pd.read_csv(cfg.sciezka_danych)
    df[KOL_DATA] = pd.to_datetime(df[KOL_DATA])
    return df


def kolumny_cech(df: pd.DataFrame, zestaw: str = "38") -> list[str]:
    """Cechy wariantu 38 albo 35 (bez CRP, MONO, %MONO)."""
    pomin = set(META_KOLUMNY) | (set(CECHY_POZA_35) if zestaw == "35" else set())
    return [c for c in df.columns if c not in pomin]


def wspolne_okno_dat(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Przeciecie zakresow dat obu kohort (2019-12-29 – 2021-12-26 dla obecnych danych)."""
    kor = df.loc[df[KOL_ETYKIETA] == 0, KOL_DATA]
    neuro = df.loc[df[KOL_ETYKIETA] == 1, KOL_DATA]
    return max(kor.min(), neuro.min()), min(kor.max(), neuro.max())


def tabela_pacjentow(df: pd.DataFrame, cfg: KonfiguracjaPodzialu) -> pd.DataFrame:
    """Jeden wiersz na pacjenta: etykieta, flagi kontrolne i warstwa stratyfikacji.

    `w_oknie` = co najmniej polowa rekordow pacjenta lezy we wspolnym oknie dat
    kohort. Dla KOR to prawie zawsze prawda; dla NEURO odroznia ok. 9% pacjentow
    z lat 2019–2021 od reszty, ktora pochodzi z innej epoki
    (results/podzial_diagnostyka.csv, kolumna poz_w_oknie).
    """
    od, do = wspolne_okno_dat(df)
    w_oknie = df[KOL_DATA].between(od, do)
    g = df.groupby(KOL_PACJENT)
    pac = pd.DataFrame({
        "label": g[KOL_ETYKIETA].max().astype(int),
        "n_rek": g.size(),
        "w_oknie": w_oknie.groupby(df[KOL_PACJENT]).mean().ge(0.5),
        "rok_mediana": g[KOL_DATA].median().dt.year,
    }).reset_index()
    pac["klasa_n_rek"] = pd.cut(pac["n_rek"], GRANICE_N_REK,
                                labels=ETYKIETY_N_REK).astype(str)

    pac["warstwa"] = _warstwy(pac, cfg)
    return pac.sort_values(KOL_PACJENT).reset_index(drop=True)


def _warstwy(pac: pd.DataFrame, cfg: KonfiguracjaPodzialu) -> pd.Series:
    zrodla = {"etykieta": pac["label"].astype(str),
              "okno": np.where(pac["w_oknie"], "okno", "pozaokno"),
              "n_rek": "n" + pac["klasa_n_rek"]}
    if "etykieta" not in cfg.warstwy:
        raise ValueError("stratyfikacja musi obejmowac etykiete")
    klucz = pd.Series("", index=pac.index)
    for w in cfg.warstwy:
        klucz = klucz + "_" + pd.Series(zrodla[w], index=pac.index).astype(str)
    klucz = klucz.str.lstrip("_")

    licznosc = klucz.map(klucz.value_counts())
    male = licznosc < cfg.min_warstwa
    klucz[male] = pac.loc[male, "label"].astype(str) + "_inne"
    return klucz


# ---------------------------------------------------------------------------
# Podzial
# ---------------------------------------------------------------------------

def _przydzial_sgkf(df: pd.DataFrame, y: np.ndarray, n_splits: int, seed: Optional[int]) -> pd.Series:
    """StratifiedGroupKFold na rekordach: patient_id -> numer foldu (0 .. n_splits-1).

    SGKF przetwarza pacjentow wedlug wartosci `groups`. Przy seed=None sa to numery
    pacjentow. Przy podanym ziarnie kazdy pacjent dostaje losowy numer roboczy, co
    daje losowa kolejnosc. RandomState, bo jego strumien jest stabilny miedzy
    wersjami NumPy; wbudowanego shuffle=True nie uzywamy (naglowek pliku).
    """
    pacjenci = df[KOL_PACJENT].to_numpy()
    ids = np.unique(pacjenci)
    grupy = pacjenci
    if seed is not None:
        numer_roboczy = np.random.RandomState(seed).permutation(len(ids))
        grupy = numer_roboczy[np.searchsorted(ids, pacjenci)]
    przydzial = pd.Series(-1, index=ids, name="fold")
    przydzial.index.name = KOL_PACJENT
    for i, (_, te) in enumerate(StratifiedGroupKFold(n_splits=n_splits).split(pacjenci, y, grupy)):
        przydzial.loc[np.unique(pacjenci[te])] = i
    return przydzial


def _warstwa_rekordu(df: pd.DataFrame, pac: pd.DataFrame) -> np.ndarray:
    """Kazdy rekord dostaje warstwe swojego pacjenta."""
    return df[KOL_PACJENT].map(pac.set_index(KOL_PACJENT)["warstwa"]).to_numpy()


def zbuduj_podzial(df: pd.DataFrame, pac: pd.DataFrame, cfg: KonfiguracjaPodzialu) -> pd.Series:
    """patient_id -> numer foldu testowego: SGKF z y = warstwa, groups = pacjent."""
    przydzial = _przydzial_sgkf(df, _warstwa_rekordu(df, pac), cfg.n_splits,
                                cfg.seed if cfg.shuffle else None)
    sprawdz_podzial(pac, przydzial, cfg.n_splits)
    return przydzial


def podzial_wewnetrzny(df: pd.DataFrame, pac: pd.DataFrame, pacjenci_train: np.ndarray,
                       n_splits: int = 3, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Foldy do strojenia, wewnatrz treningu foldu zewnetrznego: ta sama metoda i te same warstwy."""
    d = df[df[KOL_PACJENT].isin(set(pacjenci_train))]
    przydzial = _przydzial_sgkf(d, _warstwa_rekordu(d, pac), n_splits, seed)
    for i in range(n_splits):
        yield przydzial.index[przydzial != i].to_numpy(), przydzial.index[przydzial == i].to_numpy()


def sprawdz_podzial(pac: pd.DataFrame, przydzial: pd.Series, n_splits: int) -> None:
    """Warunki konieczne — ich zlamanie oznacza blad, nie gorszy podzial."""
    assert przydzial.index.is_unique, "pacjent wystepuje w przydziale wiecej niz raz"
    assert set(przydzial.index) == set(pac[KOL_PACJENT]), "przydzial nie pokrywa tabeli pacjentow"
    assert przydzial.between(0, n_splits - 1).all(), "pacjent bez foldu"
    poz = pac.set_index(KOL_PACJENT)["label"].groupby(przydzial).sum()
    assert len(poz) == n_splits and (poz > 0).all(), "fold bez pozytywnych pacjentow"


def indeksy_foldow(df: pd.DataFrame, przydzial: pd.Series
                   ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Przeklada przydzial pacjentow na pozycje rekordow: (fold, train_idx, test_idx).

    Rekordy pacjentow spoza przydzialu (np. przy podprobce) nie trafiaja
    ani do treningu, ani do testu.
    """
    fold_rek = df[KOL_PACJENT].map(przydzial)
    for f in sorted(przydzial.unique()):
        test = np.flatnonzero(fold_rek.eq(f).to_numpy())
        train = np.flatnonzero((fold_rek.notna() & fold_rek.ne(f)).to_numpy())
        yield int(f), train, test


# ---------------------------------------------------------------------------
# Diagnostyka
# ---------------------------------------------------------------------------

def diagnostyka(df: pd.DataFrame, pac: pd.DataFrame, przydzial: pd.Series) -> pd.DataFrame:
    """Statystyki foldow: to, co musi byc wyrownane, zeby fold byl porownywalny."""
    p = pac.set_index(KOL_PACJENT).join(przydzial)
    rek_fold = df[KOL_PACJENT].map(przydzial)
    wiersze = []
    for f, s in p.groupby("fold"):
        poz = s[s["label"] == 1]
        rek = df[rek_fold.eq(f)]
        wiersze.append({
            "fold": int(f),
            "pacjenci": len(s),
            "rekordy": len(rek),
            "pozytywni": int(s["label"].sum()),
            "udzial_poz_pacjenci": s["label"].mean(),
            "udzial_poz_rekordy": rek[KOL_ETYKIETA].mean(),
            "poz_w_oknie": poz["w_oknie"].mean(),
            "poz_5plus_rek": poz["klasa_n_rek"].eq("5+").mean(),
            "poz_rok_mediana": poz["rok_mediana"].median(),
            "u_1_rek": s.loc[s["label"] == 0, "klasa_n_rek"].eq("1").mean(),
        })
    return pd.DataFrame(wiersze)


def rozstepy(diag: pd.DataFrame) -> dict:
    """Najwazniejsze liczby jednym rzutem oka. Udzialy w punktach procentowych."""
    r = lambda k: float(diag[k].max() - diag[k].min())
    return {
        "udzial_poz_pacjenci_pp": 100 * r("udzial_poz_pacjenci"),
        "udzial_poz_rekordy_pp": 100 * r("udzial_poz_rekordy"),
        "rekordy_wzgl_proc": 100 * r("rekordy") / diag["rekordy"].mean(),
        "poz_w_oknie_pp": 100 * r("poz_w_oknie"),
        "poz_5plus_rek_pp": 100 * r("poz_5plus_rek"),
    }


# ---------------------------------------------------------------------------
# Porownanie strategii — uzasadnienie wyboru do raportu
# ---------------------------------------------------------------------------

def _przydzial_sgkf_wbudowane_tasowanie(df: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """SGKF z y = etykieta i shuffle=True. Wynik zalezy od wersji sklearn (naglowek pliku)."""
    pacjenci = df[KOL_PACJENT].to_numpy()
    przydzial = pd.Series(-1, index=np.unique(pacjenci), name="fold")
    przydzial.index.name = KOL_PACJENT
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgkf.split(pacjenci, df[KOL_ETYKIETA].to_numpy(), pacjenci)):
        przydzial.loc[np.unique(pacjenci[te])] = i
    return przydzial


def _przydzial_skf_pacjenci(pac: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """Wariant testowany: StratifiedKFold na tabeli pacjentow (jeden wiersz = jeden pacjent)."""
    ids = pac[KOL_PACJENT].to_numpy()
    przydzial = pd.Series(-1, index=ids, name="fold")
    przydzial.index.name = KOL_PACJENT
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(skf.split(ids, pac["warstwa"])):
        przydzial.iloc[te] = i
    return przydzial


def porownaj_strategie(df: pd.DataFrame, cfg: KonfiguracjaPodzialu,
                       ziarna=(42, 7, 13, 101, 2024)) -> pd.DataFrame:
    """Te same miary dla czterech strategii; dla losowych — najgorszy z 5 ziaren."""
    pac = tabela_pacjentow(df, cfg)
    etykieta = df[KOL_ETYKIETA].to_numpy()

    strategie = {
        "SGKF, y = etykieta, bez tasowania (wersja 1)":
            [lambda: _przydzial_sgkf(df, etykieta, cfg.n_splits, None)],
        "SGKF, y = etykieta, wbudowane shuffle=True":
            [lambda s=s: _przydzial_sgkf_wbudowane_tasowanie(df, cfg.n_splits, s) for s in ziarna],
        "StratifiedKFold na tabeli pacjentow, 12 warstw (wariant testowany)":
            [lambda s=s: _przydzial_skf_pacjenci(pac, cfg.n_splits, s) for s in ziarna],
        "SGKF, y = 12 warstw, losowa kolejnosc pacjentow (wersja 2, wybrana)":
            [lambda s=s: _przydzial_sgkf(df, _warstwa_rekordu(df, pac), cfg.n_splits, s) for s in ziarna],
    }
    wiersze = []
    for nazwa, budowniczowie in strategie.items():
        wyniki = [rozstepy(diagnostyka(df, pac, b())) for b in budowniczowie]
        najgorszy = {k: max(w[k] for w in wyniki) for k in wyniki[0]}
        wiersze.append({"strategia": nazwa, "przebiegi": len(wyniki), **najgorszy})
    return pd.DataFrame(wiersze)


# ---------------------------------------------------------------------------
# Zapis i odczyt — przydzial jest artefaktem, nie przepisem
# ---------------------------------------------------------------------------

def odcisk_pliku(sciezka: Path) -> str:
    """SHA-256 tresci z ujednoliconymi koncami linii.

    Git na Windowsie zamienia LF na CRLF przy checkoucie, wiec surowy hash
    bajtow rozni sie miedzy komputerami przy identycznych danych.
    """
    tresc = sciezka.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(tresc).hexdigest()[:16]


def zapisz_podzial(pac: pd.DataFrame, przydzial: pd.Series, diag: pd.DataFrame,
                   cfg: KonfiguracjaPodzialu, katalog: Path = WYNIKI) -> None:
    katalog.mkdir(parents=True, exist_ok=True)
    tabela = pac[[KOL_PACJENT, "label", "warstwa"]].merge(
        przydzial.reset_index(), on=KOL_PACJENT)
    tabela[[KOL_PACJENT, "fold", "label", "warstwa"]].to_csv(
        katalog / "pacjent_fold.csv", index=False)
    diag.to_csv(katalog / "podzial_diagnostyka.csv", index=False)

    meta = {
        "metoda": "StratifiedGroupKFold na rekordach: y = warstwa pacjenta, groups = pacjent, "
                  "losowa kolejnosc pacjentow sterowana ziarnem",
        "konfiguracja": {**asdict(cfg), "warstwy": list(cfg.warstwy)},
        "plik_danych": str(cfg.sciezka_danych.relative_to(BASE_DIR)).replace("\\", "/"),
        "odcisk_danych_sha256": odcisk_pliku(cfg.sciezka_danych),
        "przygotowanie_danych": (json.loads(PODSUMOWANIE_PRZYGOTOWANIA.read_text(encoding="utf-8"))
                                 if PODSUMOWANIE_PRZYGOTOWANIA.exists() else {}),
        "pacjenci": int(len(pac)),
        "pozytywni": int(pac["label"].sum()),
        "rozstepy": rozstepy(diag),
        "wersje": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                   "pandas": pd.__version__, "numpy": np.__version__},
        "utworzono": datetime.now().isoformat(timespec="seconds"),
    }
    (katalog / "podzial_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def wczytaj_podzial(katalog: Path = WYNIKI) -> tuple[pd.Series, dict]:
    """Zwraca (przydzial patient_id -> fold, metadane).

    Odcisk pliku danych jest sprawdzany: jesli dane zmienily sie od zapisu
    podzialu, przydzial przestaje byc wiarygodny i trzeba go zbudowac od nowa.
    """
    meta = json.loads((katalog / "podzial_meta.json").read_text(encoding="utf-8"))
    sciezka = BASE_DIR / meta["plik_danych"]
    teraz = odcisk_pliku(sciezka)
    if teraz != meta["odcisk_danych_sha256"]:
        raise RuntimeError(f"{sciezka.name} zmienil sie od zapisu podzialu "
                           f"({meta['odcisk_danych_sha256']} -> {teraz}); "
                           "uruchom ponownie 3-sgkf-split/podzial.py")
    t = pd.read_csv(katalog / "pacjent_fold.csv")
    return t.set_index(KOL_PACJENT)["fold"], meta


# ---------------------------------------------------------------------------

def _drukuj(diag: pd.DataFrame) -> None:
    print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for k, v in rozstepy(diag).items():
        print(f"  rozstep {k:24s} {v if isinstance(v, list) else f'{v:.3f}'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--porownanie", action="store_true",
                    help="porownaj strategie podzialu na 5 ziarnach")
    ap.add_argument("--sprawdz", action="store_true",
                    help="tylko zweryfikuj zapisany podzial wzgledem danych")
    args = ap.parse_args(argv)
    cfg = DOMYSLNA

    if args.sprawdz:
        przydzial, meta = wczytaj_podzial()
        df = wczytaj_rekordy(KonfiguracjaPodzialu(**{**meta["konfiguracja"],
                                                    "warstwy": tuple(meta["konfiguracja"]["warstwy"])}))
        brak = set(przydzial.index) ^ set(df[KOL_PACJENT])
        assert not brak, f"{len(brak)} pacjentow nie zgadza sie miedzy przydzialem a danymi"
        print(f"Podzial zgodny z danymi: {len(przydzial):,} pacjentow, "
              f"odcisk danych {meta['odcisk_danych_sha256']}, "
              f"utworzony {meta['utworzono']} "
              f"(sklearn {meta['wersje']['sklearn']})")
        return

    print(f"[1] Wczytywanie {cfg.sciezka_danych.name} ...")
    df = wczytaj_rekordy(cfg)
    pac = tabela_pacjentow(df, cfg)
    print(f"    {len(df):,} rekordow, {len(pac):,} pacjentow, "
          f"{int(pac['label'].sum()):,} pozytywnych")
    print(f"    warstw stratyfikacji: {pac['warstwa'].nunique()} "
          f"(najmniejsza: {pac['warstwa'].value_counts().min()} pacjentow)")

    print(f"\n[2] Podzial StratifiedGroupKFold: {cfg.n_splits} foldow, losowa kolejnosc={cfg.shuffle}, "
          f"seed={cfg.seed}, warstwy={'+'.join(cfg.warstwy)}\n")
    przydzial = zbuduj_podzial(df, pac, cfg)
    diag = diagnostyka(df, pac, przydzial)
    _drukuj(diag)

    if args.porownanie:
        print("\n[3] Porownanie strategii (dla losowych: najgorszy z 5 ziaren)\n")
        por = porownaj_strategie(df, cfg)
        print(por.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        WYNIKI.mkdir(parents=True, exist_ok=True)
        por.to_csv(WYNIKI / "podzial_porownanie_strategii.csv", index=False)

    zapisz_podzial(pac, przydzial, diag, cfg)
    print(f"\nZapisano -> {WYNIKI.relative_to(BASE_DIR)}/pacjent_fold.csv, "
          "podzial_diagnostyka.csv, podzial_meta.json")


if __name__ == "__main__":
    main()
