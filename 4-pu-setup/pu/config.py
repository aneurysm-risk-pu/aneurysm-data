"""
Konfiguracja protokolu PU.

Kazda wartosc w tym pliku odpowiada decyzji z etapu 0 w
02_PLAN_PRZED_MODELOWANIEM.md. Zadna z nich nie jest wpleciona w logike
pipeline'u — zmiana decyzji po spotkaniu to zmiana wartosci tutaj i ponowne
uruchomienie, nie przepisywanie kodu.

Stan decyzji na 20.09.2026:
  0.1  status 63 pacjentow mieszanych   — DO ROZSTRZYGNIECIA (spotkanie)
  0.2  kontrola czasu i pochodzenia     — DO ROZSTRZYGNIECIA (spotkanie)
  0.3  regula agregacji                 — rekomendacja: mediana
  0.4  zestaw cech                      — rekomendacja: 38 cech
  0.5  wartosci skrajne i jednostki     — DO ROZSTRZYGNIECIA (spotkanie)
  0.6  konfiguracja podzialu            — liczby gotowe, wybor do zamrozenia
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

BASE_DIR = Path(__file__).parent.parent.parent

# Dwa warianty zestawu cech roznia sie wylacznie trzema kolumnami.
# Plik z 38 cechami jest co do wartosci identyczny z plikiem z 35 cechami.
DANE_38_CECH = BASE_DIR / "data" / "imputation-inputs" / "aneurysm_concatted.csv"
DANE_35_CECH = BASE_DIR / "data" / "processed" / "aneurysm_concatted_cleaned.csv"

META_KOLUMNY = ["patient_id", "custom_id", "examination_date", "label"]

# Orientacyjne granice przezycia — patrz punkt 0.5 w ETAP0_USTALENIA.md.
ZAKRESY_FIZJOLOGICZNE = {"WBC": (0.1, 200.0), "Na": (100.0, 180.0),
                         "K": (1.5, 9.0), "KREA": (0.1, 2000.0)}


@dataclass
class KonfiguracjaPU:
    """Komplet decyzji protokolu. Zapisywany razem z wynikami jako artefakt."""

    # --- 0.4 zestaw cech -------------------------------------------------
    zestaw_cech: Literal["38", "35"] = "38"

    # --- 0.1 pacjenci obecni w obu kohortach ------------------------------
    # 'pozytywni'      : pacjent mieszany dostaje etykiete 1 (byl w NEURO)
    # 'bez_mieszanych' : pacjenci mieszani wypadaja z analizy
    mieszani: Literal["pozytywni", "bez_mieszanych"] = "pozytywni"

    # --- 0.2 kontrola czasu i pochodzenia ---------------------------------
    # 'brak'          : wariant B — modelujemy na calosci, ograniczenie do raportu
    # 'wspolne_okno'  : wariant A — tylko rekordy z okna dat wspolnego dla kohort
    # Wariant C (dopasowanie/wazenie kohort) wchodzi przez `hak_kontroli_kohort`
    # w foldy.py — gniazdo jest gotowe, zeby nie przebudowywac pipeline'u.
    kontrola_czasu: Literal["brak", "wspolne_okno"] = "brak"

    # --- 0.5 wartosci skrajne ---------------------------------------------
    # 'zostaw'  : nie ruszamy niczego
    # 'na_nan'  : wartosci poza zakresem przezycia traktujemy jak brak danych
    wartosci_skrajne: Literal["zostaw", "na_nan"] = "zostaw"

    # --- 0.3 jednostka treningu i agregacja -------------------------------
    jednostka_treningu: Literal["pacjent", "rekord"] = "pacjent"
    regula_agregacji: Literal["mediana", "srednia"] = "mediana"

    # --- 0.6 podzial -------------------------------------------------------
    poziom_podzialu: Literal["pacjent", "rekord"] = "pacjent"
    n_splits_outer: int = 5
    n_splits_inner: int = 3
    shuffle: bool = True
    seed_podzialu: int = 42

    # --- punkt 2: ukrywanie pozytywnych ------------------------------------
    udzial_ukrycia_glowny: float = 0.40
    udzialy_ukrycia_wrazliwosc: tuple = (0.20, 0.60)
    seedy_ukrycia: tuple = (1, 2, 3, 4, 5)

    # --- punkt 5: funkcja celu ---------------------------------------------
    # q = odsetek puli nieoznaczonej, ktory realnie da sie skierowac na obrazowanie
    q: float = 0.05
    alfa: float = 1.0          # 1.0 = czysty RecallHidden@q, bez skladnika RecallKnown

    # --- preprocessing -----------------------------------------------------
    # True  : scaler dopasowany na kompletnych wierszach treningu (jak w benchmarku)
    # False : scaler dopasowany kolumnowo na wszystkich dostepnych wartosciach
    # Wariant z benchmarku opiera skale na selektywnej podprobie — kompletne
    # wiersze moga roznic sie klinicznie od rekordow z brakami.
    scaler_na_kompletnych: bool = True

    # --- imputacja (parametry z benchmarku, RAPORT_IMPUTACJA.md) ------------
    mice_max_iter: int = 7
    mice_n_estimators: int = 78
    mice_max_depth: int = 10
    seed_imputacji: int = 42

    notatka: str = ""

    @property
    def sciezka_danych(self) -> Path:
        return DANE_38_CECH if self.zestaw_cech == "38" else DANE_35_CECH

    def pola(self) -> dict:
        """Same pola konfiguracji — nadaje sie do odtworzenia obiektu."""
        return asdict(self)

    def zmieniona(self, **kw) -> "KonfiguracjaPU":
        """Kopia z podmieniona wartoscia, np. cfg.zmieniona(zestaw_cech='35')."""
        return KonfiguracjaPU(**{**self.pola(), **kw})

    def do_slownika(self) -> dict:
        d = asdict(self)
        d["sciezka_danych"] = str(self.sciezka_danych)
        return d


# Konfiguracja domyslna — rekomendacje zespolu, do zamrozenia po spotkaniu.
DOMYSLNA = KonfiguracjaPU(
    notatka="Rekomendacje z etapu 0, przed zamrozeniem na spotkaniu 21.09.2026")
