"""
uruchom_sgkf.py
===============

Cały etap 3 jednym poleceniem — od danych źródłowych do zaimputowanych foldów.
Dwie części, uruchamiane po kolei; pierwszy błąd przerywa przebieg, a każdy
krok zapisuje log do results/logi/ swojej części.

CZĘŚĆ A — przygotowanie danych do SGKF (3-sgkf-split/przygotowanie/)
  A1  przygotuj_dane.py               decyzje -> data/processed/aneurysm_sgkf_input.csv + spis zmian
  A2  tests/test_przygotowanie.py     reguły + zgodność pliku z kodem
  A3  analiza_decyzji.py              liczby uzasadniające decyzje
  A4  analiza_selekcji_cech.py        dlaczego 38 cech (odtworzenie selekcji z etapu 1)

CZĘŚĆ B — podział i imputacja w foldach (3-sgkf-split/sgkf/)
  B1  podzial.py --porownanie         zamrożony przydział pacjent -> fold + porównanie strategii
  B2  tests/test_podzial.py           testy podziału
  B3  aneurysm_sgkf_mice_pipeline.py  MICE w foldach, 38 cech
  B4  ocena_imputacji.py              jakość imputacji protokołem z etapu 2 (maski na kompletnych wierszach)
  B5  analiza_wynikow.py              diagnostyka wyników: rozkłady uzupełnień, skróty, profil kohort

Jednorazowa analiza porównawcza 38 vs 35 cech (podkładka pod decyzję, wyniki
w sgkf/results/porownanie_38_35/) nie jest częścią tego przebiegu — polecenia
w sgkf/porownanie_cech.py.

Uruchomienie:
    python 3-sgkf-split/uruchom_sgkf.py                 # całość (~35 min na M-series)
    python 3-sgkf-split/uruchom_sgkf.py --czesc A       # tylko przygotowanie (sekundy)
    python 3-sgkf-split/uruchom_sgkf.py --czesc B       # tylko podział i imputacja
    python 3-sgkf-split/uruchom_sgkf.py --bez-mice      # bez kroków B3–B5
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ETAP = Path(__file__).resolve().parent

KROKI = [
    # (część, nazwa logu, katalog, polecenie, ciężki)
    ("A", "A1_przygotuj_dane", "przygotowanie", ["przygotuj_dane.py"], False),
    ("A", "A2_testy", "przygotowanie", ["tests/test_przygotowanie.py"], False),
    ("A", "A3_analiza_decyzji", "przygotowanie", ["analiza_decyzji.py"], False),
    ("A", "A4_analiza_selekcji_cech", "przygotowanie", ["analiza_selekcji_cech.py"], False),
    ("B", "B1_podzial", "sgkf", ["podzial.py", "--porownanie"], False),
    ("B", "B2_testy", "sgkf", ["tests/test_podzial.py"], False),
    ("B", "B3_mice_38", "sgkf", ["aneurysm_sgkf_mice_pipeline.py", "--zapisz"], True),
    ("B", "B4_ocena_imputacji", "sgkf", ["ocena_imputacji.py"], True),
    ("B", "B5_analiza_wynikow", "sgkf", ["analiza_wynikow.py"], True),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Etap 3: przygotowanie danych (A) i SGKF (B)")
    ap.add_argument("--czesc", choices=["A", "B"], help="uruchom tylko jedną część")
    ap.add_argument("--bez-mice", action="store_true", help="pomiń imputację, jej ocenę i analizę wyników (B3–B5)")
    args = ap.parse_args()

    start = time.time()
    for czesc, nazwa, katalog, polecenie, ciezki in KROKI:
        if (args.czesc and czesc != args.czesc) or (ciezki and args.bez_mice):
            continue
        logi = ETAP / katalog / "results" / "logi"
        logi.mkdir(parents=True, exist_ok=True)
        log = logi / f"{nazwa}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {nazwa}: {katalog}/{' '.join(polecenie)}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            wynik = subprocess.run([sys.executable, "-u", str(ETAP / katalog / polecenie[0]), *polecenie[1:]],
                                   cwd=ETAP.parent, stdout=f, stderr=subprocess.STDOUT)
        print(f"    {'OK' if wynik.returncode == 0 else 'BŁĄD'} po {time.time() - t0:.0f} s -> "
              f"{log.relative_to(ETAP.parent)}", flush=True)
        if wynik.returncode != 0:
            print(log.read_text(encoding="utf-8")[-2000:])
            sys.exit(wynik.returncode)
    print(f"\nGotowe w {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
