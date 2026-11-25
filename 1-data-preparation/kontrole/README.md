# Kontrole etapu 1 (Sprint 6, zadania #43–#46)

Każdy skrypt odpowiada na jedno zadanie, liczy tylko na plikach z `data/` i zapisuje wynik w `results/`. Niczego w danych nie zmienia.

| Zadanie | Skrypt | Wynik |
|---|---|---|
| #43 wartości ujemne i niemożliwe | `issue43_wartosci_niemozliwe.py` | `results/issue43_wartosci_niemozliwe.json` |
| #44 skąd 38 cech | `issue44_kolumny_60proc.py` | `results/issue44_kolumny_60proc.json` |
| #45 zmiany w czasie | `issue45_zmiany_w_czasie.py` | `results/issue45_*.json`, `*.csv` |
| #46 czyszczenie a różnice grup | `issue46_czyszczenie_a_grupy.py` | `results/issue46_czyszczenie_a_grupy.json` |

Uruchomienie z katalogu głównego repo, np.:

```bash
python 1-data-preparation/kontrole/issue44_kolumny_60proc.py
```

Droga danych, na której liczą skrypty (opis w `wspolne.py`):

dane surowe → `*_cleaning.csv` (krok 1) → `*_shortend.csv` (krok 2) → `aneurysm_concatted.csv` (krok 3) → `aneurysm_sgkf_input.csv` (krok 4)

Skrypty potrzebują plików pośrednich z `data/interim/` i `data/imputation-inputs/`. W `data_syntetyczne/` tych plików nie ma, więc działają tylko na prawdziwych danych.
