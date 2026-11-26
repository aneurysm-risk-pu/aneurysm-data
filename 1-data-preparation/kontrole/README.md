# Kontrole preprocessingu (Sprint 6, zadania #43–#46)

Kontrole sprawdzają preprocessing Liwii: `1-data-preparation/scripts-lf/final_preprocessing.py` (dane surowe → `data/processed/{kor,neuro}_preprocessed_final.csv` → `concatted_preprocessed.csv`).

`wspolne.py` → `przebieg()` powtarza kroki jej skryptu w tej samej kolejności, korzystając z jej funkcji, i zwraca stan danych po każdym kroku. Na końcu sprawdza, że wynik jest identyczny z zapisanymi plikami, więc liczby z kontroli dotyczą dokładnie tych danych.

| Zadanie | Skrypt | Wynik |
|---|---|---|
| #43 wartości ujemne i niemożliwe | `issue43_wartosci_niemozliwe.py` | `results/issue43_wartosci_niemozliwe.json` |
| #44 skąd 38 cech | `issue44_kolumny_60proc.py` | `results/issue44_kolumny_60proc.json` |
| #45 zmiany w czasie | `issue45_zmiany_w_czasie.py` | `results/issue45_*.json`, `*.csv` |
| #46 preprocessing a różnice grup | `issue46_czyszczenie_a_grupy.py` | `results/issue46_czyszczenie_a_grupy.json` |

Uruchomienie z katalogu głównego repo, np.:

```bash
python 1-data-preparation/kontrole/issue46_czyszczenie_a_grupy.py
```

Kontrole działają na Pythonie 3.9. Skrypt Liwii używa składni z Pythona 3.10, więc `wspolne.py` wczytuje go z odroczonymi adnotacjami, bez zmian w samym pliku. Kontrole dla poprzedniego preprocessingu (notebooki z etapu 1) są w historii gita, w commicie `b001155`.
