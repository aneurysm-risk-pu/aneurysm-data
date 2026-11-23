# Dane syntetyczne

Sztuczne dane o tej samej budowie co prawdziwe. Służą do uruchamiania i sprawdzania kodu (także kodu pisanego z pomocą AI) bez dostępu do danych pacjentów. Żaden wiersz nie opisuje prawdziwej osoby.

## Jak wygenerować

```bash
python narzedzia/dane_syntetyczne/generuj.py
```

Wynik trafia do `data_syntetyczne/` i ma taki sam układ jak `data/`. **Wersja w repo** (tryb `realny`, pełna skala, ziarno 42) jest wspólna dla całego zespołu, więc wszyscy pracują na tych samych plikach. Generatora nie trzeba uruchamiać. Jeśli potrzebny jest inny tryb albo inne ziarno, zapisz wynik do innego folderu (`--wyjscie`), np. `data_syntetyczne_brak/`, i nie commituj go.

| Plik | Odpowiada prawdziwemu | Do czego |
|---|---|---|
| `data_syntetyczne/raw/kor_merged_aggregated_1W_mean.csv`, `neuro_...` | `data/raw/` | analizy danych surowych: nazwy badań, rozpoznanie, wartości ujemne |
| `data_syntetyczne/imputation-inputs/aneurysm_concatted.csv` | `data/imputation-inputs/` | wejście `3-sgkf-split/przygotowanie/przygotuj_dane.py` |
| `data_syntetyczne/processed/aneurysm_sgkf_input.csv` | `data/processed/` | podział SGKF, imputacja, kontrole, modelowanie |
| `data_syntetyczne/OPIS.json` | — | tryb, ziarno, liczby, zasiane przypadki („klucz odpowiedzi”) |

Pełny rozmiar powstaje w około minutę. Opcje:
- `--skala 0.1`: mały zbiór do szybkich testów,
- `--ziarno 7`: inne losowanie,
- `--wyjscie folder`: inny folder wynikowy,
- `--tryb`: rodzaj zależności między etykietą a danymi (niżej).

## Tryby

| Tryb | Co to znaczy | Czego się spodziewać |
|---|---|---|
| `realny` (domyślny) | każda kohorta ma własne rozkłady wartości, braków, dat i liczby badań | te same skróty co w prawdziwych danych (braki, okres, liczba badań) |
| `brak` | etykieta nie zależy od niczego | każdy model powinien dać wynik losowy; jeśli nie, w kodzie jest wyciek |
| `wbudowany` | jak `brak`, ale chorzy mają znane przesunięcie kilku cech (lista w `OPIS.json`) | kod powinien odnaleźć właśnie te cechy |

## Jak użyć w kodzie

Skrypty w repo mają ścieżki do `data/` wpisane na stałe, np. `ZRODLO` w `przygotuj_dane.py` i `WEJSCIE` w `podzial.py`. Żeby pracować na danych syntetycznych, wystarczy podmienić ścieżkę z `data` na `data_syntetyczne`.

**Uwaga:** skrypty etapu 3 zapisują wyniki do swoich folderów `results/`. Policzone na danych syntetycznych nadpiszą prawdziwe wyniki, w tym zamrożony `pacjent_fold.csv`. Dlatego takich zmian nie commitujemy. Po pracy przywracamy pliki poleceniem `git checkout -- 3-sgkf-split/`, a najlepiej pracujemy na osobnej kopii repo.

## Na ile dane przypominają prawdziwe

Zgodne są nagłówki plików, formaty zapisu (słowniki `{'values': [...]}`, jednostki, normy, BOM, kodowanie płci), liczby pacjentów i rekordów, odsetek chorych, rozkład liczby badań na pacjenta, okresy badań, wzorce braków, kwantyle wartości i korelacje między cechami. Na pełnej skali wejście SGKF ma około 78 tys. rekordów, 40,6 tys. pacjentów i 2 tys. chorych (prawdziwe: 78 tys., 40,9 tys., 1,8 tys.).

Nie są zgodne zależności bardziej złożone niż korelacje parami ani dokładne odsetki braków (różnice do kilku punktów procentowych). Etap 1 jest uproszczony (opis w nagłówku `generuj.py`), więc liczby z etapu 1 nie odpowiadają prawdziwym. **Wyników z danych syntetycznych nie wolno raportować jako wyników o pacjentach.**

## Zasiane przypadki

Do sprawdzania, czy kod je wykrywa. Szczegóły i identyfikatory są w `OPIS.json`, w polu `klucz_odpowiedzi`:
- pacjenci obecni w obu kohortach,
- wartości niemożliwe z reguł `przygotuj_dane.py`,
- kolizja `custom_id` (2017-01-01 i 2017-12-31),
- wiek poza 18–100 (w tym 222 lata), braki wieku i płci,
- wpisy nieliczbowe (`<0.4`, `>160`),
- pusta kolumna `diagnosis_id`.

Wartości ujemnych nie ma, tak jak w prawdziwych danych.

## Skąd generator bierze rozkłady

Z pliku `parametry.json`. Są w nim wyłącznie zbiorcze statystyki: kwantyle 0,1–99,9%, odsetki, korelacje, wzorce braków występujące w co najmniej 20 rekordach i nazwy badań zliczone według roku. Nie ma w nim wierszy, identyfikatorów ani dat konkretnych badań. Plik powstaje ze skryptu `wyciagnij_parametry.py`, który wymaga prawdziwych danych.
