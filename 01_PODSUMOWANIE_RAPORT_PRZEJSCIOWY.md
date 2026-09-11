# Podsumowanie prac do raportu przejściowego

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zakres:** wszystko, co weszło do raportu przejściowego z 12.06.2026 — od danych surowych po wstępny podział StratifiedGroupKFold
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)

Dokument spina w jedno miejsce cztery etapy opisane dotąd w osobnych raportach cząstkowych. Dalsze prace są rozpisane w `02_PLAN_PRZED_MODELOWANIEM.md` i `03_PLAN_MODELOWANIE.md`.

---

## 1. Dane wejściowe i eksploracja

Projekt operuje na dwóch niezależnych kohortach:

| Kohorta | Wiersze | Cechy | Znaczenie |
|---|---|---|---|
| KOR | 73 276 | 45 | populacja ogólna — klasa **nieoznaczona** |
| NEURO | 7 611 | 45 | pacjenci oddziału neurologicznego — klasa **pozytywna** |

Rekord to jeden tydzień zagregowanych (uśrednionych) wyników laboratoryjnych pacjenta, klucz `{patient_id}-{rok}-W{tydzień}`.

**Braki danych.** W KOR zidentyfikowano 338 683 brakujących wartości; **71,12%** pacjentów miało co najmniej jeden brak w profilu laboratoryjnym. Rozkład braków jest nierównomierny — APTT, WAPTT, GLU, INR i PT brakowały w 48,0–51,5% przypadków. Osobno wyodrębniono 1 793 pacjentów KOR, u których braki przekraczały 60% profilu (w skrajnym przypadku 87,8%).

**Wartości odstające.** Parametry numeryczne przebadano regułą rozstępu ćwiartkowego (Q1 − 1,5·IQR, Q3 + 1,5·IQR), a wyniki zweryfikowano wizualnie histogramami i wykresami pudełkowymi. Zbudowano binarną macierz anomalii, która posłużyła do wskazania odchyleń niemożliwych do wyjaśnienia fizjologicznie (np. błędy wpisu laboratoryjnego).

**Korelacje i selekcja cech.** Zależność każdej cechy ze zmienną celu zbadano czterema komplementarnymi miarami: Pearsona, Spearmana, Kendalla i Phi-K. Cechy o najniższej sile związku we wszystkich czterech testach to CRP, MONO i %MONO (dla Phi-K korelacja CRP wyniosła 0,000000) — te trzy zmienne usunięto. Niską siłę wykazały też %MONO, HCT, MPV i patient_age, ale pozostawiono je ze względu na interpretowalność kliniczną i znaczenie demograficzne.

## 2. Czyszczenie i przygotowanie danych

Ujednolicono struktury obu baz: usunięto kolumny z zakresami referencyjnymi i normami, ustandaryzowano jednostki oraz odrzucono zmienne czysto techniczne (`diagnosis_id`, `examination_type`, `description_type`). Część przydatnych kolumn była zapisana w formacie `{'values': [...]}` — błąd agregacji tygodniowej — i została uśredniona, dokańczając zamierzoną operację.

Dalej zastosowano:

- filtrację demograficzną do wieku **18–100 lat**,
- dwukierunkowe odcięcie przy progu **60% braków** — usunięcie 1 793 pacjentów KOR oraz najbardziej niekompletnych kolumn,
- usunięcie najstarszych historycznie pomiarów u pacjentów z długim profilem hospitalizacji (odcięcia na 11. i 38. percentylu dla wybranych podgrup), żeby dawna historia medyczna nie zafałszowała aktualnego statusu,
- przypisanie etykiet zgodnie z paradygmatem PU: NEURO → `label = 1`, KOR → `label = 0`.

Po scaleniu powstał jeden zbiór **78 197 wierszy × 42 kolumny**, w tym 38 cech numerycznych, zredukowanych następnie do **35** po usunięciu CRP, MONO i %MONO.

## 3. Imputacja braków

**Protokół oceny.** Z kompletnych wierszy budowano macierz referencyjną, maskowano losowo 10% wartości i porównywano metody na tej samej masce. Kluczowe zabezpieczenie: maska do strojenia hiperparametrów (seed=43) była **oddzielona** od maski ewaluacyjnej (seed=42), więc parametry nie były dopasowane do tych samych braków, na których liczono wynik.

| Zbiór | Wiersze | Complete cases | Udział |
|---|---|---|---|
| KOR | 71 546 | 20 663 | 28,9% |
| NEURO | 6 651 | 780 | 11,7% |

**Porównanie metod (RMSE, seed=42):**

| Zbiór | MissForest | MICE | KNN |
|---|---|---|---|
| KOR | **0,0783** | 0,0865 | 0,1022 |
| NEURO | 0,1017 | **0,0978** | 0,1291 |

Żadna metoda nie dominowała na obu zbiorach — MissForest korzysta z dużej liczby kompletnych obserwacji w KOR, MICE lepiej radzi sobie na małym NEURO. KNN, użyty jako metoda bazowa, konsekwentnie wypadał najgorzej.

**Weryfikacja stabilności.** Powtórzenie ewaluacji na niezależnej masce seed=7 nie zmieniło rankingu żadnej metody. Walidacja multi-seed dla MICE dała KOR: RMSE 0,0865 ±0,0007, NEURO: 0,0950 ±0,0025 — seed=42 nie był przypadkiem odstającym. Test cross-param wykazał, że parametry dobrane na KOR pogarszają wynik NEURO o 0,00001 RMSE, będąc przy tym ok. 7× szybsze.

**Decyzja finalna:** MICE z estymatorem ExtraTrees, wspólne parametry dla obu zbiorów:

```
max_iter = 7,  n_estimators = 78,  max_depth = 10,  min_value = 0.0,  max_value = 1.0
```

MissForest pozostaje sensowną alternatywą dla samego KOR, gdyby priorytetem było wyłącznie minimalizowanie RMSE bez oglądania się na czas. Dla jednego spójnego pipeline'u KOR+NEURO wybrano MICE.

**Imputacja pełnego zbioru:** 377 236 braków (13,8% komórek), scaler dopasowany na 23 092 kompletnych wierszach referencyjnych, czas 23,4 min. Wynik: **0 pozostałych NaN i 0 wartości ujemnych**. Struktura korelacji została zachowana (Frobenius 1,21–1,23 dla ALL/KOR).

## 4. Kontrolowany podział danych

Prosty podział losowy był wykluczony, bo jeden pacjent ma zwykle więcej niż jeden rekord — część jego pomiarów trafiłaby do treningu, część do walidacji, a model byłby oceniany na osobie, którą już widział.

Zastosowano **StratifiedGroupKFold** na 5 foldach: mechanizm *Group* pilnuje, by wszystkie rekordy jednego `patient_id` trafiły wyłącznie do jednej części podziału, a *Stratified* utrzymuje zbliżone proporcje klas. Warunkiem poprawności było zero wspólnych identyfikatorów pacjentów między train i test w każdym foldzie — i ten warunek jest spełniony.

Podział zintegrowano z imputacją tak, aby MICE działał **wewnątrz** foldu: scaler dopasowywany wyłącznie na kompletnych obserwacjach z części treningowej, imputer uczony tylko na treningu, walidacja jedynie transformowana, na końcu odwrotna transformacja skalera. Dzięki temu żadna informacja ze zbioru walidacyjnego nie wpływa ani na skalowanie, ani na uzupełnianie braków.

## 5. Stan danych na koniec etapu

Liczby przeliczone bezpośrednio na pliku `data/processed/aneurysm_concatted_cleaned.csv`:

| Co | Wartość |
|---|---|
| Wiersze × kolumny | 78 197 × 39 (35 cech + 4 kolumny meta) |
| `label = 0` | 71 546 wierszy / 39 164 pacjentów |
| `label = 1` | 6 651 wierszy / 1 823 pacjentów |
| Unikalnych pacjentów | 40 924 |
| Rekordów na pacjenta | min 1, mediana 1, średnia 1,91, maks. 38 |

> **Drobna rozbieżność do poprawienia w raporcie końcowym.** Raport przejściowy podaje podział klas jako 71 544 / 6 653, a faktyczny stan pliku to **71 546 / 6 651**. Suma się zgadza (78 197), więc chodzi o pomyłkę w przepisaniu, nie o inny zbiór danych.

## 6. Ograniczenia odnotowane na tym etapie

1. Benchmark imputacji opiera się na kompletnych wierszach, a te mogą różnić się klinicznie od rekordów z brakami.
2. Test Kolmogorowa-Smirnowa porównywał kompletne wiersze z pełnym zbiorem po imputacji — przy tej liczbie rekordów wykrywa nawet minimalne różnice, więc nie jest samodzielną miarą jakości.
3. Dla NEURO liczba kompletnych wierszy była mała (780), stąd większa zmienność wyników; dlatego wykonano testy stabilności i walidację multi-seed.
4. Nie zbadano mechanizmu powstawania braków (MCAR / MAR / MNAR) ani wpływu samej imputacji na końcowe modele PU.

Do tej listy doszły później dwa ustalenia z weryfikacji kodu, opisane w `02_PLAN_PRZED_MODELOWANIEM.md`: brak faktycznego tasowania w `StratifiedGroupKFold` oraz selekcja cech prowadzona z użyciem etykiety na całym zbiorze.

---

## Gdzie co leży w repozytorium

Po reorganizacji struktury:

| Etap | Katalog | Kluczowe pliki |
|---|---|---|
| Czyszczenie i EDA | `1-data-preparation/` | `scripts-lk/`, `scripts-lf/`, `cleaning_summary.md` |
| Imputacja | `2-imputation/` | `RAPORT_IMPUTACJA.md`, `final/`, `mice-lk/`, `missforest-dm/`, `knn-am/` |
| Podział SGKF | `3-sgkf-split/` | `RAPORT_SGKF_MICE.md`, `aneurysm_sgkf_mice_pipeline.py` |
| Dane | `data/` | `raw/` → `interim/` → `imputation-inputs/` → `processed/` |
| Archiwum | `docs/` | `sprints/`, `reports/` |

Wejściem do dalszego modelowania jest **`data/processed/aneurysm_concatted_cleaned.csv`** — zbiór **przed** imputacją, bo imputacja wykonywana jest wewnątrz foldów. Plik `2-imputation/final/results/aneurysm_imputed_cleaned.csv` (imputowany globalnie, przed podziałem) służy wyłącznie do walidacji samej imputacji i **nie może** być używany do trenowania ani oceny modeli.
