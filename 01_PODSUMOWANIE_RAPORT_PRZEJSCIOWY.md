# Podsumowanie prac do raportu przejściowego

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zakres:** wszystko, co weszło do raportu przejściowego z 12.06.2026 — od danych surowych po wstępny podział StratifiedGroupKFold
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)

Dokument spina w jedno miejsce cztery etapy opisane dotąd w osobnych raportach cząstkowych. Dalsze prace są rozpisane w `02_PLAN_PRZED_MODELOWANIEM.md` i `03_PLAN_MODELOWANIE.md`.

---

## 1. Dane wejściowe i eksploracja

Projekt operuje na dwóch kohortach źródłowych. Nie są one całkowicie rozłączne: w finalnym zbiorze 63 pacjentów ma rekordy pochodzące z obu kohort.

| Kohorta | Surowy plik | Wiersze × kolumny | Znaczenie |
|---|---|---:|---|
| KOR | `data/raw/kor_merged_aggregated_1W_mean.csv` | 73 679 × 128 | populacja ogólna — klasa **nieoznaczona** |
| NEURO | `data/raw/neuro_merged_aggregated_1W_mean.csv` | 7 655 × 128 | pacjenci oddziału neurologicznego — klasa **pozytywna** |

Wartości 73 276 / 7 611 występujące w materiale do raportu przejściowego dotyczyły pośredniego etapu filtrowania, którego osobny artefakt nie jest obecnie przechowywany w repozytorium. Powyższa tabela podaje liczby możliwe do odtworzenia bezpośrednio z aktualnych plików surowych.

Rekord to jeden tydzień zagregowanych (uśrednionych) wyników laboratoryjnych pacjenta, klucz `{patient_id}-{rok}-W{tydzień}`.

**Braki danych.** W analizie wykonanej do raportu przejściowego raportowano dla KOR 338 683 brakujące wartości, 71,12% pacjentów z co najmniej jednym brakiem oraz 1 793 pacjentów z brakami przekraczającymi 60% profilu. Te liczby dotyczą historycznego etapu pośredniego i nie dają się obecnie odtworzyć jeden do jednego z zachowanych CSV. Dla finalnego wejścia do modelowania, które jest artefaktem referencyjnym, liczba braków wynosi 377 236 z 2 736 895 komórek cech, czyli 13,8%.

**Wartości odstające.** Parametry numeryczne przebadano regułą rozstępu ćwiartkowego (Q1 − 1,5·IQR, Q3 + 1,5·IQR), a wyniki zweryfikowano wizualnie histogramami i wykresami pudełkowymi. Zbudowano binarną macierz anomalii. Flaga IQR oznacza obserwację statystycznie nietypową, nie automatycznie błąd medyczny; część wartości skrajnych nadal wymaga potwierdzenia jednostek i oceny klinicznej.

**Korelacje i selekcja cech.** Zależność każdej cechy z etykietą źródłową P-vs-U zbadano miarami Pearsona, Spearmana, Kendalla i Phi-K. CRP, MONO i %MONO należały do najsłabiej związanych z etykietą i zostały usunięte. Nie należy interpretować tych korelacji jako związku z chorobą, ponieważ `label=0` oznacza grupę nieoznaczoną, a nie potwierdzonych zdrowych. Selekcję wykonano na całym zbiorze z użyciem etykiety, dlatego przed analizą potwierdzającą trzeba albo wrócić do 38 cech, albo umieścić selekcję wewnątrz walidacji.

## 2. Czyszczenie i przygotowanie danych

Ujednolicono struktury obu baz: usunięto kolumny z zakresami referencyjnymi i normami oraz odrzucono zmienne czysto techniczne (`diagnosis_id`, `examination_type`, `description_type`). Część przydatnych kolumn była zapisana w formacie `{'values': [...]}` — błąd agregacji tygodniowej — i została uśredniona, dokańczając zamierzoną operację. Nie można jednak uznać harmonizacji jednostek za zamkniętą: rozkład KREA wskazuje na możliwe współwystępowanie mg/dl i µmol/l, co wymaga weryfikacji w danych źródłowych.

Dalej zastosowano:

- filtrację demograficzną do wieku **18–100 lat**,
- dwukierunkowe odcięcie przy progu **60% braków** — według materiałów sprintowych usunięto 1 793 pacjentów KOR oraz najbardziej niekompletne kolumny; dokładny artefakt pośredni nie jest zachowany,
- usunięcie części najstarszych pomiarów u pacjentów z długim profilem hospitalizacji; notatki sprintowe wskazują odcięcia na 11. i 38. percentylu, ale decyzja nie ma obecnie samodzielnego, wykonywalnego testu odtwarzającego,
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

**Decyzja finalna:** `IterativeImputer` z estymatorem ExtraTrees, nazywany dalej skrótowo MICE, ze wspólnymi parametrami dla obu zbiorów:

```
max_iter = 7,  n_estimators = 78,  max_depth = 10,  min_value = 0.0,  max_value = 1.0
```

MissForest pozostaje sensowną alternatywą dla samego KOR, gdyby priorytetem było wyłącznie minimalizowanie RMSE bez oglądania się na czas. Dla jednego spójnego pipeline'u KOR+NEURO wybrano MICE.

Technicznie jest to pojedyncza, deterministyczna imputacja iteracyjna (`sample_posterior=False`), a nie klasyczne multiple imputation generujące wiele kompletnych zbiorów i propagujące niepewność imputacji.

**Imputacja pełnego zbioru:** 377 236 braków (13,8% komórek), scaler dopasowany na 23 092 kompletnych wierszach referencyjnych, czas 23,4 min. Wynik: **0 pozostałych NaN i 0 wartości ujemnych**. Wybrane podsumowania korelacji zmieniły się umiarkowanie (norma Frobeniusa 1,21–1,23 dla ALL/KOR), co nie jest jednak dowodem braku biasu imputacji.

Globalnie zaimputowany plik był artefaktem walidacji metody. Nie jest poprawnym wejściem do treningu ani oceny modeli, ponieważ imputacja została wykonana przed podziałem danych.

## 4. Kontrolowany podział danych

Prosty podział losowy był wykluczony, bo jeden pacjent ma zwykle więcej niż jeden rekord — część jego pomiarów trafiłaby do treningu, część do walidacji, a model byłby oceniany na osobie, którą już widział.

Zaimplementowano **StratifiedGroupKFold** na 5 foldach: mechanizm *Group* pilnuje, by wszystkie rekordy jednego `patient_id` trafiły wyłącznie do jednej części podziału, a *Stratified* dąży do zbliżonych proporcji klas. Asercja w kodzie wymusza zero wspólnych identyfikatorów pacjentów między train i test.

Podział zintegrowano z imputacją tak, aby MICE działał **wewnątrz** foldu: scaler dopasowywany wyłącznie na kompletnych obserwacjach z części treningowej, imputer uczony tylko na treningu, walidacja jedynie transformowana, na końcu odwrotna transformacja skalera. Dzięki temu żadna informacja ze zbioru walidacyjnego nie wpływa ani na skalowanie, ani na uzupełnianie braków.

Ten podział nie jest jeszcze zamrożonym protokołem modelowania. Aktualna implementacja używa `shuffle=False`, więc foldy zależą od kolejności wierszy, nie zapisuje przydziału pacjentów na dysk i nie rozwiązuje 63 pacjentów z obiema etykietami. Są to zadania etapu 0 z `02_PLAN_PRZED_MODELOWANIEM.md`.

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
5. Kohorty są silnie rozdzielone kalendarzowo: 99,5% rekordów KOR pochodzi z lat 2020–2021, a tylko 10,2% rekordów NEURO mieści się w dokładnym oknie dat KOR. Efekt czasu, laboratorium i sposobu pozyskania danych może być mylony z efektem klinicznym.
6. Brak daty rozpoznania tętniaka uniemożliwia potwierdzenie, że cechy NEURO pochodzą sprzed diagnozy i leczenia.
7. 63 pacjentów występuje w obu kohortach źródłowych; ich interpretacja i etykieta pacjentowa nie zostały jeszcze zamrożone.

Do tej listy doszły później ustalenia z weryfikacji kodu i diagnostyki, opisane w `02_PLAN_PRZED_MODELOWANIEM.md` oraz `4-pu-setup/ETAP0_USTALENIA.md`: brak faktycznego tasowania w `StratifiedGroupKFold`, selekcja cech prowadzona z użyciem etykiety na całym zbiorze oraz rozjazd czasowy kohort.

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
