# Realizacja projektu: od danych surowych do zamrożonego podziału SGKF

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)
**Stan na:** 04.10.2026 (branch `pu-modeling-setup-lk`)
**Zakres:** etapy 1–3: przygotowanie danych, imputacja braków, przygotowanie danych do podziału i kontrolowany podział z imputacją w foldach (SGKF). Propozycja dalszych prac (modelowanie PU) jest w `4-pu-setup/PLAN_MODELOWANIA.md`; plan obowiązujący zostanie ustalony z prowadzącym.

Dokument zbiera w jednym miejscu, **co zostało zrobione, przez kogo, na jakiej podstawie i z jakim wynikiem**, do momentu, w którym dane są gotowe do modelowania. Jest punktem wejścia; szczegóły techniczne zostają w raportach etapów, do których prowadzą odnośniki. Każda liczba pochodzi z plików i skryptów w repozytorium; przy decyzjach z 04.10.2026 wskazany jest skrypt, który ją liczy.

---

## Streszczenie

1. **Dane są przygotowane, oczyszczone i gotowe do modelowania:** 78 052 rekordy tygodniowe, 40 924 pacjentów, 1 823 (4,45%) z potwierdzonym tętniakiem (NEURO, klasa P), reszta to populacja ogólna (KOR, klasa U). Wejście do podziału ma 38 cech i jest **przed imputacją**.
2. **Imputacja jest wybrana na podstawie wspólnego benchmarku trzech metod** (KNN, MissForest, MICE) z oddzieloną maską do strojenia i maską do oceny. Wybrano MICE z estymatorem ExtraTrees, który jest wykonywany **wewnątrz każdego foldu**.
3. **04.10.2026 zapadły decyzje dotyczące danych** i zostały wdrożone osobnym, przetestowanym kodem z pełnym spisem zmian. Dotyczą 63 pacjentów obecnych w obu kohortach i 752 wartości niemożliwych fizjologicznie (KREA, K, Na). Rozjazd czasowy kohort i asymetria braków zostają bez korekty, jako opisane ograniczenia (sekcja 6).
4. **Podział jest zamrożony i zapisany:** 5 foldów po pacjentach, zero wspólnych pacjentów między train i test, stratyfikacja po etykiecie, epoce badania i liczbie rekordów. Rozstęp udziału pozytywnych między foldami wynosi 0,01 pp. Wersja z raportu przejściowego była nietasowana, niezapisana i zależna od wersji biblioteki (sekcja 5).
5. **Imputacja w foldach jest wykonana w pełni** dla wariantu głównego (38 cech) i wariantu wrażliwości (35 cech), z kontrolą braku wycieku w każdym foldzie (sekcja 5.5).
6. **Najpoważniejsze ryzyka leżą w danych, nie w kodzie:**
   - kohorty niemal nie pokrywają się w czasie,
   - wyniki NEURO najpewniej pochodzą z hospitalizacji,
   - MICE uzupełnia 26% komórek NEURO wobec 12% w KOR.

   Te ryzyka idą do ograniczeń raportu (sekcja 9).

---

## 1. Zespół i podział pracy

### 1.1 Kto za co odpowiadał

| Osoba | Główne obszary | Kluczowe artefakty |
|---|---|---|
| **Łukasz Kubik** (kierownik) | organizacja repozytorium i sprintów; skrypty czyszczące i kontrolne; imputacja MICE; **wspólny benchmark imputacji** i imputacja finalna; integracja podziału z imputacją w foldzie; weryfikacja metodologiczna po raporcie przejściowym; **decyzje dotyczące danych i ich wdrożenie (04.10)**; audyt i wersja 2 podziału | `1-data-preparation/scripts-lk/`, `2-imputation/mice-lk/`, `2-imputation/final/`, `2-imputation/RAPORT_IMPUTACJA.md`, `3-sgkf-split/`, `4-pu-setup/` |
| **Liwia Florkiewicz** | pierwsze przetworzenie danych (rozpakowanie słowników, jednostki, mapowanie badań); analiza populacyjna i wartości odstających; filtracja (wiek, płeć, próg 60% braków, ograniczenie nadreprezentowanych pacjentów); **połączenie KOR + NEURO ze zmienną celu**; analiza korelacji i selekcja cech; **pierwsza implementacja StratifiedGroupKFold**; notatki ze spotkań | `1-data-preparation/data_analysis_v1.ipynb`, `1-data-preparation/scripts-lf/`, `data/imputation-inputs/aneurysm_concatted.csv`, `3-sgkf-split/archiwum/aneurysm_data_StratifiedGroupKFold.ipynb`, `docs/sprints/` |
| **Adrian Meredyk** | imputacja **KNN**: implementacja, przegląd siatki `k` × wagi, raporty wyników dla KOR i NEURO | `2-imputation/knn-am/` |
| **Dominika Malisz** | imputacja **MissForest**: implementacja, dobór liczby drzew z kryterium zatrzymania, raport | `2-imputation/missforest-dm/` |

### 1.2 Macierz wkładu według etapów

● wkład główny ○ wkład uzupełniający

| Etap / zadanie | Łukasz | Liwia | Adrian | Dominika |
|---|:---:|:---:|:---:|:---:|
| Organizacja, plan sprintów, struktura repo | ● | ○ | | |
| Czyszczenie strukturalne (słowniki `{'values': […]}`, jednostki, duplikaty) | ● | ● | | |
| Kontrole anomalii medycznych (WBC, Na, K, KREA, PLT) | ● | ○ | | |
| Analiza populacyjna, wartości odstające (IQR, kwantyle, wykresy) | | ● | | |
| Filtracja (wiek 18–100, płeć, 60% braków, nadreprezentowani pacjenci) | | ● | | |
| Połączenie kohort, etykieta P/U | | ● | | |
| Korelacje z etykietą, selekcja cech | | ● | | |
| Imputacja: KNN | ○ | | ● | |
| Imputacja: MissForest | ○ | | | ● |
| Imputacja: MICE | ● | | | |
| Wspólny benchmark imputacji (Optuna, maski, stabilność) i imputacja finalna | ● | | | |
| Podział StratifiedGroupKFold (v1) | ○ | ● | | |
| Integracja podziału z MICE w foldzie | ● | | | |
| Weryfikacja po raporcie przejściowym (diagnostyka, agregacja, stabilność podziału) | ● | | | |
| Decyzje dotyczące danych (04.10) i przygotowanie danych do SGKF | ● | | | |
| Audyt i wersja 2 podziału, pełna imputacja w foldach | ● | | | |

Udział mierzony liczbą commitów niedoszacowuje pracę w notebookach. Analiza populacyjna i scalenie danych to kilka dużych commitów, a nie wiele małych. Tabela opiera się na zawartości zmian, nie na ich liczbie.

### 1.3 Oś czasu

| Data | Co | Kto |
|---|---|---|
| 13.04.2026 | repozytorium, dane surowe | Łukasz |
| 23.04 | spotkanie: zakres czyszczenia, plan benchmarku imputacji; pierwsze przetworzenie danych; czyszczenie `{'values': […]}` i duplikatu w NEURO | Liwia, Łukasz |
| 26.04 | czyszczenie pełne NEURO i KOR (`clean_full.py`) | Łukasz |
| 05–07.05 | KNN; analiza populacyjna i odstających; skrypty kontrolne, pierwszy test imputacji, raport MICE | Adrian, Liwia, Łukasz |
| 07.05 | spotkanie: decyzje o filtrach (wiek 18–100, 60% braków, percentyle 11/38) i o zbiorze referencyjnym do benchmarku | zespół |
| 18–28.05 | MissForest; dane przefiltrowane (`*_shortend.csv`); rozbudowa KNN; połączenie KOR + NEURO z etykietą | Dominika, Liwia, Adrian |
| 31.05–06.06 | wspólny benchmark imputacji (Optuna, 4 przebiegi), walidacja, imputacja finalna | Łukasz |
| 08.06 | analiza korelacji i usunięcie CRP, MONO, %MONO; pierwszy podział SGKF | Liwia |
| 10–11.06 | integracja: SGKF + MICE w foldzie, raport metodologiczny | Łukasz |
| **12.06** | **raport przejściowy** | zespół |
| 11–20.09 | plan modelowania PU, reorganizacja repo, diagnostyka danych, recenzja metodologiczna, pytania do prowadzącego, analiza reguły agregacji, siatka stabilności podziału | Łukasz |
| 25.09 | prototyp infrastruktury PU (wnioski w `4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 6) | Łukasz |
| **04.10** | **decyzje dotyczące danych; przygotowanie danych do SGKF; podział v2; pełna imputacja w foldach (38 i 35 cech); to podsumowanie** | Łukasz |

---

## 2. Dane wejściowe

| Kohorta | Plik surowy | Wiersze × kolumny | Rola w PU |
|---|---|---:|---|
| KOR | `data/raw/kor_merged_aggregated_1W_mean.csv` | 73 679 × 128 | populacja ogólna: **nieoznaczeni (U)** |
| NEURO | `data/raw/neuro_merged_aggregated_1W_mean.csv` | 7 655 × 128 | pacjenci z rozpoznanym tętniakiem: **pozytywni (P)** |

Rekord to tydzień uśrednionych wyników laboratoryjnych pacjenta, z kluczem `{patient_id}-{rok}-W{tydzień}`. Cechy obejmują morfologię z rozmazem, biochemię (KREA, Na, K, GLU, eGFR), koagulologię (PT, INR, APTT, WAPTT), CRP, wiek i płeć.

**Dlaczego PU, a nie zwykła klasyfikacja.** Nikt nie przebadał obrazowo pacjentów KOR, więc wśród nich mogą być osoby z niewykrytym tętniakiem. `label = 0` znaczy „nieoznaczony”, a nie „zdrowy”. Ta jedna obserwacja wyznacza większość decyzji metodologicznych w kolejnych etapach.

---

## 3. Etap 1: przygotowanie danych

**Raporty:** `1-data-preparation/cleaning_summary.md`, `docs/sprints/DATA_ANALYSIS_SPRINT1.md`, `docs/sprints/` (notatki ze spotkań)

### 3.1 Czyszczenie strukturalne (Liwia, Łukasz)

| Problem | Rozwiązanie | Uzasadnienie |
|---|---|---|
| Kolumny w formacie `{'values': […]}` | wartości liczbowe uśrednione; kolumny tekstowe (`examination_type`, `descriptive_result`) usunięte | plik jest agregatem tygodniowym `_1W_mean`, więc lista to niedokończona agregacja, a średnia ją dokańcza |
| Kolumny zakresów referencyjnych i jednostek | usunięte | norma nie jest cechą pacjenta; zespół zdecydował też, żeby nie uzupełniać norm |
| Zmienne techniczne (`diagnosis_id`, `description_type`) | usunięte | brak treści klinicznej |
| Duplikat `custom_id = 194143-2017-W52` | zachowany rekord nowszy | ten sam pacjent i tydzień; różnica wieku 71/72 wynika ze sposobu liczenia |
| Płeć: `"0.0"/"1.0"` w KOR vs `"0"/"1"` w NEURO | ujednolicona do 0/1 | — |

### 3.2 Anomalie medyczne (Łukasz: skrypty kontrolne; zespół: decyzje)

`check_med_data.py` znalazł wartości biologicznie niemożliwe: Na < 100 mmol/l, K > 9 mmol/l (do 39), WBC > 100 G/l, PLT = 0, wiek 222 lata. Przy kreatyninie postawiono hipotezę błędu jednostek (µmol/l zamiast mg/dl). Zespół zdecydował wtedy **nie przeliczać** wartości, bo KREA > 20 mg/dl bywa realne przy dializie. Sprawę zamknęła reguła oparta na eGFR z 04.10.2026 (sekcja 5.3).

### 3.3 Analiza populacyjna i filtracja (Liwia)

- **Wartości odstające.** Analiza IQR z wykresami i macierzą anomalii. Następnie usuwanie **całych rekordów** poza skrajnymi kwantylami, parametr po parametrze: KOR stracił 585 rekordów (0,8%, kwantyle 0,0001–0,0007), NEURO 228 rekordów (3,2%, kwantyle 0,0002–0,001). Liczby odczytuje z notebooka `3-sgkf-split/przygotowanie/analiza_decyzji.py`, sekcja F. Granice były statystyczne, więc część wartości niemożliwych przetrwała, np. górna granica K w KOR wynosiła 29 mmol/l.
- **Wiek 18–100 lat.** Usunięto osoby niepełnoletnie (brak zgody na badania) i wartości niemożliwe.
- **Próg 60% braków** w dwóch kierunkach: usunięto najbardziej niekompletne kolumny (wspólne dla obu kohort) oraz 1 793 pacjentów KOR z brakami powyżej 60% profilu.
- **Pacjenci nadreprezentowani:** zachowanie najnowszych pomiarów u osób z bardzo długą historią (do 11 rekordów w KOR i 38 w NEURO). Zasada ze spotkania 23.04: pacjentów z jednym pomiarem **nie usuwamy**, bo są najcenniejsi.

### 3.4 Połączenie kohort i selekcja cech (Liwia)

KOR i NEURO połączono w jeden zbiór z etykietą `label` (NEURO = 1, KOR = 0). Wynik: **78 197 rekordów × 42 kolumny** (38 cech + 4 kolumny meta), plik `data/imputation-inputs/aneurysm_concatted.csv`.

Zależność cech z etykietą zbadano czterema miarami: Pearson, Spearman, Kendall, Phi-K. Trzy najsłabiej związane cechy (CRP, MONO, %MONO) usunięto, co dało wariant 35 cech (`data/processed/aneurysm_concatted_cleaned.csv`). Weryfikacja wrześniowa zakwestionowała tę selekcję (sekcja 6, problem 4). **Wariantem głównym jest 38 cech, a 35 to analiza wrażliwości.**

### 3.5 Stan danych na wyjściu etapu 1

| | Rekordy | Pacjenci | Rekordów na pacjenta (średnio) |
|---|---:|---:|---:|
| KOR (`label = 0`) | 71 546 | 39 164 | 1,83 |
| NEURO (`label = 1`) | 6 651 | 1 823 | 3,65 |
| **Razem** | **78 197** | **40 924** | 1,91 (mediana 1, maks. 38) |

> W raporcie przejściowym podano 71 544 / 6 653; faktyczny stan pliku to 71 546 / 6 651. Suma się zgadza, więc to pomyłka przy przepisywaniu, nie inny zbiór.

---

## 4. Etap 2: imputacja braków

**Raporty:** `2-imputation/RAPORT_IMPUTACJA.md` (główny), `2-imputation/mice-lk/README_mice.md`, `2-imputation/knn-am/wyniki_knn_*.md`, `2-imputation/missforest-dm/raport_missforest.md`, `docs/reports/imputation_report_*.md`

### 4.1 Faza indywidualna: trzy metody, trzy osoby

| Metoda | Kto | Co sprawdzono | Wynik cząstkowy |
|---|---|---|---|
| KNN | Adrian | `k` ∈ {1, 3, 5, 7, 11, 15, 21} × wagi {uniform, distance} | KOR: najlepsze `k = 21`, distance, RMSE 0,0993 |
| MissForest | Dominika | 10 → 30 → 70 → 150 drzew, zatrzymanie przy poprawie < 1% | KOR: 30 drzew, RMSE 0,0846; NEURO: 70 drzew, RMSE 0,1053 |
| MICE | Łukasz | estymatory BayesianRidge vs ExtraTrees, przeszukiwanie parametrów, ocena KL | ExtraTrees wyraźnie lepsze |

Każda osoba używała nieco innego protokołu: inny zbiór referencyjny, inne maskowanie, inne metryki. **Wyników z tej fazy nie da się bezpośrednio porównać**, dlatego powstał wspólny benchmark.

### 4.2 Faza wspólna: benchmark (Łukasz)

Na tym samym zbiorze referencyjnym, tymi samymi maskami i tym samym kodem uruchomiono wszystkie trzy metody (`2-imputation/final/`).

**Protokół:**
- zbiór referencyjny to kompletne wiersze (KOR 20 663, czyli 28,9%; NEURO 780, czyli 11,7%), przeskalowane do [0, 1],
- 10% wartości maskowanych losowo; **maska do strojenia (seed 43) jest oddzielna od maski do oceny (seed 42)**,
- strojenie Optuną (TPE): KNN 30 prób, MICE 20–40, MissForest 15,
- metryki: RMSE i MAE na zamaskowanych komórkach, dywergencja KL rozkładów, liczba wartości ujemnych.

**Historia przebiegów:** przebieg 1 odrzucono, bo Optuna i ocena używały tej samej maski, co jest wyciekiem. W przebiegach 2–4 rozszerzano zakresy parametrów, gdy Optuna trafiała w ich granicę.

**Wyniki (RMSE, maska seed 42):**

| Zbiór | MissForest | MICE | KNN |
|---|---:|---:|---:|
| KOR | **0,0783** | 0,0865 | 0,1022 |
| NEURO | 0,1017 | **0,0978** | 0,1291 |

**Stabilność:** na niezależnej masce (seed 7) ranking się nie zmienił. MICE w walidacji multi-seed: KOR 0,0865 ± 0,0007, NEURO 0,0950 ± 0,0025. Parametry dobrane na KOR pogarszają wynik NEURO o 0,00001 RMSE i są przy tym około 7 razy szybsze.

**Decyzja:** `IterativeImputer` z ExtraTrees (nazywany w projekcie MICE), wspólne parametry dla obu kohort: `max_iter = 7, n_estimators = 78, max_depth = 10`.

**Dlaczego nie MissForest, skoro wygrywa na KOR:** żadna metoda nie wygrała na obu zbiorach. MICE wygrywa na mniejszym i trudniejszym NEURO, jest szybsze, a jeden spójny imputer dla obu kohort jest wymagany, bo imputacja odbywa się na połączonym zbiorze wewnątrz foldu.

### 4.3 Imputacja globalna: tylko walidacja metody

Pełny zbiór zaimputowano raz: 377 236 braków, 23,4 min, 0 NaN i 0 wartości ujemnych po imputacji. **Ten plik (`2-imputation/final/results/aneurysm_imputed_cleaned.csv`) nie może być wejściem do modeli**, bo imputacja przed podziałem przenosi informację z części testowej do treningowej. Do modelowania imputujemy wewnątrz foldu (etap 3).

### 4.4 Zastrzeżenia

- Technicznie to pojedyncza, deterministyczna imputacja iteracyjna, a nie *multiple imputation*. Niepewność uzupełnienia nie jest propagowana.
- Benchmark mierzy błąd na kompletnych wierszach, a te mogą różnić się klinicznie od wierszy z brakami. W NEURO kompletnych jest tylko 780.
- Nie zbadano mechanizmu braków (MCAR/MAR/MNAR).

---

## 5. Etap 3: przygotowanie danych do SGKF i kontrolowany podział

**Raport techniczny:** `3-sgkf-split/RAPORT_SGKF_MICE.md` · **cały etap jednym poleceniem:** `python 3-sgkf-split/uruchom_sgkf.py`

Etap ma dwie osobne części: **A** (`3-sgkf-split/przygotowanie/`) przygotowuje dane i zapisuje plik `data/processed/aneurysm_sgkf_input.csv`, a **B** (`3-sgkf-split/sgkf/`) czyta wyłącznie ten plik, zamraża podział i imputuje w foldach.

### 5.1 Wersja 1: stan na raport przejściowy (Liwia, Łukasz)

Liwia zaimplementowała `StratifiedGroupKFold` (5 foldów) w notebooku. Grupowanie po `patient_id` gwarantuje, że wszystkie rekordy pacjenta są po jednej stronie podziału. Łukasz zintegrował podział z imputacją, dzięki czemu w każdym foldzie scaler i MICE uczą się wyłącznie na części treningowej, a część testowa jest tylko transformowana.

**Ta część była poprawna i została zachowana.** Imputacja w foldzie i grupowanie po pacjencie to dwa najważniejsze zabezpieczenia przed wyciekiem.

### 5.2 Audyt wersji 1

| # | Problem | Skutek |
|---|---|---|
| 1 | brak tasowania (`shuffle=False`) | podział zależał od kolejności wierszy w CSV; „ziarno 42” nie miało na niego wpływu |
| 2 | przydział nie był zapisywany | każde uruchomienie mogło dać inne foldy bez śladu |
| 3 | SGKF z tasowaniem zależy od wersji sklearn | ta sama konfiguracja dawała **inne podziały na dwóch komputerach zespołu** (sklearn 1.6 vs 1.8) |
| 4 | stratyfikacja tylko po etykiecie | foldy niewyrównane pod względem epoki badania i liczby rekordów, czyli dwóch znanych źródeł fałszywego sygnału |
| 5 | brak reguły dla 63 pacjentów obecnych w obu kohortach | etykieta pacjenta nieokreślona |

### 5.3 Część A: przygotowanie danych do SGKF (decyzje z 04.10.2026)

| Decyzja | Skutek | Uzasadnienie |
|---|---|---|
| 63 pacjentów w obu kohortach: zostają **pozytywni**, ich rekordy KOR usunięte | −145 rekordów KOR; flaga `pacjent_mieszany` w danych | pacjent nie może być jednocześnie pozytywny i nieoznaczony; rozpoznanie jest faktem, a „nieoznaczony” tylko brakiem informacji |
| KREA > 50 mg/dl → brak | 210 wartości | poza zakresem mg/dl u żywego człowieka, więc błąd jednostki (µmol/l) |
| KREA > 10 mg/dl przy eGFR-MDRD ≥ 30 → brak | 516 wartości | eGFR jest liczony z kreatyniny; KREA > 10 mg/dl wymaga skrajnie niskiego eGFR, więc to sprzeczność |
| K > 15 mmol/l → brak | 25 wartości | niezgodne z życiem |
| Na < 80 mmol/l → brak | 1 wartość | niezgodne z życiem |
| skrajne, ale możliwe wartości **zostają** | K 9–15 (61), Na 80–100 (7), WBC > 200 (7), KREA > 10 przy eGFR < 30 (252) | możliwe w ciężkich stanach (hiperkaliemia, hiponatremia, białaczka, niewydolność nerek) |
| rozjazd czasowy kohort, asymetria braków | **bez korekty** | dane są, jakie są; opisane jako ograniczenia |

Wynik: **78 052 rekordy, 40 924 pacjentów** (bez zmian), **752 wartości zamienione na brak**, które MICE uzupełnia w foldzie. Każda zmiana jest wypisana w `3-sgkf-split/przygotowanie/results/usuniete_rekordy.csv` i `usuniete_wartosci.csv`. Reguła dla KREA opiera się na danych: do KREA ≈ 10 mg/dl mediana eGFR spada zgodnie z fizjologią (przy 5–10 mg/dl wynosi 8). Powyżej 10 wraca do wartości prawidłowej (60 przy 20–50 mg/dl), czego nie da się wyjaśnić inaczej niż błędem jednostki lub wpisu (`RAPORT_SGKF_MICE.md`, sekcja 2.2).

### 5.4 Część B: podział (wersja 2)

- podział na tabeli pacjentów (`StratifiedKFold`, 5 foldów, `shuffle=True`, seed 42); warunek grupowania jest spełniony z konstrukcji,
- 12 warstw stratyfikacji: etykieta × okno czasowe KOR × klasa liczby rekordów (1 / 2 / 3–4 / 5+),
- **przydział zapisany** w `3-sgkf-split/sgkf/results/pacjent_fold.csv` z odciskiem pliku wejściowego; każda zmiana danych unieważnia podział,
- 17 testów automatycznych (6 dla części A, 11 dla części B), wszystkie przechodzą.

Porównanie strategii (najgorszy przypadek z 5 ziaren):

| Strategia | Udział pozytywnych (pacjenci) | NEURO z okna czasowego KOR | NEURO z 5+ rekordami |
|---|---:|---:|---:|
| v1: SGKF, bez tasowania | 0,02 pp | 4,06 pp | 0,27 pp |
| pacjenci, stratyfikacja po etykiecie (rekomendacja z września) | 0,01 pp | **4,91 pp** | **8,49 pp** |
| **v2: pacjenci, etykieta × okno × liczba rekordów** | **0,01 pp** | **0,55 pp** | **0,22 pp** |

| Fold | Pacjenci | Pozytywni | Udział poz. | NEURO z okna KOR | Mediana roku NEURO |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 185 | 364 | 4,45% | 8,5% | 2019 |
| 1 | 8 185 | 365 | 4,46% | 8,5% | 2019 |
| 2 | 8 185 | 365 | 4,46% | 9,0% | 2018 |
| 3 | 8 185 | 365 | 4,46% | 9,0% | 2019 |
| 4 | 8 184 | 364 | 4,45% | 8,5% | 2018 |

### 5.5 Część B: imputacja w foldach (pełny przebieg)

Procedura w każdym foldzie: scaler dopasowany na kompletnych wierszach treningu, MICE (ExtraTrees 7/78/10) uczony tylko na treningu, część testowa jedynie transformowana, powrót do oryginalnych jednostek. Pełny przebieg całego etapu (`python 3-sgkf-split/uruchom_sgkf.py`, 04.10.2026) trwał 31 min; wszystkie kroki zakończone powodzeniem.

| | Wariant 38 cech (główny) | Wariant 35 cech (wrażliwość) |
|---|---|---|
| czas imputacji (5 foldów) | 15,5 min | 13,5 min |
| wspólni pacjenci train/test, braki i wartości ujemne po imputacji | 0 / 0 / 0 w każdym foldzie | 0 / 0 / 0 w każdym foldzie |
| uzupełnione komórki w teście: KOR / NEURO | 12,1–12,5% / 25,3–27,5% | 12,5–12,9% / 25,5–27,6% |
| zapisane foldy (poza gitem) | 44 MB | 40 MB |

**38 czy 35 cech** (`3-sgkf-split/sgkf/porownanie_cech.py`). MONO i %MONO są prawie w całości wyliczalne z innych cech (rozmaz sumuje się do 100%), więc realnie warianty różnią się głównie CRP. Pokazują to dwa porównania:
- **Test kontrolowany** (maska 10% na wspólnych 35 cechach): RMSE 0,0846 dla 38 cech i 0,0859 dla 35.
- **Prawdziwe foldy**: wartości uzupełnione w obu wariantach różnią się średnio o 0,11 odchylenia standardowego, a tylko 2,8% komórek o więcej niż 0,5 SD. Wartości obserwowane są identyczne.

**Wniosek: wybór ma mały wpływ na imputację**, a 38 cech jest nieznacznie lepsze. Zostaje wariantem głównym, a 35 jest gotową analizą wrażliwości. Wpływ na sam model, zwłaszcza CRP jako możliwego markera hospitalizacji, sprawdzimy w modelowaniu na obu wariantach foldów.

Szczegóły, tabele per fold i odciski wyników do porównań między komputerami: `3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcja 4.

---

## 6. Weryfikacja danych i decyzje

We wrześniu cały dotychczasowy materiał przejrzano pod kątem jednego pytania: *czy wynik modelu będzie mówił o tętniakach, czy o tym, jak powstały dane?* 04.10.2026 zapadły decyzje. Poniżej stan każdego ustalenia; liczby liczą skrypty `3-sgkf-split/przygotowanie/analiza_decyzji.py` (decyzje z 04.10) i `etap0_diagnostyka.py` (diagnostyka z września), a reguły agregacji `4-pu-setup/etap0_agregacja.py`.

| # | Ustalenie | Decyzja / status |
|---|---|---|
| 1 | **Rozjazd czasowy kohort** | bez korekty, ograniczenie; foldy wyrównane pod względem epoki |
| 2 | **Wyniki NEURO pochodzą najpewniej z hospitalizacji** | założenie robocze; zmienia interpretację, nie kod |
| 3 | **63 pacjentów w obu kohortach** | **rozwiązane**: pozytywni, rekordy KOR usunięte |
| 4 | **Selekcja cech z użyciem etykiety** | **rozwiązane**: 38 cech główne, 35 jako wrażliwość (wpływ zmierzony) |
| 5 | **Wartości niemożliwe, jednostki KREA** | **rozwiązane**: 752 wartości → brak (reguły fizjologiczne) |
| 6 | **Reguła agregacji rekordów do pacjenta** | rekomendacja: mediana; należy do etapu modelowania |
| 7 | **Podział nietasowany, niezapisany** | **rozwiązane**: podział v2, zamrożony |
| 8 | **Założenie SCAR wątpliwe** | ograniczenie nieusuwalne |
| 9 | **Brak pomiaru rzeczywistego odsetka pomyłek** | ograniczenie nieusuwalne |
| 10 | **Asymetria braków między kohortami** | bez korekty, ograniczenie; raportowana w każdym foldzie |

### 6.1 Rozjazd czasowy kohort

| Rok | KOR | NEURO |
|---|---:|---:|
| do 2009 | 0 | 1 |
| 2010–2018 | **0** | 3 937 |
| 2019 | 326 | 534 |
| 2020 | 31 259 | 317 |
| 2021 | 39 816 | 358 |
| 2022–2024 | **0** | 1 504 |

KOR to w 99,5% lata 2020–2021, a NEURO rozciąga się na lata 2000–2024. We wspólnym oknie dat (2019-12-29 – 2021-12-26) mieści się 680 z 6 651 rekordów NEURO i 271 z 1 823 pacjentów.

Skalę zagrożenia pokazuje porównanie median (`analiza_decyzji.json`, sekcja B). Różnica NEURO w oknie vs poza nim to różnica między okresami przy tej samej chorobie. Różnica KOR vs NEURO w tym samym oknie to różnica między kohortami przy tym samym okresie.

| Cecha | KOR (okno) | NEURO w oknie | NEURO poza oknem | Różnica między okresami | Różnica między kohortami |
|---|---:|---:|---:|---:|---:|
| WBC | 8,78 | 7,25 | 8,48 | 1,23 | 1,53 |
| NEUT | 6,06 | 4,46 | 5,71 | 1,25 | 1,60 |
| GLU | 107,0 | 99,0 | 104,0 | 5,0 | 8,0 |
| Na | 139,67 | 141,00 | 139,25 | **1,75** | 1,33 |
| K | 4,30 | 4,30 | 4,10 | **0,20** | 0,00 |
| RDW | 13,60 | 13,50 | 13,63 | **0,13** | 0,10 |
| MCV | 88,94 | 90,45 | 90,00 | 0,45 | 1,51 |
| KREA | 0,88 | 0,81 | 0,78 | 0,03 | 0,06 |

Dla większości cech różnica między okresami jest tego samego rzędu co różnica między kohortami, a dla Na, K i RDW większa. To porównanie nie jest formalnym dowodem, bo grupy NEURO z różnych okresów to różni pacjenci. Wystarcza jednak, żeby traktować problem poważnie: model może rozpoznawać epokę, laboratorium albo okres pandemii zamiast choroby.

**Decyzja (04.10.2026): dane są, jakie są.** Modelujemy na całości, a rozjazd opisujemy jako ograniczenie. Ograniczenie do wspólnego okna zostawiłoby 271 pozytywnych pacjentów. Stratyfikacja podziału wyrównuje udział NEURO z okna między foldami (rozstęp 0,55 pp), ale problemu nie usuwa. Roku nie dodajemy jako cechy, bo ułatwiłoby to rozpoznanie kohorty.

### 6.2 Pochodzenie wyników NEURO

Spośród 1 823 pacjentów NEURO 1 215 ma więcej niż jeden rekord. U nich mediana rozpiętości badań wynosi 35 dni (48,9% w 30 dniach, 58,2% w 90 dniach), czyli wzorzec pojedynczej hospitalizacji. **Założenie robocze (20.09):** wyniki NEURO pochodzą z hospitalizacji, w trakcie której rozpoznano i leczono tętniaka. Model rozpoznaje wtedy profil pacjenta **hospitalizowanego** z tętniakiem, a nie ryzyko przesiewowe, i tak musi być opisany. Założenie czeka na potwierdzenie u prowadzącego lub właściciela danych.

### 6.3 Pacjenci w obu kohortach

63 pacjentów, 264 rekordy (145 KOR + 119 NEURO). U 55 rekordy KOR poprzedzają NEURO (mediana odstępu 546 dni), u 6 jest odwrotnie, a u 2 rekordy się przeplatają. Kolejność źródeł nie dowodzi kolejności zdarzeń medycznych, bo daty rozpoznania nie ma. **Rozwiązanie:** sekcja 5.3.

### 6.4 Selekcja cech

CRP, MONO i %MONO usunięto w czerwcu na podstawie korelacji z etykietą na całym zbiorze. To formalny wyciek, a etykieta rozróżnia kohorty, nie chorobę. Korelacje były bliskie zera (|ρ| ≤ 0,02). MONO i %MONO są przy tym niemal w całości wyliczalne z innych cech: rozmaz sumuje się do 100,0%, a MONO ≈ %MONO × WBC / 100. Realną różnicą między wariantami jest więc CRP. **Rozwiązanie:** 38 cech jako wariant główny, 35 jako analiza wrażliwości na tym samym podziale; wynik porównania w sekcji 5.5.

### 6.5 Wartości niemożliwe i jednostki KREA

Czyszczenie kwantylowe z etapu 1 nie usunęło wartości niemożliwych, bo granice były statystyczne. W danych zostało K do 28 mmol/l, KREA do 196 mg/dl i Na 74 mmol/l. **Rozwiązanie:** reguły fizjologiczne, sekcja 5.3.

### 6.6 Reguła agregacji rekordów do pacjenta

NEURO ma średnio 3,65 rekordu na pacjenta, KOR 1,83. Reguła wrażliwa na liczbę pomiarów tworzy sygnał z samej częstości badania: maksimum koreluje z liczbą rekordów dla 17 z 35 cech, mediana dla 2. **Rekomendacja:** mediana, ze średnią jako wrażliwością. To decyzja etapu modelowania (`4-pu-setup/PLAN_MODELOWANIA.md`); podział już wyrównuje liczbę rekordów między foldami.

### 6.7 Asymetria braków

MICE uzupełnia 26,0% komórek NEURO i 12,3% komórek KOR; kompletne wiersze, na których uczy się scaler, to w 96% KOR. Braki NRBC: 47,7% w NEURO, 2,6% w KOR. **Decyzja:** bez korekty, ograniczenie; udział raportowany w każdym foldzie.

---

## 7. Źródła prawdy: co jest czym w repozytorium

| Rola | Plik | Uwagi |
|---|---|---|
| dane surowe | `data/raw/*_merged_aggregated_1W_mean.csv` | nie modyfikować |
| wejście części A (38 cech, przed imputacją) | `data/imputation-inputs/aneurysm_concatted.csv` | stan po etapie 1 |
| **wejście do podziału i modelowania** | `data/processed/aneurysm_sgkf_input.csv` | wynik części A; 38 cech + `pacjent_mieszany` (metadana) |
| spis zmian części A | `3-sgkf-split/przygotowanie/results/usuniete_*.csv` | każda usunięta wartość i rekord |
| **przydział pacjent → fold** | `3-sgkf-split/sgkf/results/pacjent_fold.csv` + `podzial_meta.json` | czytać przez `podzial.wczytaj_podzial()` |
| zaimputowane foldy | `3-sgkf-split/sgkf/results/foldy_imputowane_{38,35}/` | poza gitem; `python 3-sgkf-split/uruchom_sgkf.py` |
| parametry MICE | `2-imputation/RAPORT_IMPUTACJA.md`, przebieg 4 | 7 / 78 / 10 |
| wariant 35 cech z raportu przejściowego | `data/processed/aneurysm_concatted_cleaned.csv` | historyczny, sprzed decyzji z 04.10 |
| imputacja globalna | `2-imputation/final/results/aneurysm_imputed_*.csv` | **tylko walidacja metody**, nie do modeli |

---

## 8. Decyzje

### 8.1 Podjęte 04.10.2026

| Nr | Decyzja | Wdrożenie |
|---|---|---|
| 0.1 | 63 pacjentów w obu kohortach: pozytywni, rekordy KOR usunięte | `przygotuj_dane.py` |
| 0.2 | rozjazd czasowy: bez korekty, ograniczenie | stratyfikacja po oknie w podziale |
| 0.4 | 38 cech główne, 35 jako wrażliwość | `aneurysm_sgkf_mice_pipeline.py --cechy` |
| 0.5 | wartości niemożliwe → brak (KREA, K, Na) | `przygotuj_dane.py` |
| 0.6 | podział po pacjentach, wielokluczowa stratyfikacja, seed 42, zamrożony | `podzial.py` |
| — | asymetria braków: bez korekty, ograniczenie | raport per fold |

### 8.2 Pozostające do potwierdzenia

| Nr | Kwestia | Stan | Kto |
|---|---|---|---|
| 0.2b | czy wyniki NEURO pochodzą sprzed rozpoznania, czy z hospitalizacji | założenie: hospitalizacja | prowadzący / właściciel danych |
| 0.3 | reguła agregacji rekordów do pacjenta | rekomendacja: mediana | zespół, przy planie modelowania |
| — | plan modelowania (wartość `q`, udział ukrywania, metody PU) | propozycja w `4-pu-setup/PLAN_MODELOWANIA.md` | zespół + prowadzący |

---

## 9. Ograniczenia, które idą do raportu końcowego

1. **Brak grupy kontrolnej**: nie da się zmierzyć rzeczywistego odsetka fałszywych alarmów w KOR.
2. **Założenie SCAR jest wątpliwe**: znani pozytywni to pacjenci objawowi, a niewykryte tętniaki są bezobjawowe.
3. **Rozjazd czasowy kohort** (KOR 2020–2021, NEURO 2000–2024); model może rozpoznawać epokę.
4. **Prawdopodobnie szpitalne pochodzenie wyników NEURO**; model rozpoznaje pacjenta hospitalizowanego, nie ryzyko przesiewowe.
5. **Asymetria braków**: MICE odtwarza 26% profilu NEURO wobec 12% KOR.
6. **Asymetryczne czyszczenie kwantylowe na etapie 1**: NEURO stracił 3,2% rekordów, KOR 0,8%.
7. **Imputacja deterministyczna**, oceniana na kompletnych wierszach; mechanizm braków niezbadany.
8. **Reguły wartości niemożliwych są zachowawcze**; błędy ukryte w średniej tygodniowej mogły zostać. eGFR jest cenzurowane (60 / 90).
9. Części historycznych kroków filtracji (stan pośredni 73 276 / 7 611 rekordów, odcięcia percentylowe) **nie da się odtworzyć jednym skryptem**, bo istnieją tylko w notebookach i notatkach.

---

## 10. Porządki w repozytorium

| Sprawa | Stan |
|---|---|
| jeden branch roboczy (`pu-modeling-setup-lk`) zamiast pięciu; plany z dawnego `pu-pipeline-1-5-lk` streszczone w `4-pu-setup/PLAN_MODELOWANIA.md` | **zrobione** (04.10) |
| dokumenty nazwane datą (`0626_`, `1026_`), jeden dokument w `4-pu-setup/` | **zrobione** |
| etap 3 podzielony na część A (przygotowanie) i B (SGKF), cały etap jednym poleceniem, logi z każdego kroku | **zrobione** |
| `master` nie zawiera prac z września i października | do zrobienia: po akceptacji zmergować `pu-modeling-setup-lk` do `master` |
| `1-data-preparation/scripts-lf/aneurysm_data_analysis.ipynb` to zapisana strona HTML z GitHuba, a nie notebook; prawdziwy to `aneurysm_data_analysis (1).ipynb` | do zrobienia: zastąpić plik notebookiem |
| notebooki czytają dane z `/content/…` (Colab) | do zrobienia: ścieżki względne |
| kopie tych samych CSV w kilku miejscach (`data/interim/`, `2-imputation/knn-am/`) | do zrobienia: jedna kopia w `data/` |
| wersje bibliotek różnią się między komputerami (sklearn 1.8 vs 1.6) | do zrobienia: wspólny `requirements.txt` z przypiętymi wersjami; do tego czasu porównywać odcisk wyników w `przebieg_imputacji_*.json` |

---

## Dokumenty szczegółowe

| Dokument | Zawartość |
|---|---|
| `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` | materiał do raportu przejściowego z 12.06 (historyczny) |
| `2-imputation/RAPORT_IMPUTACJA.md` | pełny benchmark imputacji |
| `3-sgkf-split/RAPORT_SGKF_MICE.md` | etap 3: decyzje i przygotowanie danych, podział, imputacja w foldach, przegląd merytoryczny |
| `4-pu-setup/PLAN_MODELOWANIA.md` | propozycja planu modelowania PU, założenia i ograniczenia wejściowe |
