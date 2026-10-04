# Realizacja projektu: od danych surowych do zamrożonego podziału

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)
**Stan na:** 04.10.2026
**Zakres:** etapy 1–3: przygotowanie danych, imputacja braków, kontrolowany podział (SGKF). Propozycja dalszych prac (modelowanie PU) jest w `4-pu-setup/PLAN_MODELOWANIA.md`; plan obowiązujący zostanie ustalony po spotkaniu z prowadzącym.

Dokument zbiera w jednym miejscu, **co zostało zrobione, przez kogo, na jakiej podstawie i z jakim wynikiem**, do momentu, w którym dane są gotowe do modelowania. Zastępuje rozproszone raporty cząstkowe jako punkt wejścia. Szczegóły techniczne zostają w raportach etapów, do których prowadzą odnośniki.

Wszystkie liczby są przeliczone na plikach obecnych w repozytorium (branch `pu-modeling-setup-lk`).

---

## Streszczenie

1. **Dane są przygotowane i zweryfikowane:** 78 197 rekordów tygodniowych, 40 924 pacjentów, 1 823 (4,45%) z potwierdzonym tętniakiem (NEURO, klasa P), reszta to populacja ogólna (KOR, klasa U). Zbiór wejściowy do modelowania ma 38 cech i jest **przed imputacją**.
2. **Imputacja jest wybrana na podstawie wspólnego benchmarku trzech metod** (KNN, MissForest, MICE) z oddzieloną maską do strojenia i maską do oceny. Wybrano MICE z estymatorem ExtraTrees, który jest wykonywany **wewnątrz każdego foldu**.
3. **Podział danych jest zamrożony i zapisany:** 5 foldów po pacjentach, zero wspólnych pacjentów między train i test, stratyfikacja po etykiecie, epoce badania i liczbie rekordów. Rozstęp udziału pozytywnych między foldami wynosi 0,01 pp.
4. **Audyt podziału z raportu przejściowego** wykrył, że był on nietasowany, niezapisany i zależny od wersji biblioteki. Pokazał też, że rekomendacja z września (stratyfikacja tylko po etykiecie) rozjeżdża foldy na czynnikach zakłócających nawet o 8 pp. Wersja 2 naprawia wszystkie te problemy (sekcja 5).
5. **Najpoważniejsze ryzyka nie dotyczą kodu, tylko danych:** kohorty niemal nie pokrywają się w czasie (99,5% KOR to lata 2020–2021, NEURO to lata 2000–2024), wyniki NEURO najpewniej pochodzą z hospitalizacji, a MICE uzupełnia 26% komórek NEURO wobec 12% w KOR. Te kwestie czekają na decyzje opisane w sekcji 8.

---

## 1. Zespół i podział pracy

### 1.1 Kto za co odpowiadał

| Osoba | Główne obszary | Kluczowe artefakty |
|---|---|---|
| **Łukasz Kubik** (kierownik) | organizacja repozytorium i sprintów; skrypty czyszczące i kontrolne; imputacja MICE; **wspólny benchmark imputacji** i imputacja finalna; integracja podziału z imputacją w foldzie; weryfikacja metodologiczna po raporcie przejściowym (etap 0); audyt i wersja 2 podziału | `1-data-preparation/scripts-lk/`, `2-imputation/mice-lk/`, `2-imputation/final/`, `2-imputation/RAPORT_IMPUTACJA.md`, `3-sgkf-split/`, `4-pu-setup/` |
| **Liwia Florkiewicz** | pierwsze przetworzenie danych (rozpakowanie słowników, jednostki, mapowanie badań); analiza populacyjna i wartości odstających; filtracja (wiek, płeć, próg 60% braków, ograniczenie nadreprezentowanych pacjentów); **połączenie KOR + NEURO ze zmienną celu**; analiza korelacji i selekcja cech; **pierwsza implementacja StratifiedGroupKFold**; notatki ze spotkań | `1-data-preparation/data_analysis_v1.ipynb`, `scripts-lf/`, `aneurysm_concatted*.csv`, `3-sgkf-split/aneurysm_data_StratifiedGroupKFold.ipynb`, `docs/sprints/` |
| **Adrian Meredyk** | imputacja **KNN**: implementacja, przegląd siatki `k` × wagi, raporty wyników dla KOR i NEURO | `2-imputation/knn-am/` |
| **Dominika Malisz** | imputacja **MissForest**: implementacja, dobór liczby drzew z kryterium zatrzymania, raport | `2-imputation/missforest-dm/` |

### 1.2 Macierz wkładu według etapów

● wkład główny ○ wkład uzupełniający

| Etap / zadanie | Łukasz | Liwia | Adrian | Dominika |
|---|:---:|:---:|:---:|:---:|
| Organizacja, plan sprintów, struktura repo | ● | ○ | | |
| Czyszczenie strukturalne (słowniki `{'values': […]}`, jednostki, duplikaty) | ● | ● | | |
| Kontrole anomalii medycznych (WBC, Na, K, KREA, PLT) | ● | ○ | | |
| Analiza populacyjna, wartości odstające (IQR, wykresy) | | ● | | |
| Filtracja (wiek 18–100, płeć, 60% braków, nadreprezentowani pacjenci) | | ● | | |
| Połączenie kohort, etykieta P/U | | ● | | |
| Korelacje z etykietą, selekcja cech | | ● | | |
| Imputacja: KNN | ○ | | ● | |
| Imputacja: MissForest | ○ | | | ● |
| Imputacja: MICE | ● | | | |
| Wspólny benchmark imputacji (Optuna, maski, stabilność) i imputacja finalna | ● | | | |
| Podział StratifiedGroupKFold (v1) | ○ | ● | | |
| Integracja podziału z MICE w foldzie | ● | | | |
| Weryfikacja po raporcie przejściowym (etap 0: diagnostyka, agregacja, stabilność) | ● | | | |
| Audyt i wersja 2 podziału | ● | | | |

Udział mierzony liczbą commitów (Łukasz 40, Liwia 16, Adrian 5, Dominika 2) niedoszacowuje pracę w notebookach. Analiza populacyjna i scalenie danych to kilka dużych commitów, a nie wiele małych. Tabela powyżej opiera się na zawartości zmian, nie na ich liczbie.

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
| 11–20.09 | plan modelowania PU, reorganizacja repo, diagnostyka etapu 0, recenzja metodologiczna, pytania do prowadzącego, analiza reguły agregacji, siatka stabilności podziału | Łukasz |
| 25.09 | prototyp infrastruktury PU, punkty 1–5 (poza zakresem; wnioski w `4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 6) | Łukasz |
| **04.10** | **audyt i wersja 2 podziału; to podsumowanie** | Łukasz |

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

`check_med_data.py` znalazł wartości biologicznie niemożliwe: Na < 100 mmol/l, K > 9 mmol/l (do 39), WBC > 100 G/l, PLT = 0, wiek 222 lata. Przy kreatyninie postawiono hipotezę błędu jednostek (µmol/l zamiast mg/dl). Po analizie zespół zdecydował wtedy **nie przeliczać** wartości, bo KREA > 20 mg/dl bywa realne przy dializie, a przelicznik ÷ 88,4 dawał wartości zbyt niskie. Wrzesień doprecyzował skalę problemu (sekcja 6, punkt 0.5).

### 3.3 Analiza populacyjna i filtracja (Liwia)

- **Wartości odstające:** reguła IQR (Q1 − 1,5·IQR, Q3 + 1,5·IQR) z weryfikacją na histogramach i wykresach pudełkowych oraz binarną macierzą anomalii. Flaga IQR oznacza obserwację nietypową statystycznie, a nie błąd medyczny.
- **Wiek 18–100 lat.** Usunięto osoby niepełnoletnie (brak zgody na badania) i wartości niemożliwe.
- **Próg 60% braków** w dwóch kierunkach: usunięto najbardziej niekompletne kolumny (wspólne dla obu kohort) oraz 1 793 pacjentów KOR z brakami powyżej 60% profilu.
- **Pacjenci nadreprezentowani:** obcięcie najstarszych pomiarów u osób z bardzo długą historią (odcięcia na 11. i 38. percentylu wg notatek). Zasada ze spotkania 23.04: pacjentów z jednym pomiarem **nie usuwamy**, bo są najcenniejsi.

### 3.4 Połączenie kohort i selekcja cech (Liwia)

KOR i NEURO połączono w jeden zbiór z etykietą `label` (NEURO = 1, KOR = 0). Wynik: **78 197 rekordów × 42 kolumny** (38 cech + 4 kolumny meta), plik `data/imputation-inputs/aneurysm_concatted.csv`.

Zależność cech z etykietą zbadano czterema miarami: Pearson, Spearman, Kendall, Phi-K. Trzy najsłabiej związane cechy (CRP, MONO, %MONO) usunięto, co dało wariant 35 cech (`data/processed/aneurysm_concatted_cleaned.csv`). Weryfikacja wrześniowa odwróciła tę decyzję (sekcja 6, punkt 0.4): **wariantem głównym jest 38 cech**.

### 3.5 Stan danych na wyjściu etapu 1

| | Rekordy | Pacjenci | Rekordów na pacjenta (średnio) |
|---|---:|---:|---:|
| KOR (`label = 0`) | 71 546 | 39 164 | 1,83 |
| NEURO (`label = 1`) | 6 651 | 1 823 | 3,65 |
| **Razem** | **78 197** | **40 924** | 1,91 (mediana 1, maks. 38) |

63 pacjentów ma rekordy w obu kohortach (sekcja 6, punkt 0.1). Braki: 13,8% komórek cech, kompletnych jest 23 092 rekordy z 78 197.

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

## 5. Etap 3: kontrolowany podział danych

**Raport:** `3-sgkf-split/RAPORT_SGKF_MICE.md` (sekcja 6 zawiera pełny audyt)

### 5.1 Wersja 1, czyli stan na raport przejściowy (Liwia, Łukasz)

Liwia zaimplementowała `StratifiedGroupKFold` (5 foldów) w notebooku. Grupowanie po `patient_id` gwarantuje, że wszystkie rekordy pacjenta są po jednej stronie podziału. Łukasz zintegrował podział z imputacją, dzięki czemu w każdym foldzie scaler i MICE uczą się wyłącznie na części treningowej, a część testowa jest tylko transformowana. Asercje pilnowały zera wspólnych pacjentów i zera NaN po imputacji.

**Ta część była poprawna i zostaje.** Imputacja w foldzie i grupowanie po pacjencie to dwa najważniejsze zabezpieczenia przed wyciekiem.

### 5.2 Audyt (Łukasz, wrzesień–październik)

| # | Problem w v1 | Skutek |
|---|---|---|
| 1 | brak tasowania (`shuffle=False`) | podział zależał od kolejności wierszy w CSV; „ziarno 42” nie miało na niego wpływu |
| 2 | przydział nie był zapisywany | każde uruchomienie mogło dać inne foldy bez śladu |
| 3 | SGKF z tasowaniem zależy od wersji sklearn | ta sama konfiguracja dawała **inne podziały na dwóch komputerach zespołu** (sklearn 1.6 vs 1.8) |
| 4 | stratyfikacja tylko po etykiecie | foldy niewyrównane pod względem epoki badania i liczby rekordów, czyli dwóch znanych źródeł fałszywego sygnału |
| 5 | brak reguły dla 63 pacjentów mieszanych | etykieta pacjenta nieokreślona |

Kluczowy wynik porównania strategii (5 foldów, najgorszy przypadek z 5 ziaren):

| Strategia | Udział pozytywnych (pacjenci) | NEURO z okna czasowego KOR | NEURO z 5+ rekordami |
|---|---:|---:|---:|
| v1: SGKF, bez tasowania | 0,09 pp | 2,03 pp | 0,78 pp |
| pacjenci, stratyfikacja po etykiecie (rekomendacja z września) | 0,01 pp | **5,18 pp** | **7,95 pp** |
| **v2: pacjenci, etykieta × okno × liczba rekordów** | **0,01 pp** | **0,79 pp** | **0,49 pp** |

### 5.3 Wersja 2, czyli stan obecny

- podział na tabeli pacjentów (`StratifiedKFold`, 5 foldów, `shuffle=True`, seed 42); warunek grupowania jest spełniony z konstrukcji,
- 12 warstw stratyfikacji: etykieta × okno czasowe KOR × klasa liczby rekordów (1 / 2 / 3–4 / 5+),
- 63 pacjentów mieszanych: etykieta pozytywna i flaga `mieszany` (konfigurowalne, decyzja 0.1),
- **przydział zapisany** w `3-sgkf-split/results/pacjent_fold.csv` z odciskiem danych i wersjami bibliotek; zmiana danych unieważnia podział i wymusza świadome przebudowanie,
- pipeline MICE w foldzie czyta zapisany przydział; dodatkowo sprawdza brak wartości ujemnych i to, że każdy pacjent jest w teście dokładnie raz,
- 10 testów automatycznych (wszystkie przechodzą); szybki przebieg całej ścieżki na podpróbce działa poprawnie.

| Fold | Pacjenci | Pozytywni | Udział poz. | NEURO z okna KOR | Mediana roku NEURO |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 185 | 364 | 4,45% | 9,9% | 2019 |
| 1 | 8 185 | 365 | 4,46% | 10,1% | 2018 |
| 2 | 8 185 | 365 | 4,46% | 10,4% | 2019 |
| 3 | 8 185 | 365 | 4,46% | 10,7% | 2019 |
| 4 | 8 184 | 364 | 4,45% | 10,2% | 2018 |

### 5.4 Nowe ustalenie: imputacja w foldzie jest asymetryczna między kohortami

| | KOR | NEURO |
|---|---:|---:|
| komórki uzupełniane przez MICE | 12,3% | **26,0%** |
| kompletne wiersze (podstawa skalowania) | 20 663 | 780 |
| braki NRBC | 2,6% | 47,7% |

Około jednej czwartej profilu NEURO jest odtwarzane przez imputer uczony w 91% na KOR. To nie jest błąd podziału, ale **ryzyko interpretacyjne**: różnice między kohortami mogą wynikać po części z samego wzorca braków. Proponujemy analizę wrażliwości na etapie modelowania, np. model na cechach o niskim odsetku braków.

---

## 6. Weryfikacja po raporcie przejściowym (etap 0)

We wrześniu przejrzano cały dotychczasowy materiał pod kątem tego, czy wynik modelu będzie mówił o tętniakach, czy o tym, jak powstały dane. Skrypty: `4-pu-setup/etap0_diagnostyka.py`, `4-pu-setup/etap0_agregacja.py`, `3-sgkf-split/etap0_sgkf_stabilnosc.py`; ustalenia: `4-pu-setup/USTALENIA_DANYCH.md`. Poniżej tylko punkty, które dotyczą etapów 1–3.

| Punkt | Ustalenie | Wpływ na etapy 1–3 | Status |
|---|---|---|---|
| **0.2** rozjazd czasowy | 99,5% rekordów KOR pochodzi z lat 2020–2021, NEURO z lat 2000–2024; w oknie dat KOR mieści się 271 z 1 823 pacjentów NEURO. Różnice median między okresami są tego samego rzędu co różnice między kohortami (dla Na, K i RDW większe) | stratyfikacja v2 wyrównuje epokę między foldami, ale **nie usuwa** problemu | **do decyzji** (prowadzący) |
| **0.2b** pochodzenie NEURO | u pacjentów NEURO z kilkoma rekordami mediana rozpiętości badań to 35 dni, czyli wzorzec hospitalizacji | interpretacja: model rozpozna profil pacjenta **hospitalizowanego** z tętniakiem, nie ryzyko przesiewowe | założenie przyjęte, **do potwierdzenia** |
| **0.1** 63 pacjentów w obu kohortach | u 55 z nich rekordy KOR poprzedzają NEURO (mediana 546 dni); kolejność źródeł nie dowodzi jednak kolejności zdarzeń medycznych | v2: etykieta pozytywna + flaga, wariant wykluczenia gotowy | **do decyzji** |
| **0.4** selekcja cech | CRP, MONO i %MONO usunięto na podstawie korelacji z etykietą na całym zbiorze, co jest formalnym wyciekiem. Korelacje są bliskie zera (|ρ| ≤ 0,02), a etykieta rozróżnia kohorty, a nie chorobę | wariant główny to 38 cech, 35 jako wrażliwość; pliki różnią się tylko tymi kolumnami | rekomendacja, wdrożona w v2 |
| **0.5** wartości skrajne | 86 wartości K > 9 mmol/l u 78 pacjentów; KREA > 50 w 0,33% rekordów (prawdopodobnie mieszane jednostki) | przełącznik „wartości niemożliwe → brak danych” jest gotowy, domyślnie wyłączony | **do konsultacji klinicznej** |
| **0.3** agregacja do pacjenta | NEURO ma 2 razy więcej rekordów na pacjenta. Maksimum przenosi liczbę badań do 17 z 35 cech, mediana do 2 | v2 stratyfikuje po liczbie rekordów; reguła: mediana | rekomendacja |
| **0.6** konfiguracja podziału | brak tasowania, zależność od kolejności pliku | rozwiązane w v2 | **zamknięte** |

---

## 7. Źródła prawdy: co jest czym w repozytorium

| Rola | Plik | Uwagi |
|---|---|---|
| dane surowe | `data/raw/*_merged_aggregated_1W_mean.csv` | nie modyfikować |
| **wejście do modelowania (38 cech, przed imputacją)** | `data/imputation-inputs/aneurysm_concatted.csv` | wariant główny |
| wejście, wariant 35 cech | `data/processed/aneurysm_concatted_cleaned.csv` | analiza wrażliwości |
| **przydział pacjent → fold** | `3-sgkf-split/results/pacjent_fold.csv` + `podzial_meta.json` | czytać przez `podzial.wczytaj_podzial()` |
| parametry MICE | `2-imputation/RAPORT_IMPUTACJA.md`, przebieg 4 | 7 / 78 / 10 |
| imputacja globalna | `2-imputation/final/results/aneurysm_imputed_*.csv` | **tylko walidacja metody**, nie do modeli |

---

## 8. Decyzje otwarte

| Nr | Decyzja | Rekomendacja | Kto | Co blokuje |
|---|---|---|---|---|
| 0.2 | kontrola rozjazdu czasowego | wariant C (dopasowanie lub ważenie po czasie i źródle) + wariant A (wspólne okno) jako wrażliwość | zespół + prowadzący | protokół modelowania |
| 0.2b | czy wyniki NEURO pochodzą sprzed rozpoznania | założenie: z hospitalizacji | prowadzący / właściciel danych | tylko wnioski |
| 0.5 | jednostki KREA, wartości niemożliwe | czekamy na informację o jednostce w systemie źródłowym | konsultacja kliniczna | jakość jednej cechy |
| 0.1 | 63 pacjentów mieszanych | pozytywni z flagą | zespół | nic (parametr) |
| 0.4 | zestaw cech | 38 | zespół | nic (wdrożone) |

---

## 9. Ograniczenia, które idą do raportu końcowego

1. Brak grupy kontrolnej: nie da się zmierzyć rzeczywistego odsetka fałszywych alarmów w KOR.
2. Założenie SCAR (znani pozytywni jako losowa próbka wszystkich chorych) jest wątpliwe, bo NEURO to pacjenci objawowi, a niewykryte tętniaki są bezobjawowe.
3. Rozjazd czasowy kohort i prawdopodobnie szpitalne pochodzenie wyników NEURO.
4. Asymetria braków między kohortami, przez co imputacja odtwarza 26% profilu NEURO.
5. Imputacja deterministyczna, oceniana na kompletnych wierszach; mechanizm braków niezbadany.
6. Części historycznych kroków filtracji (np. stan pośredni 73 276 / 7 611 rekordów, odcięcia percentylowe) nie da się dziś odtworzyć jednym skryptem, bo istnieją tylko w notebookach i notatkach.

---

## 10. Porządki w repozytorium (stan na 04.10.2026)

Wykryte przy tym podsumowaniu, bez wpływu na wyniki, ale do uporządkowania:

| Sprawa | Propozycja |
|---|---|
| `master` nie zawiera prac z września. Po porządkach z 04.10 cała praca jest na jednym branchu `pu-modeling-setup-lk` (obok `dataset_analysis_lf`); skrypty z dawnych branchy `etap0-*` przeniesiono tutaj, a plany z `pu-pipeline-1-5-lk` streszczono w `4-pu-setup/PLAN_MODELOWANIA.md` | po zatwierdzeniu zmergować `pu-modeling-setup-lk` do `master` |
| `1-data-preparation/scripts-lf/aneurysm_data_analysis.ipynb` to zapisana strona HTML z GitHuba, a nie notebook; prawdziwy notebook to `aneurysm_data_analysis (1).ipynb` | zastąpić plik notebookiem, usunąć duplikat z „(1)” |
| notebooki czytają dane z `/content/…` (Colab) | ścieżki względne do `data/` |
| kopie tych samych CSV w kilku miejscach (`data/interim/`, `data/interim/cleaned/`, `2-imputation/knn-am/`) | jedna kopia w `data/`, w pozostałych miejscach odwołania |
| prototyp PU z września liczył własny podział ze stratyfikacją tylko po etykiecie | przyszły kod modelowania ma czytać przydział z etapu 3 (`podzial.wczytaj_podzial()`) |
| wersje bibliotek różnią się między komputerami (sklearn 1.8 vs 1.6) | wspólny `requirements.txt` z przypiętymi wersjami w korzeniu repo |

---

## Dokumenty szczegółowe

| Dokument | Zawartość |
|---|---|
| `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` | materiał do raportu przejściowego z 12.06 |
| `2-imputation/RAPORT_IMPUTACJA.md` | pełny benchmark imputacji |
| `3-sgkf-split/RAPORT_SGKF_MICE.md` | metodologia podziału, audyt v1, wersja 2 |
| `4-pu-setup/USTALENIA_DANYCH.md` | 10 problemów wykrytych w danych, decyzje do podjęcia, pytania do prowadzącego |
| `4-pu-setup/PLAN_MODELOWANIA.md` | propozycja planu modelowania PU (do ustalenia na spotkaniu) |
