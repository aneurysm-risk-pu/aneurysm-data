# Realizacja projektu: od danych surowych do zamrożonego podziału SGKF

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)
**Stan na:** 04.10.2026, z poprawkami po weryfikacji z 06.10 i 16–17.10.2026 oraz z podziałem przebudowanym i przeliczonym 31.10.2026 (branch `pu-modeling-setup-lk`)
**Zakres:** etapy 1–3: przygotowanie danych, imputacja braków, przygotowanie danych do podziału i kontrolowany podział z imputacją w foldach (SGKF). Propozycja dalszych prac (modelowanie PU) jest w `4-pu-setup/PLAN_MODELOWANIA.md`; plan obowiązujący zostanie ustalony z prowadzącym.

Dokument zbiera w jednym miejscu, **co zostało zrobione, przez kogo, na jakiej podstawie i z jakim wynikiem**, do momentu, w którym dane są gotowe do modelowania. Jest punktem wejścia; szczegóły techniczne zostają w raportach etapów, do których prowadzą odnośniki. Każda liczba pochodzi z plików i skryptów w repozytorium; przy decyzjach z 04.10.2026 wskazany jest skrypt, który ją liczy.

---

## Streszczenie

1. **Dane są przygotowane, oczyszczone i gotowe do modelowania:** 78 052 rekordy tygodniowe, 40 924 pacjentów, 1 823 (4,45%) z potwierdzonym tętniakiem (NEURO, klasa P), reszta to kohorta KOR (klasa U, nieoznaczona). Wejście do podziału ma 38 cech i jest **przed imputacją**.
2. **Imputacja jest wybrana na podstawie wspólnego benchmarku trzech metod** (KNN, MissForest, MICE) z oddzieloną maską do strojenia i maską do oceny. Wybrano MICE z estymatorem ExtraTrees, który jest wykonywany **wewnątrz każdego foldu**.
3. **04.10.2026 Łukasz przygotował i wdrożył decyzje dotyczące danych** osobnym, przetestowanym kodem z pełnym spisem zmian. Wymagają potwierdzenia przez zespół i prowadzącego. Dotyczą 63 pacjentów obecnych w obu kohortach i 752 wartości uznanych za błędne (KREA, K, Na). Rozjazd czasowy kohort i asymetria braków zostają bez korekty, jako opisane ograniczenia (sekcja 6).
4. **Podział jest zamrożony i zapisany:** `StratifiedGroupKFold` (SGKF) na 5 foldów, zero wspólnych pacjentów między train i test, stratyfikacja po etykiecie, okresie badania i liczbie rekordów (12 warstw). Rozstęp udziału chorych między foldami wynosi 0,02 pp. Wersja z raportu przejściowego stratyfikowała tylko po etykiecie, nie miała tasowania i nie była zapisana (sekcja 5).
5. **Imputacja w foldach jest wykonana w pełni na jednym zestawie 38 cech**, z kontrolą braku wycieku w każdym foldzie (sekcja 5.5). Selekcję cech z etapu 1 (usunięcie CRP, MONO, %MONO) cofnięto, bo opierała się na korelacji z etykietą na całym zbiorze (sekcja 6.4).
6. **Najpoważniejsze ryzyka leżą w danych, nie w kodzie** (sekcja 5.6):
   - kohorty niemal nie pokrywają się w czasie, a okres badania da się rozpoznać w danych samych chorych (AUC 0,83); najsilniejszym śladem jest zestaw zleconych badań, ale nie jedynym,
   - sam wzorzec braków odróżnia kohorty z AUC 0,82, a imputacja zostawia rozpoznawalny ślad (NRBC),
   - przy brakach całymi panelami MICE jest niewiele lepsze od uzupełniania średnią,
   - KOR ma profil szpitalny, a wyniki NEURO najpewniej pochodzą z hospitalizacji,
   - ukryci chorzy zachowują cechy kohorty NEURO: sam wzorzec braków odzyskuje około 52% z nich w górnych 5% rankingu, więc także odzyskiwanie ukrytych wymaga porównania z modelami kontrolnymi.

   Te ryzyka idą do ograniczeń raportu (sekcja 9) i do pytań na spotkanie (`1026_SPOTKANIE_PROWADZACY.md`).

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
| 10–11.06 | integracja: SGKF + MICE w foldzie (skrypt `aneurysm_sgkf_mice_pipeline.py`) | Łukasz |
| **12.06** | **raport przejściowy** | zespół |
| 11–20.09 | plan modelowania PU, reorganizacja repo, raport metodologiczny SGKF + MICE (11.09), diagnostyka danych, recenzja metodologiczna, pytania do prowadzącego, analiza reguły agregacji, siatka stabilności podziału | Łukasz |
| 25.09 | prototyp infrastruktury PU (wnioski w `4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 6) | Łukasz |
| **04.10** | **decyzje dotyczące danych i zestawu cech; przygotowanie danych do SGKF; podział v2; pełna imputacja w foldach; ocena imputacji protokołem z etapu 2; diagnostyka wyników; to podsumowanie** | Łukasz |
| **31.10** | **podział przebudowany na dosłowny `StratifiedGroupKFold` z 12 warstwami (zalecenie prowadzącego); cały etap 3 przeliczony od nowa; sprostowanie opisu podziału z czerwca** | Łukasz |

---

## 2. Dane wejściowe

| Kohorta | Plik surowy | Wiersze × kolumny | Rola w PU |
|---|---|---:|---|
| KOR | `data/raw/kor_merged_aggregated_1W_mean.csv` | 73 679 × 128 | kohorta bez rozpoznania tętniaka w danych (pochodzenie do potwierdzenia, sekcja 5.6): **nieoznaczeni (U)** |
| NEURO | `data/raw/neuro_merged_aggregated_1W_mean.csv` | 7 655 × 128 | pacjenci z rozpoznanym tętniakiem: **pozytywni (P)** |

Rekord to tydzień uśrednionych wyników laboratoryjnych pacjenta, z kluczem `{patient_id}-{rok}-W{tydzień}`. Rok w kluczu jest kalendarzowy, a tydzień według ISO, co na przełomie roku daje kolizje (sekcja 3.1). Cechy obejmują morfologię z rozmazem, biochemię (KREA, Na, K, GLU, eGFR), koagulologię (PT, INR, APTT, WAPTT), CRP, wiek i płeć.

**Dlaczego PU, a nie zwykła klasyfikacja.** O pacjentach KOR wiemy tylko, że w danych nie mają rozpoznania tętniaka. Nie wiemy, czy i u kogo wykonano obrazowanie, więc wśród nich mogą być osoby z niewykrytym tętniakiem. `label = 0` znaczy „nieoznaczony”, a nie „zdrowy”. Ta jedna obserwacja wyznacza większość decyzji metodologicznych w kolejnych etapach.

---

## 3. Etap 1: przygotowanie danych

**Raporty:** `1-data-preparation/cleaning_summary.md`, `docs/sprints/DATA_ANALYSIS_SPRINT1.md`, `docs/sprints/` (notatki ze spotkań)

### 3.1 Czyszczenie strukturalne (Liwia, Łukasz)

| Problem | Rozwiązanie | Uzasadnienie |
|---|---|---|
| Kolumny w formacie `{'values': […]}` | wartości liczbowe uśrednione; kolumny tekstowe (`examination_type`, `descriptive_result`) usunięte | plik jest agregatem tygodniowym `_1W_mean`, więc lista to niedokończona agregacja, a średnia ją dokańcza |
| Kolumny zakresów referencyjnych i jednostek | usunięte | norma nie jest cechą pacjenta; zespół zdecydował też, żeby nie uzupełniać norm |
| Zmienne techniczne (`diagnosis_id`, `description_type`) | usunięte | brak treści klinicznej |
| „Duplikat” `custom_id = 194143-2017-W52` | usunięty rekord starszy (2017-01-01), zachowany nowszy (2017-12-31) | **korekta 16.10.2026:** to nie duplikat, tylko dwa różne badania tego samego pacjenta NEURO, odległe o rok (wiek 71,05 i 72,05, inne wyniki). `custom_id` łączy rok kalendarzowy z tygodniem ISO, a 2017-01-01 należy do tygodnia ISO 2016-W52, więc oba dostały ten sam klucz. Strata: jeden prawdziwy rekord. W surowych danych nie ma duplikatów pacjent + data. Rok w kluczu różni się od roku ISO w 583 rekordach KOR i 72 NEURO obecnego wejścia, ale żaden klucz się już nie powtarza |
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

Zależność cech z etykietą zbadano czterema miarami: Pearson, Spearman, Kendall, Phi-K. Trzy najsłabiej związane cechy (CRP, MONO, %MONO) usunięto, co dało wariant 35 cech (`data/processed/aneurysm_concatted_cleaned.csv`). Weryfikacja zakwestionowała tę selekcję, a 04.10.2026 została cofnięta (sekcja 6.4). **Projekt pracuje na jednym zestawie 38 cech.**

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

**Stabilność:** na drugiej masce (seed 7) ranking się nie zmienił. Dla NEURO Optuna dobrała wtedy parametry od nowa (MICE 51/29 zamiast 18/100), więc nie był to test samej maski; dla KOR parametry są te same. Przewaga MICE nad MissForest na NEURO jest niewielka (0,004 i 0,002 RMSE). MICE w walidacji na 10 maskach: KOR 0,0865 ± 0,0007, NEURO 0,0950 ± 0,0025, przy parametrach dobranych osobno dla kohort (KOR 7/78, NEURO 28/110). Parametry dobrane na KOR, sprawdzone na jednej masce (seed 42), pogarszają wynik NEURO o 0,00001 RMSE i są przy tym około 7 razy szybsze.

**Decyzja:** `IterativeImputer` z ExtraTrees (nazywany w projekcie MICE), wspólne parametry dla obu kohort: `max_iter = 7, n_estimators = 78, max_depth = 10`. Tabela wyżej pochodzi z przebiegu 3, w którym porównano wszystkie trzy metody; z parametrami finalnymi (przebieg 4) MICE ma RMSE 0,0861 na KOR i 0,0984 na NEURO.

**Dlaczego nie MissForest, skoro wygrywa na KOR:** żadna metoda nie wygrała na obu zbiorach. MICE wygrywa na mniejszym i trudniejszym NEURO (w benchmarku było szybsze tylko na KOR: 70,2 wobec 107,6 min; na NEURO 30,4 wobec 17,2 min), a jeden spójny imputer dla obu kohort jest wymagany, bo imputacja odbywa się na połączonym zbiorze wewnątrz foldu.

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
| 1 | stratyfikacja tylko po etykiecie | udział chorych z okresu KOR różnił się między foldami o 4,06 pp, a okres badania to znane źródło fałszywego sygnału |
| 2 | brak tasowania (`shuffle=False`) | kolejność wyznaczały numery pacjentów, a „ziarno 42” nie miało wpływu na podział, więc nie dało się sprawdzić, czy wynik nie jest przypadkowy |
| 3 | przydział nie był zapisywany | każdy etap liczyłby podział od nowa i nic nie pilnowałoby, czy dane są te same |
| 4 | brak reguły dla 63 pacjentów obecnych w obu kohortach | etykieta pacjenta nieokreślona |

**Sprostowanie z 31.10.2026.** Wcześniej pisaliśmy, że podział z czerwca zależał od kolejności wierszy i od wersji sklearn. Kontrola `3-sgkf-split/sgkf/kontrola_sgkf_czerwiec.py` pokazała, że nie: po potasowaniu wierszy wszyscy pacjenci zostają w tych samych foldach, a bez tasowania oba komputery dają ten sam wynik. Różnice wersji dotyczą tylko wbudowanej opcji `shuffle=True`, która w sklearn 1.6.1 ma błąd psujący wyrównanie (`RAPORT_SGKF_MICE.md`, sekcja 3.2).

### 5.3 Część A: przygotowanie danych do SGKF (decyzje z 04.10.2026)

| Decyzja | Skutek | Uzasadnienie |
|---|---|---|
| 63 pacjentów w obu kohortach: zostają **pozytywni**, ich rekordy KOR usunięte | −145 rekordów KOR; lista pacjentów w spisie `usuniete_rekordy.csv` | pacjent nie może być jednocześnie pozytywny i nieoznaczony; rozpoznanie jest faktem, a „nieoznaczony” tylko brakiem informacji |
| KREA > 50 mg/dl → brak | 210 wartości | poza zakresem mg/dl u żywego człowieka, więc błąd jednostki (µmol/l) |
| KREA > 10 mg/dl przy eGFR-MDRD ≥ 30 → brak | 516 wartości | eGFR jest liczony z kreatyniny; KREA > 10 mg/dl wymaga skrajnie niskiego eGFR, więc to sprzeczność |
| K > 15 mmol/l → brak | 25 wartości | niezgodne z życiem |
| Na < 80 mmol/l → brak | 1 wartość | niezgodne z życiem |
| skrajne, ale możliwe wartości **zostają** | K 9–15 (61), Na 80–100 (7), WBC > 200 (7), KREA > 10 przy eGFR < 30 (250) | możliwe w ciężkich stanach (hiperkaliemia, hiponatremia, białaczka, niewydolność nerek) |
| rozjazd czasowy kohort, asymetria braków | **bez korekty** | dane są, jakie są; opisane jako ograniczenia |
| zestaw cech: **38, jeden wariant** | CRP, MONO, %MONO wracają | selekcja z etapu 1 używała etykiety całego zbioru; wpływu na modele PU nie oceniono (sekcja 6.4) |

Wynik: **78 052 rekordy, 40 924 pacjentów** (bez zmian), **752 wartości zamienione na brak**, które MICE uzupełnia w foldzie. Każda zmiana jest wypisana w `3-sgkf-split/przygotowanie/results/usuniete_rekordy.csv` i `usuniete_wartosci.csv`. Reguła dla KREA opiera się na danych: do KREA ≈ 10 mg/dl mediana eGFR spada zgodnie z oczekiwaną zależnością (przy 5–10 mg/dl wynosi 8). Powyżej 10 mediana eGFR znowu rośnie: do 30,8 przy 10–20 mg/dl i do 60 przy 20–50 mg/dl. Wartość 60 jest górną granicą raportowania eGFR-MDRD, a nie potwierdzeniem prawidłowej funkcji nerek. Odwrócenie zależności wskazuje na możliwy błąd jednostki lub wpisu (`RAPORT_SGKF_MICE.md`, sekcja 2.2).

### 5.4 Część B: podział (wersja 2)

- `StratifiedGroupKFold` (SGKF) na 5 foldów, zgodnie z zaleceniem prowadzącego: `groups` = pacjent, `y` = warstwa pacjenta,
- 12 warstw stratyfikacji: etykieta × okno czasowe KOR × klasa liczby rekordów (1 / 2 / 3–4 / 5+); okno niczego nie usuwa, służy tylko do rozłożenia pacjentów po foldach,
- losowa kolejność pacjentów sterowana ziarnem 42; mieszamy ją sami, bo wbudowane `shuffle=True` w sklearn 1.6.1 ma błąd,
- **przydział zapisany** w `3-sgkf-split/sgkf/results/pacjent_fold.csv` z odciskiem pliku wejściowego; każda zmiana danych unieważnia podział,
- 18 testów automatycznych (6 dla części A, 12 dla części B), wszystkie przechodzą.

Porównanie strategii (najgorszy przypadek z 5 ziaren):

| Strategia | Udział chorych (pacjenci) | Udział rekordów NEURO | Chorzy z okna KOR | Chorzy z 5+ rekordami |
|---|---:|---:|---:|---:|
| wersja 1: SGKF, `y` = etykieta, bez tasowania | 0,02 pp | 0,01 pp | **4,06 pp** | 0,27 pp |
| wariant testowany: `StratifiedKFold` na tabeli pacjentów, te same warstwy | 0,01 pp | 1,15 pp | 0,55 pp | 0,22 pp |
| **wersja 2: SGKF, `y` = 12 warstw** | **0,04 pp** | **0,05 pp** | **0,75 pp** | **0,39 pp** |

Wariant testowany obowiązywał od 04.10 do 31.10.2026. Na pacjentach oba warianty są praktycznie równe (rozkład chorych między foldami różni się o najwyżej jednego chorego), a na rekordach SGKF wypada wyraźnie lepiej. Dlatego zostajemy przy SGKF. Wyniki dalszych kroków na obu wariantach są praktycznie takie same: błąd MICE przy brakach całymi panelami wobec średniej to 0,98 i 0,89 na obu, AUC kohort na pacjentach 0,928 i 0,927, a `RecallHidden@5%` samego wzorca braków 51,6% i 51,7%. Największa różnica dotyczy rozdzielności we wspólnym oknie dat, opartej na małej próbie: AUC 0,801 i 0,782.

Zamrożony podział (seed 42):

| Fold | Pacjenci | Rekordy | Chorzy | Udział chorych | Chorzy z okna KOR | Mediana roku NEURO |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 185 | 15 609 | 366 | 4,47% | 9,0% | 2018 |
| 1 | 8 185 | 15 610 | 364 | 4,45% | 8,5% | 2019 |
| 2 | 8 185 | 15 610 | 365 | 4,46% | 9,0% | 2019 |
| 3 | 8 185 | 15 611 | 364 | 4,45% | 8,5% | 2018 |
| 4 | 8 184 | 15 612 | 364 | 4,45% | 8,5% | 2018 |

### 5.5 Część B: imputacja w foldach (pełny przebieg)

Procedura w każdym foldzie: scaler dopasowany na kompletnych wierszach treningu, MICE (ExtraTrees 7/78/10) uczony tylko na treningu, część testowa jedynie transformowana, powrót do oryginalnych jednostek. Pełny przebieg (`python 3-sgkf-split/uruchom_sgkf.py`, 31.10.2026, na podziale SGKF) zakończył się powodzeniem we wszystkich krokach.

| | Wynik |
|---|---|
| czas imputacji (5 foldów) | 15,0 min |
| wspólni pacjenci train/test, braki i wartości ujemne po imputacji | 0 / 0 / 0 w każdym foldzie |
| uzupełnione komórki w teście: KOR / NEURO | 12,2–12,4% / 24,1–27,3% |
| powtarzalność | dwa niezależne przebiegi: identyczne odciski wyników we wszystkich foldach |
| zapisane foldy (poza gitem) | 46 MB |

**Jakość imputacji** (`3-sgkf-split/sgkf/ocena_imputacji.py`). Mierzymy ją jak w etapie 2: ukrywamy znane wartości w kompletnych wierszach części testowej każdego foldu i porównujemy uzupełnienia z prawdą. Imputer jest dopasowany na treningu foldu, tak jak w produkcji.

| Maska | KOR: RMSE / względem średniej | NEURO: RMSE / względem średniej | Etap 2 (KOR / NEURO) |
|---|---|---|---|
| losowe 10% komórek (protokół z etapu 2) | 0,083 / **0,62** | 0,083 / **0,66** | 0,086 / 0,098 |
| to samo, bez wieku i płci | 0,043 / **0,41** | 0,040 / **0,45** | — |
| realistyczne wzorce braków (całe panele) | 0,107 / **0,98** | 0,083 / **0,89** | — |

Przy losowych brakach MICE w produkcji działa co najmniej tak dobrze jak w benchmarku i jednakowo dobrze dla obu kohort. Przy brakach całymi panelami, jak w prawdziwych danych (koagulogram i glukoza brakują w około połowie rekordów), **MICE jest niewiele lepsze od uzupełniania średnią z treningu**. Uzupełnienia skupiają się wokół średniej: odchylają się od niej trzy razy mniej niż prawdziwe wartości w KOR i dwa razy mniej w NEURO, więc niosą niewiele informacji o pacjencie. Losowa maska, tak jak w etapie 2, ukrywa też wiek i płeć, których w danych nigdy nie brakuje. Przypada na nie około 3/4 sumy kwadratów błędu, głównie na płeć. Bez nich stosunek do średniej przy losowych brakach wynosi 0,41 w KOR i 0,45 w NEURO, więc wniosek, że benchmark z etapu 2 przeszacowywał użyteczność imputacji, się wzmacnia.

**38 czy 35 cech, porównanie jednorazowe** (`3-sgkf-split/sgkf/results/porownanie_38_35/`). Zanim zapadła decyzja o jednym zestawie, oba warianty policzono na tym samym podziale:
- **test kontrolowany** (maska 10% na wspólnych 35 cechach): RMSE 0,0834 dla 38 cech wobec 0,0846 dla 35,
- **prawdziwe foldy**: wartości uzupełnione różnią się średnio o 0,11 odchylenia standardowego, a tylko 3,1% komórek o więcej niż 0,5 SD.

Wybór nie ma istotnego wpływu na imputację. Drobna przewaga 38 w teście kontrolowanym wynika częściowo z jego konstrukcji, bo dodatkowe kolumny nigdy nie są w nim maskowane. O jednym zestawie 38 cech przesądziła więc ocena samej selekcji (sekcja 6.4), a nie jakość imputacji.

Szczegóły, tabele per fold i odciski wyników: `3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcja 4.

### 5.6 Co pokazała analiza wyników SGKF

`3-sgkf-split/sgkf/analiza_wynikow.py` sprawdza, na czym model mógłby oprzeć się zamiast na chorobie. To nie jest model ryzyka, tylko diagnostyka danych.

| Test (zamrożone foldy, HistGradientBoosting) | AUC (pacjenci) |
|---|---:|
| NEURO vs KOR na 38 cechach po imputacji | **0,93** |
| NEURO vs KOR **tylko na wzorcu braków** (które badania zlecono) | **0,82** |
| NEURO vs KOR tylko we wspólnym oknie dat | 0,78 |
| NEURO z lat 2020–2021 vs NEURO z innych lat (rekordy) | **0,83** |

1. **Kohorty różnią się wieloma rzeczami poza chorobą.** Sam wzorzec braków daje AUC 0,82, a epokę widać w danych samych chorych z AUC 0,83. Kontrola przed imputacją (`kontrola_epoki.py`) pokazuje na tych samych rekordach: wzorzec braków 0,77–0,78, zmierzone wartości paneli 0,57–0,65, wszystkie 38 cech 0,86–0,87. Wysokie AUC w modelowaniu P-vs-U nie będzie więc dowodem wykrywania tętniaka.
2. **Imputacja zostawia rozpoznawalny ślad.** Zmierzone NRBC to w 78% dokładne zera, a uzupełnione nigdy nie są zerem. NRBC brakuje w 48% rekordów NEURO i 3% KOR, więc 63% rekordów z uzupełnionym NRBC to NEURO (w całych danych 8,5%). Samo NRBC po imputacji odróżnia kohorty z AUC 0,73, tak jak sam wskaźnik braku NRBC: uzupełnienie nie ukrywa informacji o braku badania. Przy skośnych rozkładach (CRP, GLU) uzupełnienia są wyższe od typowej zmierzonej wartości, bo skupiają się wokół średniej.
3. **KOR ma profil szpitalny, nie przesiewowy.** Wśród zmierzonych wyników KOR CRP > 5 mg/l ma 61% (NEURO 62%), WBC > 10 G/l 37%, glukoza > 99 mg/dl 64%. Mianownikiem są badania wykonane: CRP zmierzono w 81% rekordów KOR, glukozę w 50%; względem wszystkich rekordów KOR CRP > 5 mg/l ma 50%. KOR wygląda na pacjentów hospitalizowanych z lat 2020–2021, a nie na populację ogólną. To hipoteza i pytanie do prowadzącego.
4. **Skróty wystarczają, żeby odzyskać dużą część ukrytych chorych** (`3-sgkf-split/sgkf/kontrola_skrotow.py`, 17.10.2026, przeliczone 31.10). Ukrywamy losowo wybranych chorych z NEURO, więc ukryci zachowują okres badań, wzorzec braków i ślad imputacji swojej kohorty. Przy 40% ukrytych i 5 ziarnach `RecallHidden@5%` wynosi: losowy ranking 4% (oczekiwane 5%), sama demografia 12%, **sam wzorzec braków 52%**, 38 cech po imputacji MICE 66%, 38 cech bez imputacji 69%. Wysoki `RecallHidden@q` nie będzie więc sam w sobie dowodem wykrywania tętniaka. Każdy model porównujemy z modelami kontrolnymi na tych samych maskach, a wynik opisujemy jako odzyskiwanie znanych chorych w tym zbiorze. Protokół tej kontroli jest orientacyjny, bo nie ustalono jeszcze jednostki treningu ani agregacji.
5. **Rozdzielność we wspólnym oknie dat nie jest efektem imputacji.** W teście z tabeli okno wybierano po imputacji dopasowanej także na innych okresach. Bez imputacji (braki jako NaN) AUC na pacjentach wynosi 0,825, wobec 0,78 po imputacji.

Szczegóły: `3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcja 5.

---

## 6. Weryfikacja danych i decyzje

We wrześniu cały dotychczasowy materiał przejrzano pod kątem jednego pytania: *czy wynik modelu będzie mówił o tętniakach, czy o tym, jak powstały dane?* 04.10.2026 zapadły decyzje. Poniżej stan każdego ustalenia; liczby liczą skrypty `3-sgkf-split/przygotowanie/analiza_decyzji.py` (decyzje z 04.10) i `etap0_diagnostyka.py` (diagnostyka z września), a reguły agregacji `4-pu-setup/etap0_agregacja.py`.

| # | Ustalenie | Decyzja / status |
|---|---|---|
| 1 | **Rozjazd czasowy kohort** | bez korekty, ograniczenie; foldy wyrównane pod względem epoki |
| 2 | **Wyniki NEURO pochodzą najpewniej z hospitalizacji** | założenie robocze; zmienia interpretację, nie kod |
| 3 | **63 pacjentów w obu kohortach** | **rozwiązane**: pozytywni, rekordy KOR usunięte |
| 4 | **Selekcja cech z użyciem etykiety** | **rozwiązane**: selekcja cofnięta, jeden zestaw 38 cech |
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

### 6.4 Selekcja cech: dlaczego wracamy z 35 do 38

**Stan w czerwcu.** Raport przejściowy pracował na 35 cechach: w etapie 1 usunięto CRP, MONO i %MONO jako „najsłabiej związane z etykietą”. Raport imputacji z 11.06 zalecał plik z 35 cechami jako wejście do modelowania. Problem wskazała dopiero weryfikacja we wrześniu: selekcję wykonano z użyciem etykiety na całym zbiorze, więc przed analizą potwierdzającą trzeba albo wrócić do 38 cech, albo przenieść selekcję do walidacji krzyżowej (zapis w `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md`, sekcja 1 „Korelacje i selekcja cech”, dodany 18.09). 04.10.2026 sprawdziliśmy to dokładnie.

**Co zrobiono w czerwcu.** Skrypt `3-sgkf-split/przygotowanie/analiza_selekcji_cech.py` odtworzył procedurę z notebooka z wynikiem identycznym co do cyfry:
- dla czterech miar (Pearson, Spearman, Kendall, Phi-K) wybrano po 10 cech najsłabiej skorelowanych z etykietą,
- usunięto trzy cechy, które najczęściej trafiały na te listy.

**Dlaczego tej selekcji nie utrzymujemy:**
- **Złe kryterium w PU.** Etykieta oznacza kohortę, a nie chorobę. Kryterium „słaba korelacja z etykietą” usuwa cechy, które *nie* odróżniają kohort, a zostawia te, które je odróżniają, także przez epokę i sposób pozyskania danych. To odwrotność tego, czego potrzebujemy.
- **Przypadkowa granica.** Trzecie miejsce to remis czterech cech (%MONO, HCT, MPV, wiek pacjenta) po 3 „głosy”, rozstrzygnięty sortowaniem alfabetycznym. Ta sama metoda mogła usunąć wiek pacjenta.
- **Etykieta użyta na całym zbiorze**, także na pacjentach, którzy później trafiają do testu. Usunięte cechy miały słabą korelację z etykietą (|ρ| ≤ 0,02), a porównanie 35 i 38 cech wykazało niewielkie różnice imputacji. **Wpływu selekcji na modele PU nie oceniono; sama niska korelacja go nie wyklucza.** Cofnięcie selekcji usuwa błąd metodologiczny przed modelowaniem.

**Co mówią same cechy, niezależnie od etykiety:**
- MONO i %MONO są redundantne: %MONO = 100 − reszta rozmazu (korelacja 0,9995), a MONO ≈ %MONO × WBC / 100. Ich usunięcie dałoby się obronić, ale innym argumentem niż ten, którego użyto.
- CRP nie jest redundantne i nie odróżnia kohort (mediana 10,7 mg/l w KOR i 12,1 w NEURO). Brakuje go jednak w 48% rekordów NEURO, a jego uzupełnienia są zawyżone (sekcja 5.6).
- Imputacja: **brak istotnej różnicy** między 38 a 35 cechami (sekcja 5.5). Drobna przewaga 38 w teście kontrolowanym wynika częściowo z jego konstrukcji.

**Decyzja (04.10.2026): jeden zestaw 38 cech.** Powody:
1. Nie wybieramy cech z użyciem etykiety przed walidacją krzyżową.
2. Benchmark imputacji z etapu 2 był liczony na 38 cechach, więc parametry MICE pasują bez ponownego benchmarku.
3. Wybór nie zmienia istotnie jakości imputacji.

Ewentualny wybór cech przechodzi do modelowania, wewnątrz walidacji krzyżowej.

**Konsekwencje dla modelowania:**
- Procenty rozmazu (%NEUT, %LYMPH, %EO, %BAZO, %MONO) sumują się do 100%, więc w modelach liniowych są dokładnie współliniowe. Trzeba regularyzacji albo pominięcia jednego procentu. Modelom drzewiastym to nie przeszkadza.
- Wpływ CRP sprawdzamy w analizie wrażliwości na cechach o niskim odsetku braków (`4-pu-setup/PLAN_MODELOWANIA.md`).

### 6.5 Wartości niemożliwe i jednostki KREA

Czyszczenie kwantylowe z etapu 1 nie usunęło wartości niemożliwych, bo granice były statystyczne. W danych zostało K do 28 mmol/l, KREA do 196 mg/dl i Na 74 mmol/l. **Rozwiązanie:** reguły fizjologiczne, sekcja 5.3.

### 6.6 Reguła agregacji rekordów do pacjenta

NEURO ma średnio 3,65 rekordu na pacjenta, KOR 1,83. Reguła wrażliwa na liczbę pomiarów tworzy sygnał z samej częstości badania: maksimum koreluje z liczbą rekordów dla 19 z 38 cech, mediana dla 2 (obecne wejście; wrześniowa wersja na danych sprzed przygotowania: 17 z 35 i 2). **Rekomendacja:** mediana, ze średnią jako wrażliwością. To decyzja etapu modelowania (`4-pu-setup/PLAN_MODELOWANIA.md`); podział już wyrównuje liczbę rekordów między foldami.

### 6.7 Asymetria braków

MICE uzupełnia 26,0% komórek NEURO i 12,3% komórek KOR; kompletne wiersze, na których uczy się scaler, to w 96% KOR. Braki NRBC: 47,7% w NEURO, 2,6% w KOR. **Decyzja:** bez korekty, ograniczenie; udział raportowany w każdym foldzie.

---

## 7. Źródła prawdy: co jest czym w repozytorium

| Rola | Plik | Uwagi |
|---|---|---|
| dane surowe | `data/raw/*_merged_aggregated_1W_mean.csv` | nie modyfikować |
| wejście części A (38 cech, przed imputacją) | `data/imputation-inputs/aneurysm_concatted.csv` | stan po etapie 1 |
| **wejście do podziału i modelowania** | `data/processed/aneurysm_sgkf_input.csv` | wynik części A; 38 cech + 4 kolumny meta |
| spis zmian części A | `3-sgkf-split/przygotowanie/results/usuniete_*.csv` | każda usunięta wartość i rekord |
| **przydział pacjent → fold** | `3-sgkf-split/sgkf/results/pacjent_fold.csv` + `podzial_meta.json` | czytać przez `podzial.wczytaj_podzial()` |
| zaimputowane foldy | `3-sgkf-split/sgkf/results/foldy_imputowane_38/` | poza gitem; `python 3-sgkf-split/uruchom_sgkf.py` |
| parametry MICE | `2-imputation/RAPORT_IMPUTACJA.md`, przebieg 4 | 7 / 78 / 10 |
| zestaw 35 cech z raportu przejściowego | `data/processed/aneurysm_concatted_cleaned.csv` | historyczny, selekcja cofnięta 04.10 |
| imputacja globalna | `2-imputation/final/results/aneurysm_imputed_*.csv` | **tylko walidacja metody**, nie do modeli |

---

## 8. Decyzje

### 8.1 Podjęte 04.10.2026

| Nr | Decyzja | Wdrożenie |
|---|---|---|
| 0.1 | 63 pacjentów w obu kohortach: pozytywni, rekordy KOR usunięte | `przygotuj_dane.py` |
| 0.2 | rozjazd czasowy: bez korekty, ograniczenie | stratyfikacja po oknie w podziale |
| 0.4 | jeden zestaw 38 cech; selekcja z etapu 1 cofnięta | `3-sgkf-split/przygotowanie/analiza_selekcji_cech.py` (uzasadnienie) |
| 0.5 | wartości niemożliwe → brak (KREA, K, Na) | `przygotuj_dane.py` |
| 0.6 | podział `StratifiedGroupKFold`, `y` = 12 warstw, seed 42, zamrożony (31.10.2026: przebudowany z `StratifiedKFold` na tabeli pacjentów) | `podzial.py` |
| — | asymetria braków: bez korekty, ograniczenie | raport per fold |

### 8.2 Pozostające do potwierdzenia

| Nr | Kwestia | Stan | Kto |
|---|---|---|---|
| — | **skąd pochodzi kohorta KOR**: profil laboratoryjny wskazuje na pacjentów szpitalnych z lat 2020–2021 | pytanie | prowadzący / właściciel danych |
| 0.2b | czy wyniki NEURO pochodzą sprzed rozpoznania, z hospitalizacji, w której go rozpoznano, czy po leczeniu, czyli jaki jest cel predykcji: wcześniejsze wykrywanie czy rozpoznawanie profilu leczonych | założenie: hospitalizacja; bez odpowiedzi zakres eksploracyjny | prowadzący / właściciel danych |
| — | czyszczenie z etapu 1 ustawione osobno dla kohort (kwantyle, limit historii 11 / 38 rekordów): ograniczenie czy ujednolicenie przy ponownym zamrożeniu danych | ograniczenie; wspólny limit 11 usunąłby 943 rekordy 94 chorych (`analiza_decyzji.json`, sekcja L) | zespół + prowadzący |
| — | walidacja na niezależnej kohorcie albo kliniczna weryfikacja części wskazań | brak; bez niej wynik = odzyskiwanie znanych chorych w tym zbiorze | prowadzący |
| 0.2 | rozjazd czasowy: „bez korekty” wobec AUC 0,83 dla epoki w NEURO | decyzja zespołu do potwierdzenia; minimum: analiza wrażliwości we wspólnym oknie | zespół + prowadzący |
| — | obsługa śladu braków w modelowaniu (wzorzec braków AUC 0,82, podpis NRBC) | propozycja: analiza wrażliwości na cechach o niskim odsetku braków | zespół, przy planie modelowania |
| 0.3 | reguła agregacji rekordów do pacjenta | rekomendacja: mediana | zespół, przy planie modelowania |
| — | plan modelowania (wartość `q`, udział ukrywania, metody PU, modele kontrolne) | propozycja w `4-pu-setup/PLAN_MODELOWANIA.md` | zespół + prowadzący |

---

## 9. Ograniczenia, które idą do raportu końcowego

1. **Brak grupy kontrolnej**: nie da się zmierzyć rzeczywistego odsetka fałszywych alarmów w KOR.
2. **Założenie SCAR jest wątpliwe**: znani pozytywni to pacjenci objawowi, a niewykryte tętniaki są bezobjawowe.
3. **Rozjazd czasowy kohort** (KOR 2020–2021, NEURO 2000–2024). Epokę widać w wynikach chorych (AUC 0,83), więc model może ją rozpoznawać.
4. **Prawdopodobnie szpitalne pochodzenie wyników NEURO**; model rozpoznaje pacjenta hospitalizowanego, nie ryzyko przesiewowe.
5. **KOR prawdopodobnie nie jest populacją ogólną**, tylko szpitalną (CRP > 5 mg/l w 61% zmierzonych wyników), co ogranicza wnioski o zastosowaniu przesiewowym.
6. **Asymetria braków**: MICE odtwarza 26% profilu NEURO wobec 12% KOR. Sam wzorzec braków odróżnia kohorty z AUC 0,82, a uzupełnienia NRBC są rozpoznawalne.
7. **Przy brakach całymi panelami imputacja jest niewiele lepsza od średniej** (koagulogram, glukoza). Uzupełnienia niosą mało informacji o pacjencie, a przy skośnych rozkładach są wyższe od typowej zmierzonej wartości.
8. **Czyszczenie z etapu 1 ustawiane osobno dla kohort i na całych danych przed podziałem**: progi kwantylowe (NEURO stracił 3,2% rekordów, KOR 0,8%) i limit historii (11 rekordów w KOR, 38 w NEURO). To reguły zależne od źródła, czyli od etykiety, więc wyniki traktujemy jako eksploracyjne, a nie jako niezależnie zwalidowany cały proces.
9. **Imputacja deterministyczna**; mechanizm braków niezbadany.
10. **Reguły wartości niemożliwych są zachowawcze**; błędy ukryte w średniej tygodniowej mogły zostać. eGFR jest cenzurowane (60 / 90).
11. Części historycznych kroków filtracji (stan pośredni 73 276 / 7 611 rekordów, odcięcia percentylowe) **nie da się odtworzyć jednym skryptem**, bo istnieją tylko w notebookach i notatkach.
12. **Odzyskiwanie ukrytych pozytywnych mierzy odzyskiwanie znanych chorych w tym zbiorze.** Ukryci zachowują cechy kohorty NEURO, a sam wzorzec braków odzyskuje około 52% z nich w górnych 5% rankingu. Skuteczność wśród niewykrytych chorych w KOR wymagałaby niezależnej walidacji.
13. **Model daje ranking, a nie prawdopodobieństwo tętniaka.** Ani metoda PU, ani kalibracja nie usuwają skutków doboru kohort.

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
| kopie i wersje pośrednie tych samych plików w kilku miejscach (`data/interim/`, `2-imputation/knn-am/`), nie zawsze identyczne: `data/interim/cleaned/neuro_shortend.csv` ma 6 616 rekordów, a obowiązujący `data/imputation-inputs/neuro_shortend.csv` 6 651 | do zrobienia: jedna kopia w `data/`, wersje historyczne wyraźnie oznaczone |
| klucz `custom_id` łączy rok kalendarzowy z tygodniem ISO (stąd kolizja z sekcji 3.1) | do zrobienia przy kolejnym zamrożeniu danych: klucz z roku i tygodnia ISO albo z daty |
| wersje bibliotek różnią się między komputerami (sklearn 1.8 vs 1.6) | do zrobienia: wspólny `requirements.txt` z przypiętymi wersjami; do tego czasu porównywać odcisk wyników w `przebieg_imputacji_*.json` |

---

## Dokumenty szczegółowe

| Dokument | Zawartość |
|---|---|
| `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` | materiał do raportu przejściowego z 12.06 (historyczny) |
| `2-imputation/RAPORT_IMPUTACJA.md` | pełny benchmark imputacji |
| `3-sgkf-split/RAPORT_SGKF_MICE.md` | etap 3: decyzje i przygotowanie danych, podział, imputacja w foldach, ocena imputacji, diagnostyka wyników, przegląd merytoryczny |
| `1026_SPOTKANIE_PROWADZACY.md` | podsumowanie prac (czerwiec i wrzesień–październik) i materiał na spotkanie: decyzje do potwierdzenia, pytania, propozycja dalszych prac |
| `4-pu-setup/PLAN_MODELOWANIA.md` | propozycja planu modelowania PU, założenia i ograniczenia wejściowe |
