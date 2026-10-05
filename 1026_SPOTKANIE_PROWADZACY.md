# Spotkanie z prowadzącym: stan prac i decyzje do potwierdzenia

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Dla:** dr inż. Patryk Jasik
**Stan na:** 04.10.2026

**Cel spotkania:**
1. Przedstawić stan: dane są gotowe do modelowania (podział zamrożony, imputacja w foldach).
2. Potwierdzić decyzje dotyczące danych, które podjął zespół.
3. Uzyskać odpowiedzi na pytania, od których zależy plan modelowania.

Pełny opis: `1026_REALIZACJA_DO_SGKF.md`; szczegóły techniczne: `3-sgkf-split/RAPORT_SGKF_MICE.md`; propozycja planu modelowania: `4-pu-setup/PLAN_MODELOWANIA.md`.

---

## 1. Stan prac w skrócie

| Etap | Stan |
|---|---|
| 1. Przygotowanie danych | zamknięte: 78 197 rekordów tygodniowych, 40 924 pacjentów |
| 2. Benchmark imputacji | zamknięty: MICE (ExtraTrees) wybrane spośród KNN, MissForest i MICE |
| 3A. Przygotowanie danych do podziału | **nowe**: decyzje dotyczące danych wdrożone kodem, z pełnym spisem zmian; 78 052 rekordy, 1 823 pacjentów pozytywnych |
| 3B. Podział i imputacja w foldach | **nowe**: 5 foldów po pacjentach, zamrożone; MICE wewnątrz każdego foldu; wynik powtarzalny |
| Weryfikacja | **nowe**: ocena imputacji protokołem z etapu 2 i diagnostyka „skrótów” w danych |
| 4. Modelowanie PU | propozycja planu, czeka na ustalenia z tego spotkania |

Każda liczba w materiałach pochodzi ze skryptu w repozytorium, a cały etap 3 odtwarza jedno polecenie: `python 3-sgkf-split/uruchom_sgkf.py`.

---

## 2. Decyzje zespołu do potwierdzenia

| # | Problem | Decyzja | Uzasadnienie |
|---|---|---|---|
| 1 | 63 pacjentów ma rekordy w obu kohortach | zostają **pozytywni**, ich 145 rekordów KOR usunięte | pacjent nie może być jednocześnie P i U; bez daty rozpoznania nie wiadomo, czy rekordy KOR są sprzed choroby |
| 2 | wartości niemożliwe fizjologicznie | → brak danych, uzupełniane w foldzie: **KREA > 50 mg/dl** albo **KREA > 10 przy eGFR ≥ 30** (726), **K > 15 mmol/l** (25), **Na < 80 mmol/l** (1) | usuwamy tylko to, co niemożliwe albo sprzeczne z innym pomiarem; skrajne, ale możliwe (K 9–15, Na 80–100, WBC > 200) zostają |
| 3 | zestaw cech (w raporcie przejściowym było 35) | **38 cech**; usunięcie CRP, MONO i %MONO z etapu 1 **cofnięte**, zgodnie z zastrzeżeniem z czerwca | w PU etykieta to kohorta, więc kryterium „słaba korelacja z etykietą” działa w złą stronę; granicę rozstrzygnął remis alfabetyczny; 38 cech jest zgodne z benchmarkiem imputacji. Wpływ na wyniki znikomy (\|ρ\| ≤ 0,02, imputacja bez istotnej różnicy): to porządkowanie metody, nie naprawa błędu |
| 4 | rozjazd czasowy kohort | **bez korekty**, opisany jako ograniczenie | ograniczenie do wspólnego okna dat zostawia 271 pozytywnych z 1 823 |
| 5 | asymetria braków | **bez korekty**, raportowana w każdym foldzie | — |
| 6 | podział danych | 5 foldów po pacjentach, stratyfikacja etykieta × okno czasowe × liczba rekordów, seed 42, **zapisany** | poprzednia wersja była nietasowana i zależna od wersji biblioteki; rozstęp udziału pozytywnych między foldami 0,01 pp |

**Reguła dla kreatyniny opiera się na danych.** eGFR laboratorium liczy z kreatyniny, więc oba pomiary muszą być spójne. Do KREA ≈ 10 mg/dl mediana eGFR spada zgodnie z fizjologią: przy 5–10 mg/dl wynosi 8. Powyżej 10 wraca do normy (60 przy 20–50 mg/dl), co oznacza błąd jednostki (µmol/l) albo wpisu.

---

## 3. Co pokazały dane po podziale

### 3.1 Imputacja działa, ale przy brakach całymi panelami niewiele wnosi

Jakość imputacji zmierzyliśmy tak jak w etapie 2: ukryliśmy znane wartości w kompletnych wierszach części testowej każdego foldu i porównaliśmy uzupełnienia z prawdą. Imputer był dopasowany na treningu foldu, jak w produkcji. „Względem średniej” to błąd MICE podzielony przez błąd uzupełniania średnią; 1,0 oznacza brak przewagi nad średnią.

| Maska | KOR: RMSE / względem średniej | NEURO: RMSE / względem średniej | Etap 2 |
|---|---|---|---|
| losowe 10% komórek (jak w etapie 2) | 0,083 / **0,61** | 0,080 / **0,65** | KOR 0,086 / NEURO 0,098 |
| realistyczne wzorce braków | 0,108 / **0,98** | 0,079 / **0,89** | — |

Przy losowych brakach MICE w produkcji działa co najmniej tak dobrze jak w benchmarku i jednakowo dla obu kohort. W prawdziwych danych braki idą jednak całymi panelami: koagulogram i glukozy brakuje w około połowie rekordów obu kohort. Wtedy uzupełnienia są bliskie średniej z treningu, a benchmark z etapu 2 przeszacowywał użyteczność imputacji.

### 3.2 Kohorty różnią się wieloma rzeczami poza chorobą

Na zamrożonych foldach mierzyliśmy, jak dobrze da się odróżnić grupy, które nie powinny być odróżnialne, gdyby różniła je wyłącznie choroba:

| Test | AUC (pacjenci) |
|---|---:|
| NEURO vs KOR, 38 cech po imputacji | **0,93** |
| NEURO vs KOR, **tylko wzorzec braków** (które badania zlecono) | **0,82** |
| NEURO vs KOR, tylko rekordy ze wspólnego okna dat | 0,80 |
| NEURO z lat 2020–2021 vs NEURO z innych lat (rekordy) | **0,83** |

- **Wysokie AUC P-vs-U w modelowaniu nie będzie dowodem wykrywania tętniaka.** Kohorty odróżnia już sam wzorzec braków.
- **Epokę widać w wynikach laboratoryjnych samych chorych.** Rozjazd czasowy jest realnym ryzykiem, a nie tylko teoretycznym.
- **Imputacja zostawia ślad.** Zmierzone NRBC to w 78% dokładne zera, a uzupełnione nigdy nie są zerem. NRBC brakuje w 48% rekordów NEURO i w 3% KOR, więc uzupełniona wartość w praktyce oznacza NEURO.

### 3.3 KOR nie wygląda na populację ogólną

| Parametr | Granica | KOR | NEURO |
|---|---|---:|---:|
| CRP | > 5 mg/l | **61,4%** | 62,2% |
| WBC | > 10 G/l | 36,9% | 32,5% |
| glukoza | > 99 mg/dl | 64,0% | 57,7% |
| HGB | < 12 g/dl | 39,6% | 43,2% |

Profil KOR jest zapalny i „szpitalny”, w części parametrów bardziej odbiegający od normy niż NEURO. Wszystkie rekordy KOR pochodzą z lat 2020–2021. Jeśli KOR to pacjenci hospitalizowani, a nie populacja ogólna, zmienia to interpretację klasy U i sens zastosowania przesiewowego.

---

## 4. Pytania do prowadzącego

W kolejności wpływu na projekt:

1. **Skąd pochodzi kohorta KOR?** Jaki oddział lub kryteria włączenia, czy to pacjenci hospitalizowani, czy ma to związek z okresem pandemii?
2. **Skąd pochodzą wyniki NEURO?** Sprzed rozpoznania tętniaka czy z hospitalizacji, w trakcie której go rozpoznano i leczono? Rozpiętość badań u chorych (mediana 35 dni) wskazuje na pojedynczą hospitalizację.
3. **Rozjazd czasowy:** czy akceptuje Pan decyzję „bez korekty”, jeśli w modelowaniu dodamy analizę wrażliwości we wspólnym oknie dat?
4. **Ślad braków:** czy wykluczyć NRBC i %NRBC (podpis imputacji, niska wartość kliniczna u dorosłych)? Czy model na cechach o niskim odsetku braków jako analiza wrażliwości jest wystarczający?
5. **Plan modelowania** (`4-pu-setup/PLAN_MODELOWANIA.md`):
   - wartość `q`, czyli ilu pacjentów realnie da się skierować na obrazowanie,
   - główny udział ukrywanych pozytywnych (propozycja 40%),
   - metody PU (propozycja: PU Bagging i korekta Elkana–Noto obok modeli bazowych),
   - jednostka treningu i reguła agregacji rekordów do pacjenta (propozycja: mediana).

---

## 5. Proponowane następne kroki

1. Uwzględnić odpowiedzi ze spotkania w planie modelowania i zamrozić protokół: maski ukrywania, metryki z `q`, scenariusz główny.
2. Zbudować pipeline modelowania na zamrożonych foldach, z walidacją zagnieżdżoną do strojenia.
3. Porównać modele bazowe z metodami PU na metryce głównej `RecallHidden@q`.
4. Przeprowadzić analizy wrażliwości: wspólne okno dat, cechy o niskim odsetku braków, ewentualnie bez NRBC.
