# Podsumowanie prac i materiał na spotkanie z prowadzącym

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)
**Stan na:** 06.10.2026 (branch `pu-modeling-setup-lk`)

**Jak czytać:** 5 minut wystarczy na sekcję 1. Pełny obraz w 15 minut dają sekcje 1–8. Szczegóły techniczne są w dokumentach wymienionych w sekcji 10, a każda liczba pochodzi ze skryptu w repozytorium.

**Cel spotkania:**
1. Podsumować, co zrobiliśmy: razem do raportu przejściowego (czerwiec) i od września do zamrożonego podziału SGKF.
2. Potwierdzić decyzje dotyczące danych (sekcja 6).
3. Uzyskać odpowiedzi, od których zależy plan modelowania (sekcja 7).
4. Ustalić, co robimy dalej i kto za co odpowiada (sekcja 8).

---

## 1. W pięciu punktach

1. **Dane są gotowe do modelowania.** Zbiór liczy 78 052 rekordy tygodniowe i 40 924 pacjentów, w tym 1 823 (4,45%) z potwierdzonym tętniakiem. Ma 38 cech. Pacjentów podzielono na 5 zamrożonych foldów, a braki uzupełnia MICE wewnątrz każdego foldu. Imputacja jest wykonana, sprawdzona i powtarzalna.
2. **Kwiecień–czerwiec, cały zespół:**
   - czyszczenie i filtracja danych, połączenie kohort z etykietą,
   - benchmark trzech metod imputacji, z którego wybraliśmy MICE z ExtraTrees,
   - pierwsza wersja podziału SGKF z imputacją w foldzie.

   Na tym oparł się raport przejściowy z 12.06.
3. **Wrzesień–październik, Łukasz:**
   - weryfikacja materiału z czerwca,
   - decyzje dotyczące danych: 63 pacjentów obecnych w obu kohortach, 752 wartości niemożliwe, powrót z 35 do 38 cech,
   - nowy, zamrożony podział (v2),
   - pełna imputacja w foldach z oceną jakości,
   - diagnostyka „skrótów” w danych.
4. **Najważniejszy wniosek merytoryczny: kohorty różnią się wieloma rzeczami poza chorobą.** Różnią się okresem, tym, które badania zlecano, i najpewniej typem pacjenta. Wysokie AUC P-vs-U nie będzie więc dowodem wykrywania tętniaka. Ocena musi się opierać na odzyskiwaniu ukrytych pozytywnych i na analizach wrażliwości.
5. **Czego potrzebujemy ze spotkania:**
   - potwierdzenia decyzji,
   - informacji o pochodzeniu danych (KOR, NEURO),
   - parametrów oceny (`q`, udział ukrywania),
   - podziału prac na etap modelowania.

---

## 2. Projekt i dane

**Zadanie.** Na podstawie wyników laboratoryjnych chcemy ocenić ryzyko tętniaka mózgu. Mamy klasę **P** (NEURO, pacjenci z rozpoznanym tętniakiem) i klasę **U** (KOR, nikt nie robił im obrazowania). `label = 0` oznacza „nieoznaczony”, a nie „zdrowy”. Dlatego stosujemy uczenie Positive-Unlabeled, a nie zwykłą klasyfikację.

**Rekord** to tydzień uśrednionych wyników laboratoryjnych pacjenta. Cechy (38):

| Grupa | Cechy |
|---|---|
| morfologia | HGB, RBC, HCT, MCV, MCH, MCHC, RDW, PLT, MPV, WPT |
| rozmaz | WBC, NEUT, LYMPH, MONO, EO, BAZO, IG, NRBC, każda w wartości bezwzględnej i jako % |
| biochemia | KREA, eGFR-MDRD, eGFRCKD, Na, K, GLU, CRP |
| koagulologia | PT, INR, APTT, WAPTT |
| demografia | wiek, płeć |

| | Dane surowe | Po etapie 1 (czerwiec) | Wejście do modelowania (październik) |
|---|---:|---:|---:|
| KOR: rekordy / pacjenci | 73 679 / 39 893 | 71 546 / 39 164 | 71 401 / 39 101 |
| NEURO: rekordy / pacjenci | 7 655 / 2 016 | 6 651 / 1 823 | 6 651 / 1 823 |
| razem pacjentów | | 40 924 (63 w obu kohortach) | 40 924 |
| rekordów na pacjenta: KOR / NEURO | | 1,83 / 3,65 | 1,83 / 3,65 |
| cechy | 128 kolumn | 38 (raport przejściowy: 35) | 38 |
| braki w komórkach cech: KOR / NEURO | | | 12,3% / 26,0% |

---

## 3. Część I: praca zespołowa (kwiecień – czerwiec 2026)

### 3.1 Kto co zrobił

| Osoba | Wkład | Gdzie |
|---|---|---|
| **Łukasz Kubik** | organizacja repozytorium i sprintów; skrypty czyszczące i kontrolne (duplikaty, anomalie medyczne); imputacja MICE; **wspólny benchmark trzech metod** i imputacja finalna; integracja podziału z imputacją w foldzie | `1-data-preparation/scripts-lk/`, `2-imputation/mice-lk/`, `2-imputation/final/` |
| **Liwia Florkiewicz** | pierwsze przetworzenie danych (rozpakowanie list wartości, jednostki, mapowanie badań); analiza populacyjna i wartości odstających; filtracja (wiek, próg braków, pacjenci nadreprezentowani); **połączenie KOR + NEURO z etykietą**; analiza korelacji; **pierwsza implementacja StratifiedGroupKFold**; notatki ze spotkań | `1-data-preparation/scripts-lf/`, `data_analysis_v1.ipynb`, `3-sgkf-split/archiwum/`, `docs/sprints/` |
| **Adrian Meredyk** | imputacja **KNN**: przegląd `k` × wagi, raporty dla KOR i NEURO | `2-imputation/knn-am/` |
| **Dominika Malisz** | imputacja **MissForest**: dobór liczby drzew z kryterium zatrzymania, raport | `2-imputation/missforest-dm/` |

### 3.2 Przebieg

| Data | Co |
|---|---|
| 13.04 | repozytorium i dane surowe |
| 23.04 | spotkanie: zakres czyszczenia, pomysł benchmarku imputacji na sztucznie zamaskowanych danych; pierwsze przetworzenie danych |
| 26.04 – 07.05 | czyszczenie NEURO i KOR; skrypty kontrolne; analiza populacyjna i wartości odstających; pierwsze testy KNN i MICE |
| 07.05 | spotkanie: filtry (wiek 18–100, 60% braków, ograniczenie nadreprezentowanych pacjentów), zbiór referencyjny do benchmarku |
| 18–28.05 | MissForest; dane przefiltrowane; rozbudowa KNN i MICE; połączenie KOR + NEURO z etykietą |
| 31.05 – 06.06 | wspólny benchmark imputacji (Optuna, 4 przebiegi), walidacja, imputacja finalna |
| 08.06 | analiza korelacji i usunięcie CRP, MONO, %MONO; pierwszy podział SGKF |
| 10–11.06 | integracja SGKF z MICE w foldzie, raport metodologiczny |
| **12.06** | **raport przejściowy** |

### 3.3 Co powstało

**Etap 1: przygotowanie danych.**
- **Czyszczenie strukturalne.** Listy `{'values': […]}`, czyli niedokończoną agregację tygodniową, zastąpiono średnią. Usunięto kolumny norm i jednostek oraz zmienne techniczne. Usunięto jeden duplikat rekordu. Płeć ujednolicono do 0/1.
- **Filtracja:**
  - wiek 18–100 lat,
  - próg 60% braków, osobno dla kolumn i dla pacjentów (usunięto 1 793 pacjentów KOR),
  - u pacjentów z bardzo długą historią zostawiono najnowsze pomiary; pacjentów z jednym pomiarem nie usuwano.
- **Wartości odstające.** Usuwano całe rekordy poza skrajnymi kwantylami, parametr po parametrze.
- **Połączenie kohort** z etykietą (NEURO = 1, KOR = 0): 78 197 rekordów × 38 cech.

**Etap 2: imputacja braków.**
- **Faza indywidualna.** Każda osoba sprawdziła jedną metodę (KNN, MissForest, MICE), ale każda innym protokołem, więc wyników nie dało się porównać.
- **Wspólny benchmark (Łukasz)** usunął ten problem:
  - wszystkie metody na tych samych kompletnych wierszach (KOR 20 663, NEURO 780),
  - ta sama maska: 10% wartości ukrytych losowo,
  - strojenie Optuną na **innej masce** niż ocena (seed 43 i seed 42); pierwszy przebieg, w którym maski były te same, odrzucono jako wyciek.

  | RMSE (skala [0, 1]) | MissForest | MICE | KNN |
  |---|---:|---:|---:|
  | KOR | **0,078** | 0,087 | 0,102 |
  | NEURO | 0,102 | **0,098** | 0,129 |

- **Decyzja: MICE**, czyli `IterativeImputer` z ExtraTrees, parametry 7 / 78 / 10, wspólne dla obu kohort. Żadna metoda nie wygrała na obu zbiorach. MICE wygrywa na mniejszym i trudniejszym NEURO, jest szybsze, a imputacja w foldzie wymaga jednego imputera dla połączonych danych. Ranking potwierdziła niezależna maska (seed 7) i walidacja na 10 maskach.

**Etap 3, wersja 1: podział.**
- Liwia zaimplementowała `StratifiedGroupKFold` (5 foldów, grupowanie po `patient_id`). Wszystkie rekordy pacjenta trafiają po jednej stronie podziału.
- Łukasz zintegrował podział z imputacją: scaler i MICE uczą się tylko na części treningowej foldu.

Te dwa zabezpieczenia przed wyciekiem były poprawne i zostały zachowane w wersji 2.

### 3.4 Co zostało otwarte po raporcie przejściowym

Część była zastrzeżona już w materiale czerwcowym (`0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md`, sekcja 6):
1. 63 pacjentów ma rekordy w obu kohortach i nie miało ustalonej etykiety.
2. Zostały wartości niemożliwe fizjologicznie, a jednostki kreatyniny są mieszane.
3. Cechy CRP, MONO i %MONO wybrano z użyciem etykiety na całym zbiorze. Już wtedy zapisaliśmy, że selekcję trzeba cofnąć albo przenieść do walidacji krzyżowej.
4. Podział nie był tasowany ani zapisywany.
5. Kohorty pochodzą z różnych okresów, a dane nie mają daty rozpoznania tętniaka.

---

## 4. Część II: prace od września (Łukasz), do zamrożonego SGKF

Od 11.09 do 05.10 powstało 25 commitów: około 3 000 linii kodu i 2 800 linii dokumentacji. Wszystko leży na branchu `pu-modeling-setup-lk`. **`master` kończy się na stanie z 11.09**, więc reszta zespołu nie widziała jeszcze tych zmian w głównej gałęzi.

### 4.1 Wrzesień: weryfikacja i diagnostyka

- **Porządki.** Repozytorium przeorganizowano według etapów (`1-…`, `2-…`, `3-…`, `4-…`) i spisano materiał do raportu przejściowego.
- **Plan modelowania PU** (`4-pu-setup/PLAN_MODELOWANIA.md`). Plan opiera się na trzech elementach:
  - ocena przez ukrywanie części znanych chorych w puli U,
  - główna metryka `RecallHidden@q`, liczona na pacjentach,
  - walidacja zagnieżdżona.
- **Diagnostyka danych.** Przejrzeliśmy cały dotychczasowy materiał pod kątem jednego pytania: *czy wynik modelu będzie mówił o tętniakach, czy o tym, jak powstały dane?* Najważniejsze ustalenia:

  | Ustalenie | Liczby |
  |---|---|
  | **kohorty niemal rozłączne w czasie** | KOR: 99,5% rekordów z lat 2020–2021; NEURO: 2000–2024, większość przed 2020. We wspólnym oknie dat jest 271 z 1 823 pozytywnych |
  | **wyniki NEURO najpewniej pochodzą z hospitalizacji** | u 1 215 chorych z więcej niż jednym rekordem mediana rozpiętości badań wynosi 35 dni |
  | **reguła agregacji rekordów do pacjenta tworzy sygnał** | maksimum koreluje z liczbą badań w 17 z 35 cech, mediana w 2; rekomendacja: mediana |
  | **podział nie jest powtarzalny** | ta sama konfiguracja SGKF daje inne foldy w sklearn 1.6 i 1.8 |

- **Prototyp pipeline'u PU.** Stan sprzed spotkania 21.09, zarchiwizowany 25.09: ukrywanie pozytywnych, metryki pacjentowe, funkcja celu, 17 testów. Po spotkaniu modelowanie odłożono, a priorytetem stało się domknięcie SGKF. Branch usunięto, ale kod da się odzyskać z commita `2cb6d1a` (sekcja 9).

### 4.2 04.10: decyzje dotyczące danych (część A etapu 3)

Decyzje wdraża osobny skrypt (`3-sgkf-split/przygotowanie/przygotuj_dane.py`). Zapisuje on spis każdej zmienionej komórki i każdego usuniętego rekordu, a jego działanie pilnują testy.

| Problem | Decyzja | Skutek |
|---|---|---|
| 63 pacjentów w obu kohortach | zostają **pozytywni**, ich rekordy KOR usunięte | −145 rekordów KOR; liczba pacjentów bez zmian |
| KREA > 50 mg/dl | → brak (błąd jednostki, najpewniej µmol/l) | 210 wartości |
| KREA > 10 mg/dl przy eGFR ≥ 30 | → brak (sprzeczność z eGFR) | 516 wartości |
| K > 15 mmol/l, Na < 80 mmol/l | → brak (niezgodne z życiem) | 25 + 1 wartość |
| skrajne, ale możliwe (K 9–15, Na 80–100, WBC > 200, KREA > 10 przy eGFR < 30) | **zostają** | 327 wartości |
| zestaw cech | **38**: selekcja z etapu 1 cofnięta | CRP, MONO, %MONO wracają |
| rozjazd czasowy, asymetria braków | **bez korekty**, opisane jako ograniczenia | — |

**Dlaczego reguła dla kreatyniny odwołuje się do eGFR.** Laboratorium wylicza eGFR z kreatyniny, więc oba pomiary muszą być spójne. Do KREA ≈ 10 mg/dl mediana eGFR spada zgodnie z fizjologią i przy KREA 5–10 wynosi 8. Powyżej 10 wraca do normy: przy KREA 20–50 mediana eGFR wynosi 60. To błąd jednostki albo wpisu, a nie prawdziwy pacjent.

**Dlaczego 38 cech, a nie 35:**
1. W PU etykieta oznacza kohortę, nie chorobę. Kryterium „słaba korelacja z etykietą” usuwa więc cechy, które *nie* odróżniają kohort, czyli działa w złą stronę.
2. Granicę wyboru rozstrzygnął remis czterech cech, a remis zdecydował alfabet. Tą samą metodą mógł wypaść wiek pacjenta.
3. Realny wpływ tej selekcji na wyniki był znikomy (|ρ| ≤ 0,02). To porządkowanie metody, a nie naprawa błędu w czerwcowych wynikach.

### 4.3 Podział v2 (część B)

| Problem w v1 | Rozwiązanie w v2 |
|---|---|
| brak tasowania: foldy zależały od kolejności wierszy w CSV | `StratifiedKFold` na tabeli pacjentów, `shuffle=True`, seed 42; grupowanie jest spełnione z konstrukcji |
| przydział niezapisany, zależny od wersji sklearn | przydział zapisany w `pacjent_fold.csv` z odciskiem pliku wejściowego; zmiana danych unieważnia podział |
| stratyfikacja tylko po etykiecie | 12 warstw: etykieta × okno czasowe × liczba rekordów (1 / 2 / 3–4 / 5+) |

Porównanie strategii, rozstęp między foldami, najgorszy wynik z 5 ziaren:

| Strategia | Udział pozytywnych | NEURO z okna czasowego KOR | NEURO z 5+ rekordami |
|---|---:|---:|---:|
| v1 (SGKF, bez tasowania) | 0,02 pp | 4,06 pp | 0,27 pp |
| pacjenci, stratyfikacja tylko po etykiecie | 0,01 pp | 4,91 pp | 8,49 pp |
| **v2 (etykieta × okno × liczba rekordów)** | **0,01 pp** | **0,55 pp** | **0,22 pp** |

Każdy fold ma 8 184–8 185 pacjentów, w tym 364–365 pozytywnych (4,45–4,46%).

### 4.4 Imputacja w foldach i jej ocena

- **Przebieg** zajmuje 15,6 minuty na 5 foldów. W każdym foldzie jest 0 wspólnych pacjentów train/test, 0 braków i 0 wartości ujemnych po imputacji. Dwa niezależne przebiegi dały identyczne wyniki.
- **Zakres uzupełnień.** MICE uzupełnia około 12% komórek KOR i 25–28% komórek NEURO.
- **Ocena jakości.** Użyliśmy protokołu z etapu 2, ale out-of-fold: imputer dopasowany na treningu foldu, oceniane kompletne wiersze części testowej. Kolumna „wobec średniej” to błąd MICE podzielony przez błąd uzupełniania średnią; 1,0 oznacza brak przewagi nad średnią.

  | Maska | KOR: RMSE / wobec średniej | NEURO: RMSE / wobec średniej |
  |---|---|---|
  | losowe 10% komórek, jak w etapie 2 | 0,083 / **0,61** | 0,080 / **0,65** |
  | realistyczne wzorce braków (całe panele) | 0,108 / **0,98** | 0,079 / **0,89** |

  Przy losowych brakach MICE działa tak dobrze jak w benchmarku, jednakowo dla obu kohort. W prawdziwych danych braki idą jednak całymi panelami: koagulogramu i glukozy brakuje w około połowie rekordów. Wtedy uzupełnienie jest bliskie średniej z treningu. **Benchmark z etapu 2 przeszacowywał więc użyteczność imputacji.**
- **38 czy 35 cech.** Wybór nie zmienia istotnie imputacji. W teście kontrolowanym RMSE wynosi 0,0846 dla 38 cech i 0,0859 dla 35. W prawdziwych foldach uzupełnienia różnią się średnio o 0,11 odchylenia standardowego.

### 4.5 Diagnostyka „skrótów”

Na zamrożonych foldach sprawdziliśmy, jak łatwo odróżnić grupy, które nie powinny być odróżnialne, gdyby różniła je wyłącznie choroba. To diagnostyka danych, a nie model ryzyka.

| Test (HistGradientBoosting, część testowa foldów) | AUC |
|---|---:|
| NEURO vs KOR, 38 cech po imputacji (pacjenci) | **0,93** |
| NEURO vs KOR, **tylko wzorzec braków**, czyli które badania zlecono (pacjenci) | **0,82** |
| NEURO vs KOR, tylko rekordy ze wspólnego okna dat (pacjenci) | 0,80 |
| NEURO z okna dat KOR vs NEURO z innych lat (rekordy) | **0,83** |

**Kontrola z 06.10** (`3-sgkf-split/sgkf/kontrola_epoki.py`). Ostatni test liczono na danych po imputacji, więc nie wiadomo było, gdzie siedzi sygnał epoki. Na danych przed imputacją wychodzi tak:
- sam wzorzec braków daje AUC **0,76**,
- wartości zmierzone, bez imputacji, dają tylko 0,56–0,66 (morfologia: 0,56; morfologia z rozmazem: 0,66).

Epokę zdradza więc głównie to, *które badania zlecano* w danym okresie. W samych wartościach sygnał jest umiarkowany.

**Profil KOR nie wygląda na populację ogólną.** CRP > 5 mg/l ma 61% rekordów KOR (NEURO 62%), WBC > 10 G/l 37%, glukozę > 99 mg/dl 64%. KOR wygląda na pacjentów szpitalnych z lat 2020–2021, czyli z okresu pandemii.

**Ślad imputacji.** Zmierzone NRBC to w 78% dokładne zera, a uzupełnione nigdy nie są zerem. NRBC brakuje w 48% rekordów NEURO i w 3% KOR. Uzupełniona wartość w praktyce oznacza więc „rekord NEURO”.

**Wnioski dla modelowania:**
- AUC P-vs-U raportujemy tylko jako rozdzielność kohort.
- Główną miarą jest odzyskiwanie ukrytych pozytywnych.
- Niezbędne są analizy wrażliwości: model na wartościach zmierzonych z paneli o niskim odsetku braków, model we wspólnym oknie dat i model bez NRBC.

### 4.6 Porządki i odtwarzalność

- **Jedno polecenie.** Cały etap 3 uruchamia `python 3-sgkf-split/uruchom_sgkf.py`: około 35 minut, logi z każdego kroku. Opcja `--bez-mice` pozwala sprawdzić etap w około 2 minuty.
- **Testy.** 16 testów automatycznych (6 w części A, 10 w części B). Pilnują m.in., że plik wejściowy w repozytorium jest dokładnie wynikiem kodu.
- **Jeden branch roboczy** zamiast pięciu, dokumenty nazwane datą.

---

## 5. Czerwiec a październik: co się zmieniło

| Element | Raport przejściowy (06.2026) | Teraz (10.2026) |
|---|---|---|
| wejście do modelowania | `aneurysm_concatted_cleaned.csv`: 78 197 rekordów, 35 cech | `aneurysm_sgkf_input.csv`: 78 052 rekordy, **38 cech** |
| 63 pacjentów w obu kohortach | nierozstrzygnięte | pozytywni, rekordy KOR usunięte |
| wartości niemożliwe | częściowo (kwantyle statystyczne) | 752 wartości → brak według reguł fizjologicznych |
| podział | SGKF na rekordach, bez tasowania, niezapisany | po pacjentach, 12 warstw, seed 42, **zapisany z odciskiem danych** |
| imputacja w foldach | kod gotowy | **pełny przebieg wykonany**, powtarzalny |
| ocena imputacji | benchmark na losowej masce | + ocena out-of-fold + realistyczne wzorce braków |
| ryzyka w danych | opisane jako ograniczenia | **zmierzone** (AUC skrótów, profil KOR, ślad NRBC) |
| odtwarzalność | notebooki, ścieżki z Colab | jedno polecenie, 16 testów, logi |
| **bez zmian** | metoda imputacji i jej parametry, imputacja w foldzie, grupowanie po pacjencie, czyszczenie z etapu 1 | |

---

## 6. Decyzje do potwierdzenia

Decyzje przygotował i wdrożył Łukasz 04.10, gdy reszta zespołu nie pracowała nad projektem. **Są propozycją do akceptacji przez zespół i prowadzącego.** Zmiana każdej z nich to edycja jednej reguły w `przygotuj_dane.py` i ponowne uruchomienie etapu 3 (około 35 minut).

| # | Kwestia | Decyzja | Uzasadnienie |
|---|---|---|---|
| 1 | 63 pacjentów w obu kohortach | pozytywni, 145 rekordów KOR usuniętych | pacjent nie może być jednocześnie P i U; bez daty rozpoznania nie wiadomo, czy rekordy KOR są sprzed choroby |
| 2 | wartości niemożliwe | KREA > 50 albo > 10 przy eGFR ≥ 30, K > 15, Na < 80 → brak, uzupełniane w foldzie | usuwamy tylko to, co niemożliwe albo sprzeczne z innym pomiarem |
| 3 | zestaw cech | 38, selekcja z etapu 1 cofnięta | sekcja 4.2; ewentualny wybór cech tylko wewnątrz walidacji krzyżowej |
| 4 | rozjazd czasowy kohort | bez korekty, ograniczenie + analiza wrażliwości we wspólnym oknie dat | obcięcie do wspólnego okna zostawia 271 z 1 823 pozytywnych |
| 5 | asymetria braków | bez korekty, raportowana w każdym foldzie; analiza wrażliwości na wartościach zmierzonych | sam wzorzec braków odróżnia kohorty (AUC 0,82) |
| 6 | podział | 5 foldów po pacjentach, 12 warstw, seed 42, zamrożony | rozstęp udziału pozytywnych 0,01 pp; niezależny od wersji bibliotek |

---

## 7. Pytania do prowadzącego

W kolejności wpływu na projekt:

1. **Skąd pochodzi kohorta KOR?** Jaki oddział lub jakie kryteria włączenia? Czy to pacjenci hospitalizowani i czy ma to związek z okresem pandemii? Od odpowiedzi zależy interpretacja klasy U i sens zastosowania przesiewowego.
2. **Skąd pochodzą wyniki NEURO?** Czy są sprzed rozpoznania tętniaka, czy z hospitalizacji, w trakcie której go rozpoznano i leczono? Czy da się uzyskać datę rozpoznania?
3. **Rozjazd czasowy:** czy akceptuje Pan „bez korekty” z analizą wrażliwości we wspólnym oknie dat?
4. **NRBC i %NRBC:** wykluczyć z modelowania? To ślad imputacji, a u dorosłych cechy mają niską wartość kliniczną.
5. **Parametry oceny:**
   - `q`, czyli ilu pacjentów realnie da się skierować na obrazowanie,
   - główny udział ukrywanych pozytywnych (propozycja: 40%),
   - metody PU (propozycja: PU Bagging i korekta Elkana–Noto obok modeli bazowych),
   - reguła agregacji rekordów do pacjenta (propozycja: mediana).
6. **Termin i oczekiwany zakres raportu końcowego.**

Pełna lista kwestii technicznych: `4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 7.

---

## 8. Co dalej: propozycja etapu 4 i podziału prac

**Zanim powstanie pierwszy model, trzeba zamrozić protokół:**
- maski ukrywania (całych pacjentów, identyczne dla wszystkich modeli),
- metryki liczone na pacjentach z ustalonym `q`,
- scenariusz główny i analizy wrażliwości.

Strojenia `q` pod wynik nie dopuszczamy.

| # | Pakiet | Zakres | Zależy od | Kto |
|---|---|---|---|---|
| 0 | przegląd i scalenie | zespół przegląda zmiany od 11.09; scalenie `pu-modeling-setup-lk` → `master`; wspólny `requirements.txt` z przypiętymi wersjami | — | wszyscy |
| 1 | protokół oceny | ukrywanie pozytywnych, `RecallHidden@q`, Lift@q, RecallKnown, funkcja celu; punkt wyjścia: prototyp `2cb6d1a` | pytanie 5 z sekcji 7 | do ustalenia |
| 2 | pipeline modelowania | walidacja zagnieżdżona na zamrożonych foldach, imputacja od nowa w foldach wewnętrznych, agregacja do pacjenta | 1 | do ustalenia |
| 3 | modele bazowe | regresja logistyczna (z regularyzacją, bo procenty rozmazu sumują się do 100%), Random Forest, jeden boosting | 2 | do ustalenia |
| 4 | metody PU | PU Bagging, Elkan–Noto (class prior szacowany tylko na treningu) | 2 | do ustalenia |
| 5 | analizy wrażliwości | wartości zmierzone z paneli o niskim odsetku braków; wspólne okno dat; bez NRBC; bez 63 pacjentów mieszanych | 3, 4 | do ustalenia |
| 6 | raport końcowy | metodologia etapów 1–3 (materiał gotowy), ograniczenia, wyniki | wszystko | do ustalenia |

Pakiety 3 i 4 można prowadzić równolegle, gdy gotowe są pakiety 1 i 2.

**Koszt obliczeń.** MICE zajmuje około 3 minut na fold (Mac M-series). Walidacja zagnieżdżona 5 × 3 oznacza około godziny na jedną konfigurację preprocessingu. Wynik preprocessingu nie zależy od hiperparametrów klasyfikatora, więc da się go zapisać raz na split i używać ponownie.

**Czego się spodziewać.** Przy 40% ukrywania na fold przypada około 146 ukrytych chorych, więc rozdzielczość `RecallHidden@q` wynosi około 0,7 pp. We wspólnym oknie dat zostałoby około 22 ukrytych na fold. Dlatego to okno nadaje się na analizę wrażliwości, ale nie na scenariusz główny.

---

## 9. Uwagi z weryfikacji repozytorium (06.10.2026)

Przed spotkaniem sprawdziliśmy, czy dokumenty zgadzają się z danymi i kodem. Środowisko: Python 3.13, sklearn 1.9.1, pandas 3.0, czyli inne niż to, w którym powstały wyniki.

**Co się zgadza:**
- liczby z danych: rekordy, pacjenci, pozytywni, 63 pacjentów mieszanych, rozkład lat, braki per kohorta, kompletne wiersze,
- testy: 16/16 przechodzą,
- zapisany podział pasuje do danych (`podzial.py --sprawdz`),
- przydział pacjentów do foldów odtwarza się identycznie,
- część A odtwarza plik wejściowy (test zgodności przechodzi).

Pełnej imputacji (około 35 minut) nie powtarzaliśmy. Jej wyniki opieramy na zapisanych logach i odciskach.

**Poprawione przy okazji:**
- liczba testów w `1026_REALIZACJA_DO_SGKF.md`: 16 (6 + 10), a nie 17,
- `2-imputation/RAPORT_IMPUTACJA.md`, sekcja 4.7: nadal stało tam, że globalnie imputowany plik 35 cech jest „właściwym wejściem do pipeline PU”; dodaliśmy adnotację, że to nieaktualne,
- przy decyzji o parametrach MICE dopisaliśmy jego wynik z parametrami finalnymi (0,0861 / 0,0984). Tabela porównawcza pochodzi z wcześniejszego przebiegu.

**Do zrobienia:**
1. **`master` nie zawiera prac z września i października.** Zespół powinien je przejrzeć przed scaleniem (pakiet 0).
2. **Odcisk danych zależy od wersji pandas.** Ponowne przygotowanie danych w pandas 3.x zmienia ostatnią cyfrę kilku wartości, więc zmienia się odcisk pliku. Zapisany podział odmawia wtedy działania, choć przydział pacjentów jest identyczny. Rozwiązanie: przypięte wersje bibliotek albo odcisk liczony z zaokrąglonych wartości.
3. **Kod prototypu PU** (commit `2cb6d1a`, 25.09) jest dostępny tylko po SHA, bo branch usunięto. Warto oznaczyć go tagiem, zanim GitHub go usunie.
4. **Drobne porządki:**
   - plik `scripts-lf/aneurysm_data_analysis.ipynb` to zapisana strona HTML,
   - notebooki mają ścieżki z Colab,
   - kopie tych samych CSV leżą w kilku miejscach (`data/interim/` to 138 MB),
   - na serwerze został nieaktualny branch `dataset_analysis_lf`.

---

## 10. Gdzie co jest

| Co | Plik |
|---|---|
| **wejście do modelowania** (38 cech, przed imputacją) | `data/processed/aneurysm_sgkf_input.csv` |
| przydział pacjent → fold | `3-sgkf-split/sgkf/results/pacjent_fold.csv` (czytać przez `podzial.wczytaj_podzial()`) |
| cały etap 3 jednym poleceniem | `python 3-sgkf-split/uruchom_sgkf.py` |
| pełny opis etapów 1–3, kto co zrobił, decyzje | `1026_REALIZACJA_DO_SGKF.md` |
| szczegóły techniczne etapu 3 | `3-sgkf-split/RAPORT_SGKF_MICE.md` |
| benchmark imputacji | `2-imputation/RAPORT_IMPUTACJA.md` |
| propozycja planu modelowania | `4-pu-setup/PLAN_MODELOWANIA.md` |
| stan na raport przejściowy (historyczny) | `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` |
| **nie do modeli** (imputacja przed podziałem) | `2-imputation/final/results/aneurysm_imputed_*.csv` |
