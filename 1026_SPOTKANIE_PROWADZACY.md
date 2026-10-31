# Podsumowanie prac i materiał na spotkanie z prowadzącym

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Prowadzący:** dr inż. Patryk Jasik (IFiIS)
**Stan na:** 31.10.2026 (branch `pu-modeling-setup-lk`; podział przebudowany na `StratifiedGroupKFold` i cały etap 3 przeliczony 31.10, sekcja 9)

**Jak czytać:** 5 minut wystarczy na sekcję 1. Pełny obraz w 15 minut dają sekcje 1–8. Szczegóły techniczne są w dokumentach wymienionych w sekcji 10, a każda liczba pochodzi ze skryptu w repozytorium.

**Cel spotkania:**
1. Podsumować, co zrobiliśmy: razem do raportu przejściowego (czerwiec) i od września do zamrożonego podziału SGKF.
2. Potwierdzić decyzje dotyczące danych (sekcja 6).
3. Uzyskać odpowiedzi, od których zależy plan modelowania (sekcja 7).
4. Ustalić, co robimy dalej i kto za co odpowiada (sekcja 8).

---

## 1. W pięciu punktach

**Oznaczenia:** P to znani chorzy z kohorty NEURO, U to pacjenci z kohorty KOR bez znanego rozpoznania; U nie oznacza potwierdzonych zdrowych. „Ukrywanie” oznacza czasowe potraktowanie części znanych chorych jako U, aby sprawdzić, czy model umieści ich wysoko w rankingu.

1. **Etapy 1–3 są wykonane; decyzje o danych wymagają potwierdzenia.** Zbiór liczy 78 052 rekordy tygodniowe i 40 924 pacjentów, w tym 1 823 (4,45%) z potwierdzonym tętniakiem. Ma 38 cech. Pacjentów podzielono na 5 zamrożonych foldów, a braki uzupełnia MICE wewnątrz każdego foldu. Wynik imputacji sprawdzono, a dwa przebiegi w środowisku źródłowym dały identyczne odciski. Po akceptacji decyzji dane pozwalają rozpocząć modelowanie eksploracyjne. Do strojenia modeli trzeba będzie jeszcze uzupełniać braki osobno w foldach wewnętrznych (sekcja 8, pakiet 2).
2. **Kwiecień–czerwiec, cały zespół:**
   - czyszczenie i filtracja danych, połączenie kohort z etykietą,
   - benchmark trzech metod imputacji, z którego wybraliśmy MICE z ExtraTrees,
   - pierwsza wersja podziału SGKF z imputacją w foldzie.

   Na tym oparł się raport przejściowy z 12.06.
3. **Wrzesień–październik, Łukasz:**
   - weryfikacja materiału z czerwca,
   - decyzje dotyczące danych: 63 pacjentów obecnych w obu kohortach, 752 wartości niemożliwe, powrót z 35 do 38 cech,
   - nowy, zamrożony podział SGKF (wersja 2),
   - pełna imputacja w foldach z oceną jakości,
   - diagnostyka „skrótów” w danych.
4. **Najważniejszy wniosek merytoryczny: kohorty różnią się wieloma rzeczami poza chorobą.** Różnią się okresem, tym, które badania zlecano, i najpewniej typem pacjenta. Wysokie AUC P-vs-U nie będzie więc dowodem wykrywania tętniaka. Nie będzie nim też sam wysoki wynik odzyskiwania ukrytych chorych. Ukryci zachowują bowiem cechy kohorty NEURO, a model, który widzi tylko to, jakie badania zlecono, odzyskuje około połowy z nich w górnych 5% rankingu. Ocena musi więc porównywać modele z modelami kontrolnymi na tych samych maskach i opierać się na analizach wrażliwości.
5. **Czego potrzebujemy ze spotkania:**
   - potwierdzenia decyzji,
   - informacji o pochodzeniu danych (KOR, NEURO) i o celu predykcji: wcześniejsze wykrywanie choroby czy rozpoznawanie profilu pacjentów już leczonych,
   - parametrów oceny (`q`, czyli odsetek pacjentów wybieranych z góry rankingu, oraz odsetek znanych chorych czasowo przenoszonych do puli U na potrzeby oceny),
   - podziału prac na etap modelowania.

---

## 2. Projekt i dane

**Zadanie.** Na podstawie wyników laboratoryjnych chcemy ocenić ryzyko tętniaka mózgu. Mamy klasę **P** (NEURO, pacjenci z rozpoznanym tętniakiem) i klasę **U** (KOR: w danych nie ma u nich rozpoznania tętniaka, ale nie wiemy, czy i u kogo wykonano obrazowanie). `label = 0` oznacza „nieoznaczony”, a nie „zdrowy”. Dlatego stosujemy uczenie Positive-Unlabeled, a nie zwykłą klasyfikację.

**Rekord** to tydzień uśrednionych wyników laboratoryjnych pacjenta. Cechy (38):

| Grupa | Cechy |
|---|---|
| morfologia | HGB, RBC, HCT, MCV, MCH, MCHC, RDW, PLT, MPV |
| rozmaz | WBC, NEUT, LYMPH, MONO, EO, BAZO, IG, NRBC, każda w wartości bezwzględnej i jako % |
| biochemia | KREA, eGFR-MDRD, eGFRCKD, Na, K, GLU, CRP |
| koagulologia | PT, INR, WPT (wskaźnik protrombinowy), APTT, WAPTT |
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
| 10–11.06 | integracja SGKF z MICE w foldzie (skrypt `aneurysm_sgkf_mice_pipeline.py`); raport metodologiczny dopisano 11.09 |
| **12.06** | **raport przejściowy** |

### 3.3 Co powstało

**Etap 1: przygotowanie danych.**
- **Czyszczenie strukturalne.** Listy `{'values': […]}`, czyli niedokończoną agregację tygodniową, zastąpiono średnią. Usunięto kolumny norm i jednostek oraz zmienne techniczne. Płeć ujednolicono do 0/1. Usunięto też jeden rekord uznany za duplikat. Weryfikacja w październiku pokazała, że to dwa różne badania jednego pacjenta NEURO (2017-01-01 i 2017-12-31). Klucz `custom_id` łączy rok kalendarzowy z numerem tygodnia ISO, więc oba badania dostały ten sam identyfikator. Strata to jeden prawdziwy rekord; w obecnym wejściu identyfikatory się nie powtarzają.
- **Filtracja:**
  - wiek 18–100 lat,
  - próg 60% braków, osobno dla kolumn i dla pacjentów (usunięto 1 793 pacjentów KOR),
  - u pacjentów z bardzo długą historią zostawiono najnowsze pomiary: limit wynosi 11 rekordów w KOR i 38 w NEURO, ustalony osobno dla każdej kohorty; pacjentów z jednym pomiarem nie usuwano.
- **Wartości odstające.** Usuwano całe rekordy poza skrajnymi kwantylami, parametr po parametrze. Progi dobierano osobno dla kohort, na całych kohortach przed podziałem. Usunięto 0,8% rekordów KOR i 3,2% rekordów NEURO.
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

- **Decyzja: MICE**, czyli `IterativeImputer` z ExtraTrees, parametry 7 / 78 / 10, wspólne dla obu kohort. Żadna metoda nie wygrała na obu zbiorach. MICE wygrywa na mniejszym i trudniejszym NEURO, a imputacja w foldzie wymaga jednej metody z jednym zestawem parametrów dla połączonych danych. W benchmarku MICE było szybsze na KOR (70,2 wobec 107,6 min), ale wolniejsze na NEURO (30,4 wobec 17,2 min). Na drugiej masce (seed 7) ranking się powtórzył. Dla NEURO strojenie uruchomiono wtedy ponownie, więc nie był to czysty test samej maski. Przewaga MICE nad MissForest na NEURO jest niewielka: 0,002–0,004 RMSE. Walidacja na 10 maskach dotyczyła tylko MICE. Pokazała stabilny wynik (odchylenie 0,0007 na KOR i 0,0025 na NEURO), ale przy parametrach dobranych osobno dla każdej kohorty. Wspólne parametry 7 / 78 / 10 sprawdzono na jednej masce.

**Etap 3, wersja 1: podział.**
- Liwia zaimplementowała `StratifiedGroupKFold` (5 foldów, grupowanie po `patient_id`). Wszystkie rekordy pacjenta trafiają po jednej stronie podziału.
- Łukasz zintegrował podział z imputacją: scaler i MICE uczą się tylko na części treningowej foldu.

Te dwa zabezpieczenia przed wyciekiem były poprawne i zostały zachowane w wersji 2.

### 3.4 Co zostało otwarte po raporcie przejściowym

Listę zebrała weryfikacja materiału czerwcowego we wrześniu (`0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md`, sekcja 6; dokument spisano 11.09, a punkty o okresach kohort i 63 pacjentach dopisano 18.09). Wcześniej, od kwietnia, znaliśmy błędy jednostek kreatyniny i wartości niemożliwe (notatki ze sprintu 1) oraz pacjentów obecnych w obu kohortach (`check_duplicates.py`, 31.05), ale raport przejściowy żadnego z tych problemów nie rozstrzygnął:
1. 63 pacjentów ma rekordy w obu kohortach i nie miało ustalonej etykiety.
2. Zostały wartości niemożliwe fizjologicznie, a jednostki kreatyniny są mieszane.
3. Cechy CRP, MONO i %MONO usunięto na podstawie korelacji z etykietą liczonej na całym zbiorze. Ten problem wskazała weryfikacja we wrześniu; w czerwcu raport imputacji zalecał jako wejście do modelowania plik z 35 cechami.
4. Podział nie był tasowany ani zapisywany.
5. Kohorty pochodzą z różnych okresów, a dane nie mają daty rozpoznania tętniaka.

---

## 4. Część II: prace od września (Łukasz), do zamrożonego SGKF

Od 11.09 do 05.10 powstały 24 commity: około 3 000 linii kodu i 2 800 linii dokumentacji. Zmiany znajdują się na gałęzi `pu-modeling-setup-lk`. **`master` kończy się na stanie z 11.09**; zmiany wymagają przeglądu zespołu przed scaleniem z główną gałęzią.

### 4.1 Wrzesień: weryfikacja i diagnostyka

- **Porządki.** Repozytorium przeorganizowano według etapów (`1-…`, `2-…`, `3-…`, `4-…`) i spisano materiał do raportu przejściowego.
- **Plan modelowania PU** (`4-pu-setup/PLAN_MODELOWANIA.md`). Plan opiera się na trzech elementach:
  - ocena przez ukrywanie części znanych chorych w puli U,
  - główna metryka `RecallHidden@q`, czyli odsetek ukrytych chorych odnalezionych w górnych `q`% rankingu pacjentów,
  - walidacja zagnieżdżona.
- **Diagnostyka danych.** Przejrzeliśmy cały dotychczasowy materiał pod kątem jednego pytania: *czy wynik modelu będzie mówił o tętniakach, czy o tym, jak powstały dane?* Najważniejsze ustalenia:

  | Ustalenie | Liczby |
  |---|---|
  | **kohorty niemal rozłączne w czasie** | KOR: 99,5% rekordów z lat 2020–2021; NEURO: 2000–2024, większość przed 2020. We wspólnym oknie dat (2019-12-29 – 2021-12-26) co najmniej jeden rekord ma 271 z 1 823 pozytywnych (680 rekordów), a co najmniej połowę rekordów 159 |
  | **wyniki NEURO najpewniej pochodzą z hospitalizacji** | u 1 215 chorych z więcej niż jednym rekordem mediana rozpiętości badań wynosi 35 dni |
  | **reguła agregacji rekordów do pacjenta tworzy sygnał** | maksimum koreluje z liczbą badań w 19 z 38 cech, mediana w 2 (wrzesień, na starszym wejściu: 17 z 35 i 2); rekomendacja: mediana |
  | **SGKF z wbudowanym tasowaniem działa inaczej w sklearn 1.6 i 1.8** | ta sama konfiguracja daje inne foldy, bo w 1.6 tasowanie ma błąd; wersja z czerwca nie używała tasowania, więc jej to nie dotyczy (sekcja 4.3) |

- **Prototyp pipeline'u PU.** Stan sprzed spotkania 21.09, zarchiwizowany 25.09: ukrywanie pozytywnych, metryki pacjentowe, funkcja celu, 17 testów. Po spotkaniu modelowanie odłożono, a priorytetem stało się domknięcie SGKF. Branch usunięto, ale kod da się odzyskać z commita `2cb6d1a` (sekcja 9).

### 4.2 04.10: decyzje dotyczące danych (część A etapu 3)

Decyzje wdraża osobny skrypt (`3-sgkf-split/przygotowanie/przygotuj_dane.py`). Zapisuje on spis każdej zmienionej komórki i każdego usuniętego rekordu, a jego działanie pilnują testy.

| Problem | Decyzja | Skutek |
|---|---|---|
| 63 pacjentów w obu kohortach | zostają **pozytywni**, ich rekordy KOR usunięte | −145 rekordów KOR; liczba pacjentów bez zmian |
| KREA > 50 mg/dl | → brak (błąd jednostki, najpewniej µmol/l) | 210 wartości |
| KREA > 10 mg/dl przy eGFR ≥ 30 | → brak (sprzeczność z eGFR) | 516 wartości |
| K > 15 mmol/l, Na < 80 mmol/l | → brak (niezgodne z życiem) | 25 + 1 wartość |
| skrajne, ale możliwe (K 9–15, Na 80–100, WBC > 200, KREA > 10 przy eGFR < 30) | **zostają** | 325 wartości |
| zestaw cech | **38**: selekcja z etapu 1 cofnięta | CRP, MONO, %MONO wracają |
| rozjazd czasowy, asymetria braków | **bez korekty**, opisane jako ograniczenia | — |

**Dlaczego reguła dla kreatyniny odwołuje się do eGFR.** Laboratorium wylicza eGFR z kreatyniny, więc oba pomiary muszą być spójne. Do KREA ≈ 10 mg/dl mediana eGFR spada zgodnie z fizjologią i przy KREA 5–10 wynosi 8. Powyżej 10 mediana eGFR znowu rośnie: 30,8 przy KREA 10–20 i 60 (górna granica raportowania eGFR-MDRD) przy KREA 20–50. Wysokiej kreatyninie z tak wysokim eGFR odpowiada raczej błąd jednostki albo wpisu niż rzeczywisty wynik pomiaru.

**Dlaczego 38 cech, a nie 35:**
1. W PU etykieta oznacza kohortę, nie chorobę. Kryterium „słaba korelacja z etykietą” usuwa więc cechy, które *nie* odróżniają kohort, czyli działa w złą stronę.
2. Na granicy odcięcia był remis czterech cech (%MONO, HCT, MPV, wiek), który rozstrzygnęła kolejność alfabetyczna. Tą samą metodą mógł wypaść wiek pacjenta.
3. Usunięte cechy miały słabą korelację z etykietą (|ρ| ≤ 0,02), a porównanie 35 i 38 cech wykazało niewielkie różnice imputacji. Nie oceniliśmy jednak wpływu tej selekcji na przyszłe modele PU; sama niska korelacja go nie wyklucza.

### 4.3 Podział SGKF, wersja 2 (część B)

**Metoda.** Używamy `StratifiedGroupKFold` ze sklearn, tak jak w czerwcu i zgodnie z zaleceniem prowadzącego: 5 foldów, `groups` = pacjent, `y` = warstwa pacjenta. Wszystkie rekordy pacjenta trafiają do jednego foldu, a skład foldów jest wyrównany.

| Problem w wersji 1 | Zmiana w wersji 2 |
|---|---|
| stratyfikacja tylko po etykiecie: udział chorych z okresu KOR różnił się między foldami o 4,06 pp | `y` = 12 warstw: etykieta × okno czasowe × liczba rekordów (1 / 2 / 3–4 / 5+) |
| brak tasowania: kolejność wyznaczały numery pacjentów, a seed nie miał wpływu | losowa kolejność pacjentów sterowana ziarnem 42 |
| przydział niezapisany: każdy etap liczyłby go od nowa | przydział zapisany w `pacjent_fold.csv` z odciskiem pliku wejściowego; zmiana danych unieważnia podział |

Okno czasowe niczego nie usuwa. Wszystkie rekordy zostają w danych, a znacznik „co najmniej połowa rekordów w oknie dat KOR” (159 chorych) służy tylko do równego rozłożenia pacjentów po foldach.

**Wynik (seed 42).** Każdy fold ma 8 184–8 185 pacjentów, w tym 364–366 chorych (4,45–4,47%), oraz 15 609–15 612 rekordów, z których 8,51–8,54% to rekordy NEURO.

Porównanie strategii, rozstęp między foldami, przy losowaniu najgorszy wynik z 5 ziaren:

| Strategia | Udział chorych | Chorzy z okna KOR | Chorzy z 5+ rekordami | Udział rekordów NEURO |
|---|---:|---:|---:|---:|
| wersja 1: SGKF, `y` = etykieta, bez tasowania | 0,02 pp | 4,06 pp | 0,27 pp | 0,01 pp |
| wariant testowany: `StratifiedKFold` na tabeli pacjentów, te same warstwy | 0,01 pp | 0,55 pp | 0,22 pp | 1,15 pp |
| **wersja 2: SGKF, `y` = 12 warstw** | **0,04 pp** | **0,75 pp** | **0,39 pp** | **0,05 pp** |

**Wariant testowany.** Sprawdziliśmy też `StratifiedKFold` na tabeli pacjentów (obowiązywał od 04.10 do 31.10). Na pacjentach oba warianty są praktycznie równe (rozkład chorych między foldami różni się o najwyżej jednego chorego), a na rekordach SGKF wypada wyraźnie lepiej (w wariancie testowanym: 15 540–15 722 rekordy i 8,07–9,18% rekordów NEURO w foldzie). Dlatego zostajemy przy SGKF. Wyniki dalszych kroków na obu wariantach są praktycznie takie same: błąd MICE przy brakach całymi panelami wobec średniej to 0,98 i 0,89 na obu, AUC kohort na pacjentach 0,928 i 0,927, a `RecallHidden@5%` samego wzorca braków 51,6% i 51,7%. Największa różnica dotyczy rozdzielności we wspólnym oknie dat, opartej na małej próbie: AUC 0,801 i 0,782.

**Sprostowanie.** Wcześniej pisaliśmy, że podział z czerwca zależał od kolejności wierszy i od wersji sklearn. Kontrola (`3-sgkf-split/sgkf/kontrola_sgkf_czerwiec.py`) pokazała, że nie: po potasowaniu wierszy wszyscy pacjenci zostają w tych samych foldach, a bez tasowania oba komputery dają ten sam wynik. Różnice wersji dotyczą tylko wbudowanej opcji `shuffle=True`, która w sklearn 1.6.1 ma błąd psujący wyrównanie (rozstęp 0,80 pp wobec 0,11 pp w sklearn 1.8). Dlatego kolejność pacjentów mieszamy sami i tej opcji nie używamy.

### 4.4 Imputacja w foldach i jej ocena

- **Sama imputacja** zajmuje około 15 minut na 5 foldów; cały etap 3 z oceną jakości trwa około 35 minut. W żadnym foldzie ten sam pacjent nie trafia jednocześnie do części treningowej i testowej. Po imputacji nie ma braków ani wartości ujemnych. Dwa niezależne przebiegi w środowisku źródłowym dały identyczne wyniki.
- **Zakres uzupełnień.** MICE uzupełnia około 12% komórek KOR i 24–27% komórek NEURO.
- **Ocena jakości.** Użyliśmy protokołu z etapu 2, ale poza danymi uczącymi (out-of-fold): imputer uczył się na części treningowej foldu, a ocenialiśmy go na kompletnych wierszach części testowej. Kolumna „wobec średniej” to błąd MICE podzielony przez błąd uzupełniania średnią z treningu; 1,0 oznacza brak przewagi nad średnią.

  | Maska | KOR: RMSE / wobec średniej | NEURO: RMSE / wobec średniej |
  |---|---|---|
  | losowe 10% komórek, jak w etapie 2 | 0,083 / **0,62** | 0,083 / **0,66** |
  | to samo, bez wieku i płci² | 0,043 / **0,41** | 0,040 / **0,45** |
  | realistyczne wzorce braków (całe panele) | 0,107 / **0,98** | 0,083 / **0,89** |

  Przy losowych brakach MICE działa tak dobrze jak w benchmarku, jednakowo dla obu kohort. W prawdziwych danych braki idą jednak całymi panelami: koagulogramu i glukozy brakuje w około połowie rekordów. Wtedy MICE jest tylko trochę lepsze od średniej z treningu, a jego uzupełnienia skupiają się wokół niej. Odchylają się od średniej trzy razy mniej niż prawdziwe wartości w KOR i dwa razy mniej w NEURO. **Benchmark z etapu 2 przeszacowywał więc użyteczność imputacji.**

  ² Losowa maska, tak jak w benchmarku z etapu 2, ukrywa też wiek i płeć, których w danych nigdy nie brakuje. Przypada na nie około 3/4 sumy kwadratów błędu, głównie na płeć. Bez nich przewaga MICE przy losowych brakach jest jeszcze większa. Wniosek o przeszacowaniu nie wynika więc z demografii.
- **38 czy 35 cech.** Wybór nie zmienia istotnie imputacji. W teście kontrolowanym RMSE wynosi 0,0834 dla 38 cech i 0,0846 dla 35. W prawdziwych foldach uzupełnienia różnią się średnio o 0,11 odchylenia standardowego.

### 4.5 Diagnostyka „skrótów”

Na zamrożonych foldach sprawdziliśmy, jak łatwo odróżnić grupy, które nie powinny być odróżnialne, gdyby różniła je wyłącznie choroba. To diagnostyka danych, a nie model ryzyka.

| Test (HistGradientBoosting, część testowa foldów) | AUC |
|---|---:|
| NEURO vs KOR, 38 cech po imputacji (pacjenci) | **0,93** |
| NEURO vs KOR, **tylko wzorzec braków**, czyli które badania zlecono (pacjenci) | **0,82** |
| NEURO vs KOR, tylko rekordy ze wspólnego okna dat (pacjenci)¹ | 0,78 |
| NEURO z okna dat KOR vs NEURO z innych lat (rekordy) | **0,83** |

¹ Okno wybrano dopiero po imputacji, a imputer dopasowano na całym treningu foldu, ze wszystkich okresów. Ten sam test na danych bez imputacji (braki jako NaN) daje 0,825, czyli nie mniej, więc rozdzielność w oknie nie jest efektem imputacji. Ograniczeniem pozostaje mała próba: rekord w oknie ma 271 chorych.

**Kontrola z 06.10, uzupełniona 16.10 i przeliczona 31.10** (`3-sgkf-split/sgkf/kontrola_epoki.py`). Ostatni test liczono na danych po imputacji, więc nie wiadomo było, gdzie siedzi sygnał epoki. Na danych przed imputacją, na tych samych rekordach z kompletnym panelem, wychodzi tak:
- same zmierzone wartości panelu dają AUC 0,57–0,65 (morfologia: 0,57; morfologia z rozmazem: 0,65),
- wzorzec braków pozostałych badań daje 0,77–0,78,
- wszystkie 38 cech razem dają 0,86–0,87,
- wiek i płeć prawie nie zdradzają okresu (0,52).

Najsilniejszym pojedynczym śladem epoki jest więc to, *które badania zlecano* w danym okresie. W wartościach podstawowych paneli sygnał jest umiarkowany. Nie jest to jednak jedyne źródło, bo 38 cech razem daje więcej niż sam wzorzec braków: część sygnału niosą też wartości pozostałych badań.

**Profil KOR nie wygląda na populację ogólną.** Wśród zmierzonych wyników KOR CRP > 5 mg/l ma 61% (NEURO 62%), WBC > 10 G/l 37%, a glukoza > 99 mg/dl 64%. Odsetki dotyczą badań wykonanych: CRP zmierzono w 81% rekordów KOR, glukozę w 50%. Względem wszystkich rekordów KOR CRP > 5 mg/l ma 50%. Profil pasuje do pacjentów szpitalnych z lat 2020–2021, czyli z okresu pandemii. To hipoteza do potwierdzenia (pytanie 1 w sekcji 7).

**Ślad imputacji.** Zmierzone NRBC to w 78% dokładne zera, a uzupełnione nigdy nie są zerem. NRBC brakuje w 48% rekordów NEURO i w 3% KOR, więc 63% rekordów z uzupełnionym NRBC to NEURO, wobec 8,5% w całych danych. Model widzący tylko NRBC po imputacji odróżnia kohorty z AUC 0,73 (rekordy), tak samo jak sam wskaźnik braku NRBC. Uzupełnienie nie ukrywa więc informacji, że badania nie zrobiono. Nie jest to jednak jednoznaczny znacznik NEURO: wśród wszystkich rekordów z małym niezerowym NRBC (0–0,05) NEURO stanowi 19%, bo takie wartości dają też pomiary.

**Kontrola z 17.10, przeliczona 31.10: ile odzyskiwania ukrytych chorych dają same skróty** (`3-sgkf-split/sgkf/kontrola_skrotow.py`). Ukrywamy losowo wybranych chorych z NEURO, więc ukryci zachowują cechy swojej kohorty: okres badań, wzorzec braków i ślad imputacji. Sprawdziliśmy, jaki `RecallHidden@q` dają modele, które widzą tylko takie cechy. Protokół jest orientacyjny, bo nie został jeszcze zamrożony: 40% ukrytych chorych, 5 ziaren, zamrożone foldy, średnio 146 ukrytych na fold.

| Model | Ukryci chorzy w górnych 5% rankingu (`RecallHidden@5%`) |
|---|---:|
| losowy ranking | 4% (oczekiwane 5%) |
| tylko wiek i płeć | 12% |
| **tylko wzorzec braków** (które badania zlecono) | **52%** |
| 38 cech po imputacji MICE | 66% |
| 38 cech bez imputacji (braki jako NaN) | 69% |

Sam wzorzec braków osiąga około 3/4 wyniku pełnego modelu. W wariancie z jednym profilem na pacjenta zamiast rekordów wychodzi podobnie (56% i 68,5%). Wysoki `RecallHidden@q` mierzy więc odzyskiwanie znanych chorych NEURO w tym zbiorze, także dzięki skrótom. Nie jest to dowód wykrywania tętniaka.

**Wnioski dla modelowania:**
- AUC P-vs-U raportujemy tylko jako rozdzielność kohort.
- Główną miarą zostaje odzyskiwanie ukrytych pozytywnych (`RecallHidden@q`), ale zawsze obok modeli kontrolnych liczonych na tych samych maskach: losowego, samej demografii i samego wzorca braków. Liczy się przewaga nad nimi, a nie sama wartość. Przewaga nie dowodzi jeszcze wykrywania choroby, ale jej brak pokazuje, że model opiera się głównie na skrótach.
- Wynik opisujemy jako „odzyskiwanie znanych chorych w tym zbiorze”. Model daje ranking pacjentów, a nie prawdopodobieństwo tętniaka. Skuteczność wśród niewykrytych chorych w KOR wymagałaby niezależnej walidacji.
- Niezbędne są analizy wrażliwości:
  - model na wartościach zmierzonych z paneli o niskim odsetku braków,
  - model we wspólnym oknie dat, z preprocessingiem dopasowanym tylko w oknie,
  - model bez NRBC,
  - porównanie MICE z prostym uzupełnieniem średnią albo medianą.

### 4.6 Porządki i odtwarzalność

- **Jedno polecenie.** Cały etap 3 uruchamia `python 3-sgkf-split/uruchom_sgkf.py`: około 35 minut, logi z każdego kroku. Opcja `--bez-mice` pozwala sprawdzić etap w około minutę na Macu z procesorem M.
- **Testy.** 18 testów automatycznych (6 w części A, 12 w części B). Pilnują m.in., że plik wejściowy w repozytorium jest dokładnie wynikiem kodu.
- **Jeden branch roboczy** zamiast pięciu, dokumenty nazwane datą.

---

## 5. Czerwiec a październik: co się zmieniło

| Element | Raport przejściowy (06.2026) | Teraz (10.2026) |
|---|---|---|
| wejście do modelowania | `aneurysm_concatted_cleaned.csv`: 78 197 rekordów, 35 cech | `aneurysm_sgkf_input.csv`: 78 052 rekordy, **38 cech** |
| 63 pacjentów w obu kohortach | nierozstrzygnięte | pozytywni, rekordy KOR usunięte |
| wartości niemożliwe | częściowo (kwantyle statystyczne) | 752 wartości → brak według reguł fizjologicznych |
| podział | SGKF, `y` = etykieta, bez tasowania, niezapisany | SGKF, `y` = 12 warstw, losowa kolejność pacjentów (seed 42), **zapisany z odciskiem danych** |
| imputacja w foldach | kod gotowy | **pełny przebieg wykonany**, powtarzalny |
| ocena imputacji | benchmark na losowej masce | + ocena out-of-fold + realistyczne wzorce braków |
| ryzyka w danych | opisane jako ograniczenia | **zmierzone** (AUC skrótów, profil KOR, ślad NRBC) |
| odtwarzalność | notebooki, ścieżki z Colab | jedno polecenie, 18 testów, logi |
| **bez zmian** | metoda imputacji i jej parametry, imputacja w foldzie, grupowanie po pacjencie, czyszczenie z etapu 1 | |

---

## 6. Decyzje do potwierdzenia

Decyzje przygotował i wdrożył Łukasz 04.10. **Są propozycją do akceptacji przez zespół i prowadzącego.** Zmiana każdej z nich to edycja jednej reguły w `przygotuj_dane.py` i ponowne uruchomienie etapu 3 (około 35 minut).

| # | Kwestia | Decyzja | Uzasadnienie |
|---|---|---|---|
| 1 | 63 pacjentów w obu kohortach | pozytywni, 145 rekordów KOR usuniętych | pacjent nie może być jednocześnie P i U; bez daty rozpoznania nie wiadomo, czy rekordy KOR są sprzed choroby |
| 2 | wartości niemożliwe | KREA > 50 albo > 10 przy eGFR ≥ 30, K > 15, Na < 80 → brak, uzupełniane w foldzie | usuwamy tylko to, co niemożliwe albo sprzeczne z innym pomiarem |
| 3 | zestaw cech | 38, selekcja z etapu 1 cofnięta | sekcja 4.2; ewentualny wybór cech tylko wewnątrz walidacji krzyżowej |
| 4 | rozjazd czasowy kohort | bez korekty, ograniczenie + analiza wrażliwości we wspólnym oknie dat | obcięcie do wspólnego okna zostawia 271 z 1 823 pozytywnych |
| 5 | asymetria braków | bez korekty, raportowana w każdym foldzie; analiza wrażliwości na wartościach zmierzonych | sam wzorzec braków odróżnia kohorty (AUC 0,82) |
| 6 | podział | SGKF: 5 foldów, `y` = 12 warstw, seed 42, zamrożony | zalecenie prowadzącego; rozstęp udziału chorych między foldami 0,02 pp; sprawdzony wariant z `StratifiedKFold` na tabeli pacjentów nie daje przewagi (sekcja 4.3) |
| 7 | czyszczenie z etapu 1: progi kwantylowe i limit historii (11 rekordów w KOR, 38 w NEURO) ustalone osobno dla kohort, na całych danych przed podziałem | na razie bez zmian, opisane jako ograniczenie; wyniki traktujemy jako eksploracyjne. **Do decyzji:** czy ujednolicić reguły przy ponownym zamrożeniu danych | wspólny limit 11 rekordów usunąłby 943 rekordy 94 chorych (3,13 zamiast 3,65 rekordu na chorego). Ujednolicenie wymaga odtworzenia etapu 1 z notebooków i ponownego przeliczenia etapu 3; przy okazji wróciłby rekord usunięty przez kolizję klucza (sekcja 3.3) |

---

## 7. Pytania do prowadzącego

W kolejności wpływu na projekt:

1. **Skąd pochodzi kohorta KOR?** Jaki oddział lub jakie kryteria włączenia? Czy to pacjenci hospitalizowani i czy ma to związek z okresem pandemii? Od odpowiedzi zależy interpretacja klasy U i sens zastosowania przesiewowego.
2. **Skąd pochodzą wyniki NEURO i jaki jest cel predykcji?** Czy badania wykonano przed rozpoznaniem tętniaka, w trakcie hospitalizacji, w której go rozpoznano, czy po leczeniu? Czy da się uzyskać datę rozpoznania? Od odpowiedzi zależy, czy badamy możliwość wcześniejszego wykrywania choroby, czy rozpoznawanie profilu pacjentów już hospitalizowanych lub leczonych. Bez tej informacji bezpieczny zakres projektu to eksploracyjny ranking PU w dostępnych kohortach.
3. **Rozjazd czasowy:** czy akceptuje Pan „bez korekty” z analizą wrażliwości we wspólnym oknie dat?
4. **NRBC i %NRBC:** wykluczyć z modelowania? Uzupełnione wartości zdradzają brak badania (sekcja 4.5), a u dorosłych cechy mają niską wartość kliniczną.
5. **Parametry oceny:**
   - `q`, czyli jaki odsetek pacjentów wybieramy z góry rankingu; w przyszłym zastosowaniu odpowiadałby możliwości skierowania ich na obrazowanie,
   - odsetek znanych chorych ukrywanych w puli U w scenariuszu głównym (propozycja: 40%),
   - metody PU (propozycja: PU Bagging i korekta Elkana–Noto obok modeli bazowych; sama korekta Elkana–Noto dzieli wynik przez stałą, więc nie zmienia rankingu ani `RecallHidden@q` względem modelu bazowego, a inny wynik daje dopiero wariant z ważonym ponownym treningiem),
   - reguła agregacji rekordów do pacjenta (propozycja: mediana),
   - modele kontrolne na tych samych maskach (losowy, sama demografia, sam wzorzec braków) jako stały punkt odniesienia.
6. **Walidacja:** czy jest szansa na sprawdzenie modelu na niezależnej kohorcie, zebranej w podobny sposób, albo na kliniczną weryfikację części wskazań? Bez tego wynik możemy opisać tylko jako odzyskiwanie znanych chorych w tym zbiorze.
7. **Termin i oczekiwany zakres raportu końcowego.**

Pełna lista kwestii technicznych: `4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 7.

---

## 8. Co dalej: propozycja etapu 4 i podziału prac

**Zanim powstanie pierwszy model, trzeba zamrozić protokół:**
- maski ukrywania (całych pacjentów, identyczne dla wszystkich modeli),
- metryki liczone na pacjentach z ustalonym `q`,
- modele kontrolne liczone na tych samych maskach (losowy ranking pacjentów, sama demografia, sam wzorzec braków),
- scenariusz główny i analizy wrażliwości.

Strojenia `q` pod wynik nie dopuszczamy.

| # | Pakiet | Zakres | Zależy od | Kto |
|---|---|---|---|---|
| 0 | przegląd i scalenie | zespół przegląda zmiany od 11.09; scalenie `pu-modeling-setup-lk` → `master`; wspólny `requirements.txt` z przypiętymi wersjami | — | wszyscy |
| 1 | protokół oceny | ukrywanie pozytywnych, `RecallHidden@q`, Lift@q, RecallKnown, funkcja celu, modele kontrolne; jednostka treningu i kolejność agregacji względem imputacji; punkt wyjścia: prototyp `2cb6d1a` i `kontrola_skrotow.py` | pytania 2 i 5 z sekcji 7 | do ustalenia |
| 2 | pipeline modelowania | walidacja zagnieżdżona na zamrożonych foldach, imputacja od nowa w foldach wewnętrznych, agregacja do pacjenta | 1 | do ustalenia |
| 3 | modele bazowe | regresja logistyczna (z regularyzacją, bo procenty rozmazu sumują się do 100%), Random Forest, jeden boosting | 2 | do ustalenia |
| 4 | metody PU | PU Bagging, Elkan–Noto w wariancie z ważonym treningiem (class prior szacowany tylko na treningu) | 2 | do ustalenia |
| 5 | analizy wrażliwości | wartości zmierzone z paneli o niskim odsetku braków; wspólne okno dat z preprocessingiem dopasowanym w oknie; bez NRBC; bez 63 pacjentów mieszanych; MICE wobec prostego uzupełnienia średnią lub medianą | 3, 4 | do ustalenia |
| 6 | raport końcowy | metodologia etapów 1–3 (materiał gotowy), ograniczenia, wyniki | wszystko | do ustalenia |

Pakiety 3 i 4 można prowadzić równolegle, gdy gotowe są pakiety 1 i 2.

**Koszt obliczeń.** MICE zajmuje około 3 minut na fold (Mac M-series). Walidacja zagnieżdżona 5 × 3 oznacza około godziny na jedną konfigurację preprocessingu. Wynik preprocessingu nie zależy od hiperparametrów klasyfikatora, więc da się go zapisać raz na split i używać ponownie.

**Czego się spodziewać.** Przy 40% ukrywania na fold przypada około 146 ukrytych chorych, więc rozdzielczość `RecallHidden@q` wynosi około 0,7 pp. Poprzeczkę ustawiają modele kontrolne: sam wzorzec braków odzyskuje około 52% ukrytych w górnych 5% (sekcja 4.5). We wspólnym oknie dat zostałoby około 22 ukrytych na fold. Dlatego to okno nadaje się na analizę wrażliwości, ale nie na scenariusz główny.

---

## 9. Weryfikacja repozytorium (06.10, 16–17.10 i 31.10.2026)

### 9.1 Zmiana podziału na SGKF i trzecia weryfikacja, 31.10.2026

Prowadzący zalecił `StratifiedGroupKFold`, a przegląd pokazał, że część powodów odejścia od tej metody była błędna (sekcja 4.3). Podział przebudowaliśmy więc na SGKF z `y` = 12 warstw, a cały etap 3 policzyliśmy od nowa w środowisku źródłowym (Python 3.9.6, sklearn 1.6.1, pandas 2.3.3). Dane wejściowe się nie zmieniły (odcisk `5b822c09ebc03011`). Około 80% pacjentów ma teraz inny numer foldu, a wszystkie liczby zależne od foldów w tym dokumencie są przeliczone.

**Co sprawdziliśmy niezależnym skryptem, bez kodu projektu:**
- przydział zbudowany od zera bezpośrednio klasą `StratifiedGroupKFold` jest identyczny z `pacjent_fold.csv`, żaden pacjent nie ma rekordów w dwóch foldach, a etykiety i warstwy są zgodne z danymi;
- wszystkie 10 plików po imputacji ma odciski zgodne z zapisem przebiegu (5/5). Pacjenci treningu i testu są rozłączni, każdy pacjent jest w teście dokładnie raz, nie ma braków ani wartości ujemnych, a wartości zmierzone są zachowane (maks. odchylenie 2,3 · 10⁻¹³);
- dwa pełne przebiegi imputacji na nowych foldach dały identyczne odciski;
- testy przechodzą 18/18, a `podzial.py --sprawdz` potwierdza zgodność podziału z danymi.

**Co się zmieniło w wynikach.** Wnioski są te same, a liczby zmieniły się nieznacznie, na przykład RMSE NEURO przy losowej masce z 0,080 na 0,083. Największa różnica to rozdzielność kohort we wspólnym oknie dat: spadek z 0,80 do 0,78 (mała próba).

**Czego nie sprawdziliśmy.** Podziału nie uruchamialiśmy na drugim komputerze (sklearn 1.8). Kod nie korzysta z wbudowanego tasowania, więc wynik powinien być ten sam, ale to trzeba potwierdzić (sekcja 9.4).

### 9.2 Druga weryfikacja, 16–17.10.2026

Liczby i opisy w tej podsekcji dotyczą podziału obowiązującego do 31.10 (`StratifiedKFold` na tabeli pacjentów, ostatni commit `f65a112`).

Liczby i wnioski sprawdziliśmy ponownie w środowisku, w którym powstały wyniki (Python 3.9.6, sklearn 1.6.1, pandas 2.3.3). Najważniejsze liczby policzyliśmy niezależnym skryptem, bez kodu projektu.

**Co się zgadza:**
- **część A:** plik wejściowy odtwarza się z `aneurysm_concatted.csv` wartość w wartość. Zgadzają się liczby: 63 pacjentów mieszanych, 145 rekordów KOR, 210 + 516 + 25 + 1 = 752 wartości;
- **podział:** przydział zbudowany od zera (StratifiedKFold na 12 warstwach) jest identyczny z `pacjent_fold.csv`. Każdy pacjent jest w jednym foldzie, a etykiety są zgodne z danymi;
- **imputacja w foldach:** wszystkie 10 zapisanych plików ma odciski zgodne z zapisem przebiegu (5/5). Pacjenci train i test są rozłączni, nie ma braków ani wartości ujemnych, a wartości zmierzone zostały zachowane (maks. odchylenie 2,3 · 10⁻¹³);
- **ocena imputacji:** ponowny pełny przebieg daje te same wyniki we wszystkich foldach (różnice rzędu 10⁻¹⁷);
- **pozostałe wyniki:** AUC diagnostyki, porównanie 38/35, liczby z etapu 2 (tabela RMSE, czasy) i oś czasu prac zgadzają się z zapisanymi wynikami i historią repozytorium;
- **testy:** 16/16 przechodzą, a `podzial.py --sprawdz` potwierdza zgodność podziału z danymi.

**Co poprawiliśmy.** Żadna poprawka nie zmienia danych, podziału ani imputacji. Zmieniają się opisy i siła kilku wniosków:
1. „Duplikat” z etapu 1 to kolizja klucza: dwa różne badania dostały ten sam `custom_id` (sekcja 3.3).
2. Opis stabilności benchmarku: seed 7 dla NEURO był liczony z ponownym strojeniem, a walidacja na 10 maskach dotyczyła tylko MICE (sekcja 3.3; korekta w `RAPORT_IMPUTACJA.md`, sekcja 4.2).
3. Losowa maska ukrywa wiek i płeć, które dają około 3/4 jej błędu. Wynik bez nich wzmacnia wniosek o przeszacowaniu użyteczności imputacji (sekcja 4.4).
4. „Uzupełnienie bliskie średniej” zastąpiliśmy liczbami. MICE jest niewiele lepsze od średniej, a jego uzupełnienia skupiają się wokół niej, ale jej nie powtarzają (sekcja 4.4).
5. Epoka: porównanie na tych samych rekordach. Wzorzec braków jest najsilniejszym, ale nie jedynym śladem (sekcja 4.5).
6. Profil KOR: odsetki liczone są wśród zmierzonych wyników, teraz z podanym mianownikiem (sekcja 4.5).
7. NRBC: zamiast „oznacza NEURO” podajemy liczby: 63% uzupełnień dotyczy NEURO, a samo NRBC daje AUC 0,73 (sekcja 4.5).
8. Drobne poprawki:
   - 325, a nie 327 wartości skrajnych: liczone po przygotowaniu danych, bez rekordów KOR pacjentów mieszanych,
   - 24, a nie 25 commitów,
   - analiza agregacji powtórzona na obecnym wejściu: 19 z 38 cech, wniosek bez zmian,
   - opis KOR: „nie wiemy, czy wykonano obrazowanie” zamiast „nikt nie robił obrazowania”,
   - czyszczenie z etapu 1 było ustawiane osobno dla kohort (sekcja 3.3).

**Dodane 17.10, po przeglądzie listy problemów przed modelowaniem:**
9. Odzyskiwanie ukrytych chorych też jest podatne na skróty: sam wzorzec braków daje `RecallHidden@5%` około 52% (sekcja 4.5). Zmieniło to wnioski dla modelowania (modele kontrolne na tych samych maskach), opis wyniku w sekcji 1 i pakiety 1 i 5 w sekcji 8.
10. Rozdzielność kohort we wspólnym oknie dat bez imputacji wynosi 0,83, więc wcześniejsze zastrzeżenie o imputacji dopasowanej na innych okresach nie zmienia wniosku (sekcja 4.5).
11. Czyszczenie z etapu 1 trafiło do decyzji do potwierdzenia (sekcja 6, pkt 7), a pytanie o pochodzenie wyników NEURO doprecyzowaliśmy o cel predykcji. Doszło pytanie o niezależną walidację (sekcja 7).
12. W planie modelowania poprawiliśmy opis reguły „ostatni”: `groupby.last()` bierze ostatnią dostępną wartość każdej cechy, a nie jedno ostatnie badanie. Dopisaliśmy też, że losowy punkt odniesienia trzeba liczyć na pacjentach. Losowy wynik rekordów zagregowany medianą daje bowiem 3% zamiast 5%, bo agregacja predykcji wchodzi w interakcję z liczbą rekordów pacjenta.

Nowe liczby liczą skrypty z repozytorium:
- `ocena_imputacji.py`: wynik bez wieku i płci, rozrzut uzupełnień,
- `analiza_wynikow.py`: sekcja E (NRBC) i mianowniki profilu,
- `kontrola_epoki.py`: porównanie na tych samych rekordach,
- `kontrola_skrotow.py` (17.10): `RecallHidden@q` modeli kontrolnych i rozdzielność w oknie bez imputacji,
- `analiza_decyzji.py`: wartości skrajne po przygotowaniu i limit historii z etapu 1 (sekcja L),
- `4-pu-setup/etap0_agregacja.py`: obecne wejście.

### 9.3 Pierwsza weryfikacja, 06.10.2026

Sprawdzono ją w innym środowisku niż to, w którym powstały wyniki (Python 3.13, sklearn 1.9.1, pandas 3.0). Liczby z danych, testy 16/16, zgodność zapisanego podziału i odtworzenie przydziału się zgadzały. W środowisku źródłowym części A i B bez imputacji (`--bez-mice`) odtwarzają pliki wynikowe co do bajtu, poza datą utworzenia w `podzial_meta.json`. Poprawiono wtedy liczbę testów (16, nie 17) i dodano adnotację w `RAPORT_IMPUTACJA.md` (sekcja 4.7), że globalnie imputowany plik 35 cech nie jest wejściem do modeli. Przy decyzji o parametrach dopisano też wynik MICE z parametrami finalnymi (0,0861 / 0,0984).

### 9.4 Do zrobienia

1. **Sprawdzić podział na drugim komputerze (sklearn 1.8).** Po `python 3-sgkf-split/sgkf/podzial.py` plik `pacjent_fold.csv` powinien zostać bez zmian w `git status`.
2. **`master` nie zawiera prac z września i października.** Zespół powinien je przejrzeć przed scaleniem (pakiet 0).
3. **Kod SGKF przed modelowaniem** (sprawdzenie kodu przed pakietem 2):
   - odczyt przydziału kontroluje dziś tylko odcisk danych, a nie sam plik `pacjent_fold.csv` (unikalność pacjentów, dozwolone numery foldów, etykiety). Obecny plik jest poprawny;
   - zaimputowane foldy są czytane po nazwie pliku, bez manifestu z konfiguracją i odciskami;
   - imputacja w foldach wewnętrznych nie jest jeszcze zaimplementowana. Do strojenia nie wolno dzielić zaimputowanego treningu foldu zewnętrznego.
4. **Odcisk danych zależy od wersji pandas.** Ponowne przygotowanie danych w pandas 3.x zmienia ostatnią cyfrę kilku wartości, więc zmienia się odcisk pliku. Zapisany podział odmawia wtedy działania, choć przydział pacjentów jest identyczny. Rozwiązanie: przypięte wersje bibliotek (wspólny `requirements.txt`, łącznie z PyArrow) albo odcisk liczony z zaokrąglonych wartości.
5. **Odtwarzalność od danych surowych.** `uruchom_sgkf.py` odtwarza etap 3 od `aneurysm_concatted.csv`, a nie cały proces od danych surowych. Kroki z etapu 1 istnieją w notebookach i skryptach o niespójnych ścieżkach.
6. **Kod prototypu PU** (commit `2cb6d1a`, 25.09) jest dostępny tylko po SHA, bo branch usunięto. Warto oznaczyć go tagiem, zanim GitHub go usunie.
7. **Drobne porządki:**
   - plik `scripts-lf/aneurysm_data_analysis.ipynb` to zapisana strona HTML,
   - notebooki mają ścieżki z Colab,
   - kopie i wersje pośrednie plików CSV leżą w kilku miejscach (`data/interim/` to 138 MB) i nie zawsze są identyczne. Na przykład `data/interim/cleaned/neuro_shortend.csv` ma 6 616 rekordów, a obowiązujący `data/imputation-inputs/neuro_shortend.csv` ma 6 651,
   - `custom_id` łączy rok kalendarzowy z tygodniem ISO; przy kolejnym zamrożeniu danych klucz powinien łączyć rok i tydzień ISO.

---

## 10. Gdzie co jest

| Co | Plik |
|---|---|
| **wejście do modelowania** (38 cech, przed imputacją) | `data/processed/aneurysm_sgkf_input.csv` |
| przydział pacjent → fold | `3-sgkf-split/sgkf/results/pacjent_fold.csv` (czytać przez `podzial.wczytaj_podzial()`) |
| cały etap 3 jednym poleceniem | `python 3-sgkf-split/uruchom_sgkf.py` |
| pełny opis etapów 1–3, kto co zrobił, decyzje | `1026_REALIZACJA_DO_SGKF.md` |
| szczegóły techniczne etapu 3, w tym podział SGKF | `3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcja 3 |
| kontrole skrótów: okres badań, modele kontrolne w odzyskiwaniu ukrytych | `3-sgkf-split/sgkf/kontrola_epoki.py`, `3-sgkf-split/sgkf/kontrola_skrotow.py` (`RAPORT_SGKF_MICE.md`, sekcje 5.2 i 5.4) |
| benchmark imputacji | `2-imputation/RAPORT_IMPUTACJA.md` |
| propozycja planu modelowania | `4-pu-setup/PLAN_MODELOWANIA.md` |
| stan na raport przejściowy (historyczny) | `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` |
| **nie do modeli** (imputacja przed podziałem) | `2-imputation/final/results/aneurysm_imputed_*.csv` |
