# Audyt projektu przed modelowaniem — 15.10.2026

## 1. Ocena ogólna

**Projekt ma sprawne, zweryfikowane technicznie wejście do eksperymentów modelowych. Nie ma jeszcze zamkniętego protokołu, który pozwalałby interpretować wynik jako skuteczność wykrywania tętniaka.** Największym ryzykiem jest rozpoznawanie źródła i okresu danych zamiast choroby. Ukrywanie pozytywnych pomaga mierzyć odzyskiwanie znanych przypadków, ale samo nie usuwa tego ryzyka.

Etapy 1–3 wykonano, jednak ich statusy są różne: etap 1 pozostaje historycznym procesem z notebooków i ma problemy z odtwarzalnością; etap 2 uzasadnia wybór praktycznej metody imputacji w ograniczonym benchmarku; aktualny etap 3 jest spójny z danymi i zapisanymi artefaktami. Etap 4 zawiera propozycję i analizę agregacji, bez bieżącej implementacji modelowania PU.

Audyt dotyczy gałęzi `pu-modeling-setup-lk`, bazowego commita `b17647c` i zastanego stanu roboczego. W repo były już niezacommitowane zmiany dokumentacji oraz katalogi `_podsumowanie/` i `dla claudea do analizy/`. Audyt ich nie zmienia. Nie zmieniono również danych, przydziału foldów ani kodu pipeline'u.

## 2. Co faktycznie zrobiono

| Etap | Wynik prac | Ocena na dziś |
|---|---|---|
| Przygotowanie danych | Spłaszczenie komórek z listami, usunięcie kolumn technicznych, filtracja, ograniczenie historii, połączenie kohort | Wykonane; pełne odtworzenie od raw wymaga uporządkowania |
| Benchmark imputacji | Wspólne porównanie KNN, MissForest i MICE/ExtraTrees; rozszerzone strojenie; walidacja masek | Uzasadnia MICE jako kompromis, z ograniczeniami opisanymi niżej |
| Przygotowanie do podziału | Rozstrzygnięcie 63 pacjentów obecnych w obu kohortach, usunięcie 145 rekordów KOR, 752 komórki zamienione na braki, powrót do 38 cech | Aktualny plik jest zgodny z kodem przygotowania |
| Podział | 5 foldów pacjentowych, stratyfikacja etykieta × okno czasowe × liczba rekordów, zapis przydziału i odcisku wejścia | Poprawny i odtworzony w obecnym środowisku |
| Imputacja w foldach | Scaler i MICE dopasowywane na treningu; kompletne lokalne pliki train/test | Aktualne artefakty sprawdzone pozytywnie |
| Ocena i diagnostyka | Kontrolowane maskowanie out-of-fold, wzorce panelowe, rozdzielność kohort, kontrola okresu badań | Przydatne wyniki eksploracyjne; wymagają ostrożnej interpretacji |
| Modelowanie PU | Plan metryk, ukrywania i zagnieżdżonej walidacji; historyczny prototyp opisany w dokumentacji | Protokół do ustalenia; implementacja docelowa jeszcze nie powstała |

Ważną poprawą względem czerwca jest cofnięcie selekcji CRP/MONO/%MONO opartej na etykiecie całego zbioru, zapis podziału i ilościowa diagnostyka różnic między kohortami. Podział na tabeli pacjentów przez `StratifiedKFold` jest poprawny: grupowanie wynika z jednego wiersza na pacjenta.

## 3. Wyniki kontroli wykonanych w audycie

### Dane i podział

| Właściwość | Wynik |
|---|---:|
| Rekordy w `data/processed/aneurysm_sgkf_input.csv` | 78 052 |
| Unikalni pacjenci | 40 924 |
| NEURO: rekordy / pacjenci | 6 651 / 1 823 |
| KOR: rekordy / pacjenci | 71 401 / 39 101 |
| Cechy / kolumny metadanych | 38 / 4 |
| Powtórzone `custom_id` w aktualnym wejściu | 0 |
| Pacjenci z niespójnymi etykietami w aktualnym wejściu | 0 |
| Puste metadane, nieskończone lub ujemne wartości cech | 0 |
| Testy przygotowania / podziału | 6/6 i 10/10 |
| Zgodność odcisku danych z metadanymi | `5b822c09ebc03011`, zgodny |
| Odtworzenie obecnego przydziału pacjentów | Identyczny wynik |

Braki w cechach przed imputacją są oczekiwane. Płeć jest kompletna i stała wewnątrz pacjenta. Wiek jest kompletny, może rosnąć wraz z historią pacjenta i nie wykazuje spadków w sprawdzonych danych.

### Zapisane wyniki imputacji

Sprawdzono wszystkie 10 plików w `3-sgkf-split/sgkf/results/foldy_imputowane_38/`:

- rekordy i ich kolejność odpowiadają aktualnemu podziałowi;
- pacjenci train/test są rozłączni w każdym foldzie;
- każdy pacjent występuje dokładnie w jednym outer-test;
- brak NaN, inf i wartości ujemnych po imputacji;
- 5/5 odcisków wyników zgadza się z `przebieg_imputacji_38.json`;
- wartości obserwowane są zachowane; maksymalne odchylenie numeryczne wynosi `2,27 × 10⁻¹³`.

Kod `aneurysm_sgkf_mice_pipeline.py:126–157` poprawnie wykonuje fit scalera i imputera na train, a na test tylko transform. Etykieta, identyfikatory i data nie są cechami imputera. Lekka kontrola na danych syntetycznych potwierdziła, że zmiana testu nie zmienia zaimputowanego treningu.

Środowisko lokalne: Python 3.9.6, scikit-learn 1.6.1, pandas 2.3.3, NumPy 2.0.2, SciPy 1.13.1, PyArrow 21.0.0. Podstawowe wersje są zgodne z metadanymi zapisanych wyników.

## 4. Najważniejsze kwestie przed modelowaniem

### A. Ukrywanie pozytywnych nie oddziela choroby od pochodzenia danych

Zapisana diagnostyka daje AUC pacjentowe 0,928 dla pełnych cech oraz 0,823 dla samych wskaźników braków. To mocny sygnał, że można rozpoznawać kohortę również przez sposób zbierania badań.

**Ukryty pacjent NEURO nadal zachowuje okres badań, wzorzec braków, profil hospitalizacji i ślady imputacji NEURO.** Model uczący się tych właściwości może osiągnąć wysoki `RecallHidden@q`. Wniosek ten wynika z konstrukcji proponowanego eksperymentu; nie wymaga stwierdzenia, że wszystkie różnice kohort są niezwiązane z chorobą.

Poprawna interpretacja: „odzyskiwanie ukrytych znanych przypadków NEURO w tym zbiorze”. Udowodnienie skuteczności wśród rzeczywistych, niewykrytych chorych KOR wymaga dodatkowej walidacji, np. niezależnej kohorty o porównywalnym sposobie zbierania danych lub weryfikacji klinicznej. Analizy wrażliwości ograniczą część ryzyk, ale nie zastąpią takiej walidacji.

Klasyczna korekta Elkan–Noto zakłada losowy wybór oznaczonych spośród wszystkich pozytywnych, niezależny od cech (SCAR). Losowe ukrywanie części NEURO nie dowodzi, że samo zebranie NEURO spełniało to założenie. Literatura rozróżnia SCAR i mechanizmy selekcji zależne od cech. [Elkan i Noto, 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf), [Bekker i Davis, 2018](https://proceedings.mlr.press/v94/bekker18a.html).

**Do ustalenia:** pochodzenie KOR, sposób potwierdzania rozpoznania w NEURO, relacja badań do diagnozy i leczenia, populacja docelowa oraz moment, w którym ma powstawać predykcja. Bez tego bezpieczny zakres projektu to eksperymentalny ranking PU w dostępnych kohortach. Dane w repo nie potwierdzają samodzielnie twierdzeń „KOR to populacja ogólna”, „nikt nie miał obrazowania” ani „wyniki są sprzed rozpoznania”.

### B. Historyczne czyszczenie nie jest niezależne od przyszłej oceny

Notebook `1-data-preparation/scripts-lf/aneurysm_data_analysis (1).ipynb:1503–1518` wyznacza kwantyle na całej analizowanej kohorcie. Wywołania czyszczenia KOR i NEURO mają różne ustawienia; limit historii wynosi odpowiednio 11 i 38 rekordów (`:6054–6055`).

Są to reguły zależne od źródła odpowiadającego etykiecie, wyznaczone przed podziałem. Oznacza to ryzyko selekcji oraz udział późniejszych danych testowych w ustalaniu statystyk czyszczenia. Nie jest to błąd obecnego fit MICE, ale poprawna imputacja w foldzie nie czyni automatycznie całego procesu wolnym od wycieku.

**Zalecenie:** jawnie opisać zbiór jako historycznie wyselekcjonowany. Przed wynikami potwierdzającymi przygotować wspólne, uzasadnione reguły; parametry uczone z danych wyznaczać na train. Jeśli pozostaje obecny zbiór, raportować wyniki jako eksploracyjne i nie twierdzić, że cały preprocessing został zwalidowany niezależnie. Dobór parametrów imputacji i wcześniejsze decyzje na całym zbiorze również należą do tej historii.

### C. Plan zagnieżdżonej walidacji jest poprawny, ale jeszcze niewykonany

`podzial.podzial_wewnetrzny()` istnieje, lecz nie ma bieżącego pipeline'u modeli, który dopasowuje preprocessing od nowa dla każdego inner-train. Plan w `4-pu-setup/PLAN_MODELOWANIA.md:81–94` opisuje właściwą zasadę.

Zapisane outer-foldy wystarczają do bazowego eksperymentu z konfiguracją ustaloną z góry. **Nie wolno dzielić zaimputowanego outer-train na inner-foldy i na tej podstawie stroić modeli.** Inner-validation uczestniczyłby już wtedy w dopasowaniu imputera. Ten brak trzeba zamknąć przed Optuną i raportowaniem porównania strojonych modeli.

### D. Protokół oceny wymaga jednoznacznej definicji

Przed pierwszym porównaniem trzeba zapisać:

1. jednostkę treningu i moment predykcji; agregację cech albo predykcji oraz kolejność względem imputacji;
2. główne `q`, udział ukrywania, ziarna i ewentualne `α`;
3. maskowanie osobno w treningu i ocenie, z wykluczeniem prawdziwego statusu ukrytych z wejścia modelu;
4. definicję puli U, zaokrąglenie liczby miejsc w top-q, remisy, próg `RecallKnown` i agregację metryk;
5. główny zestaw cech i z góry ustalone analizy wrażliwości;
6. porównywanie modeli na identycznych pacjentach, foldach i maskach oraz sposób liczenia niepewności.

Proponowane `true_label` nie powinno sugerować znanego statusu choroby KOR. Warto rozdzielić `original_label` (pochodzenie/znane P), `observed_label` po ukrywaniu oraz `is_hidden`; rzeczywisty status choroby KOR pozostaje nieznany. Metryki „ukryci kontra pozostałe U” również nie są pełną oceną choroby.

Sama korekta wyniku Elkan–Noto przez dzielenie przez dodatnią stałą zachowuje ranking w danym foldzie, więc nie poprawi `RecallHidden@q` wobec tego samego klasyfikatora bazowego. Wariant z ponownym ważonym treningiem jest odrębnym eksperymentem. Trzeba wskazać, którą wersję porównujemy. [Elkan i Noto, 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf).

## 5. Co naprawdę potwierdza ocena imputacji

W etapie 2 MissForest wygrywa na KOR (RMSE około 0,0783 wobec 0,0865 MICE), a MICE na NEURO (0,0978 wobec 0,1017). Wspólne parametry MICE: 7 iteracji, 78 drzew, głębokość 10, są uzasadnionym kompromisem. Nie dowiedziono, że to najlepsza metoda dla rzeczywistych braków lub końcowego zadania PU.

Zapisane wyniki etapu 3:

| Maska | KOR: RMSE / stosunek do średniej | NEURO: RMSE / stosunek do średniej |
|---|---|---|
| Losowe 10% komórek | 0,0829 / 0,6099 | 0,0801 / 0,6520 |
| Wzorce z niekompletnych rekordów treningowych | 0,1080 / 0,9824 | 0,0792 / 0,8910 |

Przy wzorcach panelowych przewaga MICE nad średnią wynosi około 1,8% RMSE na KOR i 10,9% na NEURO. To nie jest dowód identycznych wartości uzupełnień; tabela porównuje błędy.

Ocena wykorzystuje kompletne rekordy (`ocena_imputacji.py:99`): 20 383 KOR od 16 688 pacjentów i 772 NEURO od 582 pacjentów. Dla NEURO to 11,6% rekordów. Przeniesienie wzorców braków na kompletne rekordy przybliża strukturę maski, ale nie odtwarza nieznanego mechanizmu powstawania rzeczywistych braków ani nie daje ich prawdy referencyjnej.

**Nowe ustalenie:** w losowej masce NEURO 66,33% sumy kwadratów błędu pochodzi z płci, a 8,78% z wieku — razem 75,11%. Tymczasem wiek i płeć nie mają rzeczywistych braków. Wartości obliczono z zapisanych metryk per cecha: `SSE = RMSE² × liczba zamaskowanych komórek`, sumując pięć foldów. Ogólny wynik losowego benchmarku silnie zależy więc od sztucznego zadania uzupełniania demografii. W masce wzorców demografia nie jest ukrywana.

**Przed dużym strojeniem:** porównać wpływ średniej/mediany i MICE na końcowy ranking oraz osobno raportować laboratoryjną jakość imputacji. Jeśli ponawiany jest benchmark, maskować osobno cechy faktycznie niekompletne i pokazać wyniki per cecha/per panel. Nie ma potrzeby powtarzać wszystkich historycznych poszukiwań Optuny.

## 6. Konkretne błędy i nadmierne uproszczenia

| Ustalenie | Dowód | Zalecenie |
|---|---|---|
| Historyczny „duplikat” jest kolizją identyfikatora dwóch różnych tygodni | `clean_duplicate.py:21–26` usuwa rekord datowany 2017-01-01 na rzecz 2017-12-31. Daty odpowiadają różnym latom ISO; raw nie ma duplikatów pacjent + data ani pacjent + rok ISO + tydzień ISO | Poprawić definicję klucza i opis. Decyzję o odtworzeniu jednego rekordu uwzględnić przy ewentualnym nowym zamrożeniu danych |
| `custom_id` miesza rok kalendarzowy z tygodniem ISO | Niezgodność roku klucza z rokiem ISO dotyczy 583 rekordów KOR i 72 NEURO w aktualnym wejściu | Samo w sobie nie oznacza duplikatów ani błędnego splitu; używać spójnego klucza daty/ISO przy kolejnych przekształceniach |
| Opis seed=7 nie zgadza się z artefaktami | `RAPORT_IMPUTACJA.md:172` mówi o stałych parametrach bez Optuny. NEURO seed42 ma MICE 18/100, seed7 51/29; `run_all.py:102–106` wywołuje optymalizację | Ranking w zapisanych wynikach pozostaje taki sam, ale ten eksperyment nie izoluje wpływu samej maski |
| Walidacja 10 masek nie dotyczy jednego wspólnego zestawu parametrów na obu kohortach | Multi-seed NEURO używa 28/110; wspólne 7/78 porównano tam w cross-param na masce 42 | Rozdzielić dowód stabilności metody i dowód dla finalnej konfiguracji |
| „Uzupełniony/mały niezerowy NRBC oznacza NEURO” jest za mocne | Obserwowane NRBC: 77,57% zer; imputowane: 0% zer. Jednak wśród wszystkich rekordów z NRBC w (0; 0,05) tylko 19,30% należy do NEURO | Opisać jako ślad imputacji i wzbogacenie względem bazowych 8,52% rekordów NEURO; sprawdzić wariant bez NRBC/%NRBC |
| „CRP > 5 w 61% rekordów KOR” pomija mianownik | `analiza_wynikow.py:171` usuwa NaN. To 35 668 z 58 109 **zmierzonych** CRP, czyli 61,38%; względem wszystkich rekordów KOR: 49,95% z zaobserwowanym CRP > 5 | Podawać mianownik. Profil szpitalny pozostaje hipotezą do potwierdzenia |
| AUC 0,801 w oknie dat nie pochodzi z pipeline'u w pełni ograniczonego do okna | `analiza_wynikow.py:143–145` filtruje zapisane dane po MICE dopasowanym także na innych okresach outer-train | W analizie wrażliwości ograniczyć dane przed dopasowaniem preprocessingu |
| Kontrola okresu nie rozstrzyga ilościowo, że sygnał pochodzi „głównie” z braków | AUC 0,756 dla wskaźników braków oraz 0,562–0,656 dla kompletnych paneli dotyczą różnych populacji rekordów; 38 cech z NaN daje 0,841 | Stwierdzić, że braki niosą istotny sygnał okresu. Silniejszy wniosek wymaga porównania na tych samych rekordach |

Kolizja klucza jest faktycznym błędem czyszczenia o małym zakresie liczbowym. Pozostałe ustalenia nie unieważniają sprawdzonej poprawności bieżących foldów. Ich znaczenie dotyczy interpretacji i możliwości odtworzenia procesu.

## 7. Uzupełnienia techniczne przed dłuższymi obliczeniami

1. **Walidacja odczytywanego przydziału.** `podzial.py:349–363` sprawdza odcisk danych, ale nie integralność samego `pacjent_fold.csv`. Na kopii w `/tmp` loader przyjął fold 99. Należy kontrolować unikalność i kompletność pacjentów, dozwolone foldy, etykiety oraz odcisk przydziału. Aktualny plik jest poprawny.
2. **Manifest i bezpieczny zapis imputacji.** `analiza_wynikow.py:66–68` odczytuje parquet po nazwie, bez automatycznego potwierdzenia splitu/konfiguracji/odcisków. Kolejne nadpisywanie plików może po przerwaniu zostawić mieszany przebieg. Zapisywać cały przebieg do nowego katalogu i zatwierdzać manifest po ukończeniu. W tym audycie obecne pliki sprawdzono ręcznie.
3. **Pełna recepta środowiska.** Zapisać zgodny zestaw wersji i zależności, w tym PyArrow. Dolne granice w `2-imputation/final/requirements.txt` oraz wersje w JSON nie zastępują odtwarzalnego środowiska całego projektu.
4. **Uczciwa granica odtwarzalności.** `uruchom_sgkf.py` odtwarza etap 3 od gotowego `aneurysm_concatted.csv`, nie cały proces od raw. Historyczne `clean_full.py:16–25` i `clean_duplicate.py:10–14` mają niespójne ścieżki po zmianie katalogu roboczego. Docelowo potrzebny jest jawny, wykonywalny przepływ raw → wejście etapu 3.

Istnieją też nieidentyczne kopie danych pośrednich: `data/interim/cleaned/neuro_shortend.csv` ma 6616 rekordów i 1820 pacjentów, a kanoniczny `data/imputation-inputs/neuro_shortend.csv` 6651 i 1823. Plik `aneurysm_data_analysis.ipynb` bez `(1)` jest zapisanym HTML-em, podczas gdy wariant `(1)` jest notebookiem JSON. Należy jednoznacznie oznaczyć wersje historyczne oraz wejścia obowiązujące, aby kolejny etap nie wczytał przypadkowej kopii.

## 8. Proponowana kolejność domknięcia

| Kolejność | Zadanie | Warunek ukończenia |
|---|---|---|
| 1 | Potwierdzić pochodzenie kohort i cel predykcji | Jednoznaczny opis populacji, rozpoznania i czasu dostępności cech albo jawnie ograniczony cel eksploracyjny |
| 2 | Poprawić opis dotychczasowych dowodów | Rozstrzygnięte: kolizja klucza, seed7, zakres multi-seed, mianowniki CRP, siła wniosków o NRBC i okresie |
| 3 | Zdecydować o wersji danych | Zachowanie historycznej wersji z jawnymi ograniczeniami albo korekta procesu i ponowne zamrożenie z nowymi odciskami |
| 4 | Zamrozić protokół modelowania | Jednostka i agregacja, maski, q/udział/ziarna, metryki, cechy, scenariusze i kryteria porównań |
| 5 | Domknąć kontrolę artefaktów i inner preprocessing | Odczyt odrzuca niespójny split/cache; każde inner dopasowanie korzysta wyłącznie z inner-train |
| 6 | Uruchomić ograniczony zestaw modeli kontrolnych | Losowy ranking, wiek/płeć, sam wzorzec braków, baseline P-vs-U, co najmniej jedna metoda PU; jednakowe maski |
| 7 | Wykonać zaplanowane analizy wrażliwości | Bez NRBC/%NRBC, wybrane częste panele, wspólne okno z preprocessingiem dopasowanym w oknie; porównanie prostej imputacji z MICE |
| 8 | Dopiero potem rozszerzać strojenie i interpretację | Predykcje OOF, niepewność na poziomie pacjenta, raport ograniczeń i pełna konfiguracja |

Mediana jest rozsądnym kandydatem do agregacji, ale istniejąca analiza `4-pu-setup/etap0_agregacja.py:33` dotyczy starszego wejścia z 35 cechami. Wariant `groupby.last()` oznacza ostatnią niebrakującą wartość każdej cechy, niekoniecznie jeden ostatni rekord. Nie należy traktować tej analizy jako zamknięcia wyboru profilu pacjenta dla obecnych danych.

We wspólnym oknie 2019-12-29–2021-12-26 jest 680 rekordów NEURO od 271 pacjentów; warunek co najmniej połowy historii w oknie spełnia 159 pacjentów. To różne definicje. Przy 40% ukrywania i pięciu foldach 271 pacjentów oznacza średnio około 22 ukrytych na fold, zatem wynik będzie znacznie mniej precyzyjny niż na pełnych danych. Mała próba ogranicza siłę wniosku, ale nie usuwa potrzeby kontroli okresu.

## 9. Zakres i odtworzenie audytu

Wykonano analizę bieżącego kodu, odczyt danych i wszystkich outer-parquet, istniejące testy, weryfikację podziału i odcisków oraz lekkie kontrole syntetyczne. Ponownie obliczono liczebności, kompletność, zgodność obserwowanych wartości, wybrane statystyki NRBC/CRP i udział demografii w błędzie imputacji.

**Nie uruchamiano pełnego MICE, Optuny ani ponownie modeli diagnostycznych.** Podane AUC i zagregowane RMSE pochodzą z zapisanych wyników, których odciski wejścia są zgodne z aktualnymi danymi. Testy potwierdzają zgodność aktualnego wejścia z przygotowaniem etapu 3, a nie pełnego procesu od surowych danych.

Podstawowe kontrole bez nadpisywania danych:

```bash
.venv/bin/python 3-sgkf-split/przygotowanie/tests/test_przygotowanie.py
.venv/bin/python 3-sgkf-split/sgkf/tests/test_podzial.py
.venv/bin/python 3-sgkf-split/sgkf/podzial.py --sprawdz
```

Źródła lokalne: `README.md`, `1026_SPOTKANIE_PROWADZACY.md`, `1026_REALIZACJA_DO_SGKF.md`, kod i wyniki `1-data-preparation/`, `2-imputation/final/`, `3-sgkf-split/`, `4-pu-setup/PLAN_MODELOWANIA.md`. Najważniejsze źródła liczb: `podsumowanie.json`, `podzial_meta.json`, `przebieg_imputacji_38.json`, `ocena_imputacji.json`, `analiza_wynikow.json`, `kontrola_epoki.json`.
