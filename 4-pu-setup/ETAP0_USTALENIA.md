# Etap 0 — ustalenia z diagnostyki danych

**Skrypt źródłowy:** `4-pu-setup/etap0_diagnostyka.py`
**Dane:** `data/processed/aneurysm_concatted_cleaned.csv` (78 197 × 39)
**Data:** 11.09.2026
**Status:** wyniki diagnostyki i propozycje robocze — decyzje protokołu nie są jeszcze zamrożone

Dokument dostarcza danych do pytań postawionych w etapie 0 w `02_PLAN_PRZED_MODELOWANIEM.md`. Liczby opierają się na wyliczeniach, natomiast interpretacje i propozycje wymagają zatwierdzenia przez zespół oraz, tam gdzie wskazano, konsultacji klinicznej.

---

## Najważniejsze: kohorty prawie nie pokrywają się w czasie

To znalezisko jest poważniejsze niż pierwotnie sformułowane pytanie o datę diagnozy i wymaga decyzji przed jakimkolwiek modelowaniem.

**Rozkład rekordów według roku badania:**

| Rok | KOR | NEURO |
|---|---:|---:|
| 2000 | 0 | 1 |
| 2010–2018 | **0** | 3 937 |
| 2019 | 326 | 534 |
| 2020 | 31 321 | 317 |
| 2021 | 39 899 | 358 |
| 2022–2024 | **0** | 1 504 |

KOR istnieje praktycznie wyłącznie w latach **2020–2021** (71 220 z 71 546 rekordów, 99,5%). NEURO rozciąga się na **24 lata**, od 2000 do 2024.

**W oknie czasowym KOR (2019-12-29 – 2021-12-26) mieści się tylko 10,2% rekordów NEURO** (680 z 6 651) i 271 z 1 823 pacjentów. Pozostałe 90% pochodzi z lat, w których KOR nie ma ani jednego rekordu.

### Różnice kalendarzowe są porównywalne z różnicami między kohortami

Żeby wstępnie ocenić skalę problemu, porównaliśmy mediany cech w trzech grupach: KOR, NEURO z okna czasowego KOR oraz NEURO spoza tego okna. Różnica między dwiema grupami NEURO może wynikać z epoki, laboratorium, zmian praktyki klinicznej lub różnego składu pacjentów. Nie są to te same osoby mierzone w dwóch okresach, dlatego porównanie nie izoluje czystego efektu czasu. Analogicznie różnica KOR vs NEURO w tym samym oknie jest różnicą między kohortami, a nie czystym efektem choroby.

| Cecha | KOR | NEURO w oknie | NEURO poza | Różnica NEURO między okresami | Różnica kohort w oknie |
|---|---:|---:|---:|---:|---:|
| WBC | 8,78 | 7,25 | 8,48 | 1,23 | 1,53 |
| NEUT | 6,06 | 4,46 | 5,71 | 1,25 | 1,60 |
| GLU | 107,00 | 99,00 | 104,00 | 5,00 | 8,00 |
| Na | 139,67 | 141,00 | 139,25 | **1,75** | 1,33 |
| K | 4,30 | 4,30 | 4,10 | **0,20** | 0,00 |
| RDW | 13,60 | 13,50 | 13,63 | **0,13** | 0,10 |
| MCV | 88,93 | 90,45 | 90,00 | 0,45 | 1,52 |
| KREA | 0,88 | 0,81 | 0,79 | 0,02 | 0,07 |

Dla większości cech różnica między okresami jest **tego samego rzędu** co różnica między kohortami, a dla Na, K i RDW jest **większa**. Model uczony na tych danych może więc w znacznej mierze rozpoznawać nie tętniaka, tylko epokę, laboratorium, sposób pozyskania danych albo odmienny skład pacjentów.

Dochodzi jeszcze jeden potencjalny czynnik: KOR pochodzi niemal w całości z okresu pandemii COVID-19, podczas gdy większość NEURO pochodzi sprzed 2020 roku. Bez dodatkowych danych nie można jednak przypisać obserwowanych różnic pandemii ani oddzielić jej wpływu od zmian laboratoryjnych i składu kohort.

> To porównanie median jest wskazaniem pierwszego rzędu, nie formalnym testem. Ale skala różnic wystarcza, żeby traktować problem poważnie.

### Co z tym zrobić — do decyzji zespołu

| Wariant | Na czym polega | Koszt |
|---|---|---|
| A. Ograniczyć do wspólnego okna | Tylko rekordy z 2019–2021 | Zostaje 271 pacjentów NEURO zamiast 1 823 — bardzo mała klasa pozytywna |
| B. Przyjąć i opisać | Modelujemy na całości, ograniczenie idzie do raportu | Wyniki mogą być zawyżone i trudne do obrony |
| C. Kontrolować czas i źródło | Dopasować/ważyć kohorty po czasie i dostępnych zmiennych; nie używać roku jako prostego predyktora pochodzenia | Więcej pracy, ale zachowuje dane |

Rekomendowany jest **wariant C z wariantem A jako analizą wrażliwości** — model główny z jawnie zdefiniowaną kontrolą czasu i źródła oraz dodatkowe sprawdzenie w wąskim wspólnym oknie. Samo dodanie roku jako cechy nie wystarcza i może wręcz ułatwić rozpoznawanie kohorty.

---

## 0.1 — 63 pacjentów z obiema etykietami

Ci pacjenci to łącznie 264 rekordy (145 KOR + 119 NEURO), mediana 1 rekord KOR i 1 NEURO na osobę.

**Relacja czasowa jest bardzo wymowna:**

| Układ | Liczba pacjentów |
|---|---:|
| wszystkie rekordy KOR **przed** NEURO | **55** |
| wszystkie NEURO przed KOR | 6 |
| przeplatane w czasie | 2 |

Dla tych 55 przypadków odstęp między ostatnim badaniem w KOR a pierwszym w NEURO wynosi **medianę 546 dni** (min 35, maks. 1 365).

Nie wygląda to na zwykłe duplikaty techniczne. Są to pacjenci, którzy najpierw pojawili się w źródle KOR, a później w źródle NEURO. Bez daty rozpoznania nie wiemy jednak, czy rekordy KOR rzeczywiście pochodzą sprzed diagnozy; wiemy jedynie, że poprzedzają pierwszy zachowany rekord NEURO.

**Propozycja robocza:** przypisać tym pacjentom pacjentowy status pozytywny, zachować osobną flagę źródła i analizować wcześniejsze rekordy KOR jako kohortę przejścia KOR→NEURO. Może ona stać się cenną walidacją czasową dopiero po potwierdzeniu daty diagnozy lub klinicznego znaczenia pierwszego rekordu NEURO. Do tego czasu nie należy nazywać jej walidacją przeddiagnozową.

Pozostałych 8 przypadków (6 + 2) trzeba obejrzeć ręcznie.

---

## 0.3 — agregacja rekordów do pacjenta

| Grupa | Liczba | Udział |
|---|---:|---:|
| Pacjenci z 1 rekordem — wybór reguły bez znaczenia | 24 027 | 58,7% |
| Pacjenci z >1 rekordem — wybór reguły ma znaczenie | 16 897 | 41,3% |

Rozrzut wartości w obrębie jednego pacjenta (tylko pacjenci z wieloma rekordami):

| Cecha | Mediana \|max−min\| | Mediana odch. std |
|---|---:|---:|
| WBC | 2,78 | 1,68 |
| GLU | 2,00 | 15,98 |
| KREA | 0,14 | 0,10 |
| PLT | 49,00 | 30,64 |

Przy 41% pacjentów i takim rozrzucie (PLT potrafi się wahać o 49 jednostek u tej samej osoby) wybór między średnią, medianą, maksimum a ostatnim pomiarem realnie zmienia wynik. Wysokie odchylenie standardowe GLU sugeruje pojedyncze skoki glikemii — tam średnia i maksimum dadzą zupełnie różne obrazy.

### Porównanie czterech reguł — wyniki

Skrypt `4-pu-setup/etap0_agregacja.py` (branch `etap0-agregacja-lk`) porównał średnią, medianę, maksimum i ostatni pomiar. Punktem wyjścia jest nierównowaga, która przesądza sprawę:

| Klasa | Pacjentów | Mediana rekordów | Średnia | Maks. | Ma >1 rekord |
|---|---:|---:|---:|---:|---:|
| KOR | 39 164 | 1 | 1,83 | 11 | 40% |
| NEURO | 1 823 | 2 | 3,65 | 38 | 67% |

Pacjenci NEURO mają średnio dwukrotnie więcej badań. Każda reguła wrażliwa na liczbę pomiarów tworzy więc sygnał z samej częstości badania — a częstość wynika z hospitalizacji, nie ze stanu zdrowia.

**Czy reguła przemyca liczbę badań** (korelacja Spearmana liczby rekordów z wartością cechy, liczona wyłącznie wewnątrz KOR, żeby nie mieszać z efektem choroby):

| Reguła | Mediana \|ρ\| | Maks. \|ρ\| | Cech z \|ρ\| > 0,2 |
|---|---:|---:|---:|
| mediana | 0,079 | 0,257 | **2** |
| ostatni | 0,083 | 0,296 | 1 |
| średnia | 0,125 | 0,284 | 6 |
| maksimum | 0,190 | 0,379 | **17** |

**Czy reguła wciąga wartości nierealne** (liczba pacjentów poza orientacyjnym zakresem przeżycia):

| Cecha | Mediana | Średnia | Ostatni | Maksimum |
|---|---:|---:|---:|---:|
| K | 20 | 24 | 31 | **78** |
| WBC | 1 | 1 | 2 | **7** |

Pozorna siła sygnału jest przy tym niemal identyczna dla średniej, mediany i maksimum (mediana \|AUC−0,5\| odpowiednio 0,0440, 0,0438 i 0,0467), więc maksimum nie kupuje nam nic w zamian za te wady. Reguły dają zbliżone, ale nie wymienne profile — korelacja między nimi waha się od 0,985 (średnia vs mediana) do 0,806 (maksimum vs ostatni).

**Propozycja:** **mediana jako reguła główna**, **średnia jako analiza wrażliwości** (najbliższa medianie, ρ = 0,985).

Maksimum odpada — nie jako wariant zapasowy, tylko w ogóle. Koreluje z liczbą badań dla 17 cech i wciąga cztery razy więcej pacjentów z niemożliwym potasem, czyli wzmacnia dokładnie te błędy wpisu, które opisano w punkcie 0.5.

Ostatni pomiar wypada dobrze w liczbach, ale przy założeniu, że wyniki NEURO pochodzą z hospitalizacji, „ostatni" oznacza u chorych pomiar po leczeniu, a u KOR zwykły wynik kontrolny. To reguła, która znaczy co innego w każdej klasie, więc jej nie bierzemy.

Decyzja powinna zostać zamrożona przed treningiem.

---

## 0.5 — wartości skrajne

| Cecha | Braki | Mediana | p99 | Maks. | Wartości nierealne |
|---|---:|---:|---:|---:|---|
| WBC | 2 648 | 8,75 | 25,43 | 253,06 | 7 wartości > 200 |
| Na | 12 514 | 139,67 | 149,00 | 177,25 | 8 wartości < 100 (min 74) |
| K | 12 747 | 4,30 | 6,00 | 28,32 | **86 wartości > 9** u 78 pacjentów |
| KREA | 13 643 | 0,87 | 19,56 | 196,15 | patrz niżej |

Potas powyżej 9 mmol/l jest stanem zagrażającym życiu, a wartości rzędu 26–28 są fizjologicznie niemożliwe — to niemal na pewno błędy wpisu albo hemoliza próbki.

### KREA — podejrzenie pomieszanych jednostek

Mediana 0,87 wskazuje na **mg/dl**, ale p99 wynosi 19,56, a maksimum 196,15 — wartości nieosiągalne w tej jednostce, za to typowe dla **µmol/l**.

| Próg | Liczba wartości | Udział |
|---|---:|---:|
| > 15 (wartość skrajna w mg/dl, wymaga weryfikacji) | 767 | 1,19% |
| > 50 (niezgodne z typowym zakresem mg/dl, możliwe µmol/l) | 210 | 0,33% |

Problem dotyczy ok. 1% rekordów i występuje w obu kohortach, ale podobny udział nie dowodzi braku systematycznego biasu. Jednostkę trzeba odzyskać z danych źródłowych dla konkretnego pomiaru; prosta korekta wszystkich wartości powyżej jednego progu mogłaby zmienić prawdziwe przypadki ciężkiej niewydolności nerek.

---

## Co z tego wynika dla planu

1. **Rozjazd czasowy kohort trafia do `02_PLAN_PRZED_MODELOWANIEM.md` jako nowy punkt etapu 0** — jest ważniejszy niż pierwotne pytanie o datę diagnozy i wymaga decyzji zespołu oraz konsultacji z prowadzącym.
2. **55 pacjentów z rekordami KOR poprzedzającymi pierwszy rekord NEURO** to zasób, którego nie mieliśmy w planie — potencjalny mały zbiór walidacji czasowej po potwierdzeniu relacji względem diagnozy.
3. **Reguła agregacji i naprawa KREA** mogą zostać rozstrzygnięte w zespole na podstawie powyższych liczb.
4. Pierwotne pytanie 0.2 o datę rozpoznania **nadal pozostaje bez odpowiedzi** — w danych nie ma kolumny z datą diagnozy. Dla 55 pacjentów z punktu 0.1 znamy wyłącznie kolejność źródeł KOR→NEURO; nie daje ona dolnego oszacowania daty rozpoznania.
