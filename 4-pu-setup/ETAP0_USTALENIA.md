# Etap 0 — ustalenia z diagnostyki danych

**Skrypt źródłowy:** `4-pu-setup/etap0_diagnostyka.py`
**Dane:** `data/processed/aneurysm_concatted_cleaned.csv` (78 197 × 39)
**Data:** 11.09.2026

Dokument odpowiada na pytania postawione w etapie 0 w `02_PLAN_PRZED_MODELOWANIEM.md`. Wnioski opierają się na wyliczeniach, nie na założeniach.

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

### Efekt epoki jest porównywalny z efektem choroby

Żeby sprawdzić, czy to realnie przeszkadza, porównaliśmy mediany cech w trzech grupach: KOR, NEURO z okna czasowego KOR oraz NEURO spoza tego okna. Różnica między dwiema grupami NEURO to **czysty efekt epoki** — ci sami chorzy, inne lata. Różnica KOR vs NEURO w tym samym oknie to **efekt choroby**.

| Cecha | KOR | NEURO w oknie | NEURO poza | Efekt epoki | Efekt choroby |
|---|---:|---:|---:|---:|---:|
| WBC | 8,78 | 7,25 | 8,48 | 1,23 | 1,53 |
| NEUT | 6,06 | 4,46 | 5,71 | 1,25 | 1,60 |
| GLU | 107,00 | 99,00 | 104,00 | 5,00 | 8,00 |
| Na | 139,67 | 141,00 | 139,25 | **1,75** | 1,33 |
| K | 4,30 | 4,30 | 4,10 | **0,20** | 0,00 |
| RDW | 13,60 | 13,50 | 13,63 | **0,13** | 0,10 |
| MCV | 88,93 | 90,45 | 90,00 | 0,45 | 1,52 |
| KREA | 0,88 | 0,81 | 0,79 | 0,02 | 0,07 |

Dla większości cech efekt epoki jest **tego samego rzędu** co efekt choroby, a dla Na, K i RDW jest **większy**. Model uczony na tych danych może więc w znacznej mierze rozpoznawać nie tętniaka, tylko to, z której dekady i z jakiego laboratorium pochodzi próbka.

Dochodzi jeszcze jeden czynnik: KOR to niemal w całości okres pandemii COVID-19, podczas gdy większość NEURO pochodzi sprzed 2020 roku. Parametry zapalne i limfocytarne różnią się w tym okresie systemowo.

> To porównanie median jest wskazaniem pierwszego rzędu, nie formalnym testem. Ale skala różnic wystarcza, żeby traktować problem poważnie.

### Co z tym zrobić — do decyzji zespołu

| Wariant | Na czym polega | Koszt |
|---|---|---|
| A. Ograniczyć do wspólnego okna | Tylko rekordy z 2019–2021 | Zostaje 271 pacjentów NEURO zamiast 1 823 — bardzo mała klasa pozytywna |
| B. Przyjąć i opisać | Modelujemy na całości, ograniczenie idzie do raportu | Wyniki mogą być zawyżone i trudne do obrony |
| C. Kontrolować epokę | Uwzględnić rok jako zmienną kontrolną albo dopasować kohorty | Więcej pracy, ale zachowuje dane |

Rekomendowałbym **wariant C z wariantem A jako analizą wrażliwości** — czyli model główny na całości z kontrolą epoki, a dodatkowo sprawdzenie na wąskim wspólnym oknie, czy wnioski się utrzymują.

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

To nie są błędy w danych. To pacjenci, którzy najpierw pojawili się w populacji ogólnej, a półtora roku później trafili do NEURO jako chorzy. **Ich rekordy KOR to jedyne w całym zbiorze pomiary, o których wiemy, że pochodzą sprzed rozpoznania u osoby faktycznie chorej** — czyli dokładnie taki materiał, jaki model miałby w praktyce przesiewowej.

**Propozycja:** przypisać tym pacjentom `label = 1` (są potwierdzonymi przypadkami), ale oznaczyć ich osobną flagą. Te 55 osób to naturalny, nieliczny, ale prawdziwy zbiór walidacyjny do sprawdzenia, czy model wykrywa ryzyko **przed** diagnozą — coś, czego sztuczne ukrywanie pozytywnych (punkt 2 planu) tylko udaje.

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

**Propozycja:** mediana jako reguła domyślna (odporna na pojedyncze skoki) oraz maksimum jako wariant w analizie wrażliwości. Decyzja powinna zapaść przed treningiem i zostać zamrożona.

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
| > 15 (nierealne dla mg/dl) | 767 | 1,19% |
| > 50 (wygląda na µmol/l) | 210 | 0,33% |

Problem dotyczy ok. 1% rekordów i jest **rozłożony podobnie w obu klasach** (1,00% w KOR, 0,78% w NEURO), więc nie zaburza systematycznie sygnału — ale wprowadza szum i powinien zostać naprawiony przed modelowaniem, bo dotyczy istotnej klinicznie cechy.

---

## Co z tego wynika dla planu

1. **Rozjazd czasowy kohort trafia do `02_PLAN_PRZED_MODELOWANIEM.md` jako nowy punkt etapu 0** — jest ważniejszy niż pierwotne pytanie o datę diagnozy i wymaga decyzji zespołu oraz konsultacji z prowadzącym.
2. **55 pacjentów z rekordami KOR sprzed przejścia do NEURO** to zasób, którego nie mieliśmy w planie — realny, choć mały, zbiór do walidacji przesiewowej.
3. **Reguła agregacji i naprawa KREA** mogą zostać rozstrzygnięte w zespole na podstawie powyższych liczb.
4. Pierwotne pytanie 0.2 o datę rozpoznania **nadal pozostaje bez odpowiedzi** — w danych nie ma kolumny z datą diagnozy. Dla 55 pacjentów z punktu 0.1 mamy jednak dolne oszacowanie: rozpoznanie nastąpiło po ich ostatnim badaniu w KOR.
