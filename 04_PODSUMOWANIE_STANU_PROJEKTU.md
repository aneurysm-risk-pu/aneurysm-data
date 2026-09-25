# Podsumowanie stanu projektu

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zespół:** Łukasz Kubik (kierownik), Liwia Florkiewicz, Dominika Malisz, Adrian Meredyk
**Opiekun:** dr inż. Patryk Jasik (IFiIS)
**Data:** 20.09.2026

Dokument spina w jedno miejsce wszystko, co wiemy i co zrobiliśmy: stan danych, wszystkie problemy wykryte w etapie 0 wraz z liczbami i konsekwencjami, gotową implementację punktów 1–5 oraz listę decyzji, które musi podjąć zespół.

Jest napisany tak, żeby dało się go przeczytać bez zaglądania do kodu i bez pamiętania wcześniejszych ustaleń. Każde twierdzenie opiera się na wyliczeniu, nie na przypuszczeniu — liczby pochodzą z aktualnych plików w repozytorium.

**Mapa dokumentów:**

| Dokument | Zawiera |
|---|---|
| `01_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` | prace do raportu przejściowego |
| `02_PLAN_PRZED_MODELOWANIEM.md` | etap 0 i punkty 1–5 |
| `03_PLAN_MODELOWANIE.md` | punkty 6–9, trening |
| `4-pu-setup/ETAP0_USTALENIA.md` | wyniki diagnostyki danych |
| `4-pu-setup/DOKUMENTACJA_PIPELINE.md` | dokumentacja techniczna kodu |
| `4-pu-setup/PYTANIA_DO_PROWADZACEGO.md` | kwestie wymagające potwierdzenia klinicznego |
| **ten dokument** | **synteza całości** |

---

## Część I — Na czym polega ten projekt

### Problem w jednym akapicie

Mamy dwie bazy wyników laboratoryjnych. Pierwsza, **KOR**, to populacja ogólna — nie wiemy o tych ludziach nic poza morfologią krwi i biochemią. Druga, **NEURO**, to pacjenci oddziału neurologicznego z rozpoznanym tętniakiem mózgu. Chcemy nauczyć model wskazywać osoby o profilu podobnym do chorych, żeby skierować je na badanie obrazowe.

### Dlaczego to nie jest zwykła klasyfikacja

Kluczowa trudność: **nie mamy grupy kontrolnej.** Gdybyśmy mieli listę osób z potwierdzonym brakiem tętniaka, byłby to zwykły problem klasyfikacji dwuklasowej. Ale nikt nie przebadał obrazowo 39 164 pacjentów KOR. Wśród nich prawie na pewno są osoby z niewykrytym tętniakiem — tętniaki bywają bezobjawowe przez całe życie.

Formalnie: mamy klasę **P** (positive, potwierdzeni chorzy) i klasę **U** (unlabeled, nieoznaczeni), a nie P i N. Stąd nazwa **Positive-Unlabeled learning**.

Ma to trzy poważne konsekwencje:

1. **Nie możemy zmierzyć, ile pomyłek robi model.** Jeśli model wskaże pacjenta z KOR jako podwyższonego ryzyka, nie wiemy, czy się pomylił, czy właśnie wykrył niezdiagnozowanego chorego. Klasyczne „false positive" traci sens.
2. **Accuracy jest bezużyteczna.** Model, który każdemu przypisze „zdrowy", osiągnie 95,5% trafności, bo tylko 4,45% pacjentów ma etykietę pozytywną. I będzie kompletnie bezwartościowy.
3. **Potrzebujemy zastępczego sposobu oceny.** Stosujemy **kontrolowany test odzyskiwania**: bierzemy część znanych chorych, ukrywamy ich w grupie nieoznaczonej i sprawdzamy, czy model mimo to wypchnie ich na górę rankingu ryzyka. To jest pomysł z uwag prowadzącego i stanowi rdzeń naszej ewaluacji.

### Skąd wiadomo, że to zadziała

Nie wiadomo z góry — i to jest uczciwa odpowiedź. Dlatego cały protokół jest zbudowany wokół jednego pytania: *czy wynik, który zobaczymy, mówi coś o tętniakach, czy tylko o tym, jak powstały nasze dane?* Większość problemów opisanych w Części III to właśnie odpowiedzi na to pytanie.

---

## Część II — Co zostało zrobione do tej pory

### Etap 1: eksploracja i czyszczenie danych

Dane surowe to dwa pliki po 128 kolumn:

| Kohorta | Plik surowy | Wiersze × kolumny |
|---|---|---|
| KOR | `data/raw/kor_merged_aggregated_1W_mean.csv` | 73 679 × 128 |
| NEURO | `data/raw/neuro_merged_aggregated_1W_mean.csv` | 7 655 × 128 |

Jeden rekord to **tydzień zagregowanych wyników** jednego pacjenta, z kluczem `{patient_id}-{rok}-W{tydzień}`.

Co zrobiono: usunięto kolumny z zakresami referencyjnymi i zmienne techniczne, naprawiono kolumny zapisane błędnie jako `{'values': [...]}`, odfiltrowano wiek poza 18–100 lat, usunięto pacjentów i kolumny z ponad 60% braków (1 793 pacjentów KOR), obcięto najstarsze pomiary u pacjentów z długą historią hospitalizacji.

Zbadano też zależność cech z etykietą czterema miarami (Pearson, Spearman, Kendall, Phi-K) i usunięto trzy najsłabiej związane: CRP, MONO i %MONO. **Ta decyzja okazała się problematyczna** — patrz problem 4 w Części III.

**Wynik:** jeden zbiór **78 197 wierszy × 39 kolumn** (35 cech + 4 kolumny meta).

| Klasa | Rekordy | Pacjenci |
|---|---:|---:|
| `label = 0` (KOR, nieoznaczeni) | 71 546 | 39 164 |
| `label = 1` (NEURO, pozytywni) | 6 651 | 1 823 |
| **Razem** | **78 197** | **40 924** |

Rekordów na pacjenta: mediana 1, średnia 1,91, maksimum 38.

### Etap 2: uzupełnianie braków danych

W finalnym zbiorze brakuje **377 236 wartości, czyli 13,8% wszystkich komórek**. Kompletnych wierszy jest tylko 23 092 z 78 197. Bez uzupełnienia braków większość modeli po prostu nie zadziała.

Porównano trzy metody na tej samej, losowo zamaskowanej próbce. Zabezpieczenie, które warto podkreślić: maska do strojenia parametrów (seed 43) była **oddzielona** od maski do oceny (seed 42), więc parametry nie były dopasowywane do tych samych braków, na których liczono wynik.

| Zbiór | MissForest | MICE | KNN |
|---|---|---|---|
| KOR | **0,0783** | 0,0865 | 0,1022 |
| NEURO | 0,1017 | **0,0978** | 0,1291 |

Żadna metoda nie wygrała na obu zbiorach. MissForest korzysta z dużej liczby kompletnych obserwacji w KOR, MICE lepiej radzi sobie na małym NEURO. Dla jednego spójnego pipeline'u wybrano **MICE z estymatorem ExtraTrees**: `max_iter=7, n_estimators=78, max_depth=10`.

Stabilność sprawdzono na niezależnej masce (seed 7) i na wielu ziarnach — ranking metod się nie zmienił.

**Zastrzeżenie techniczne:** to, co nazywamy MICE, jest technicznie pojedynczą, deterministyczną imputacją iteracyjną (`sample_posterior=False`), a nie klasycznym multiple imputation generującym wiele kompletnych zbiorów. Nie propagujemy niepewności uzupełnienia do wyników końcowych.

### Etap 3: kontrolowany podział danych

Zwykły losowy podział był wykluczony, bo jeden pacjent ma zwykle kilka rekordów. Część jego pomiarów trafiłaby do treningu, część do walidacji, a model byłby oceniany na osobie, którą już widział — i dostałby za to sztucznie wysoką ocenę.

Zastosowano **StratifiedGroupKFold** na 5 foldach: *Group* pilnuje, żeby wszystkie rekordy jednego pacjenta trafiły do jednej części, *Stratified* wyrównuje proporcje klas.

Imputację **wciągnięto do wnętrza foldu**: scaler uczy się tylko na kompletnych wierszach treningu, imputer tylko na treningu, walidacja jest jedynie transformowana. Dzięki temu informacja ze zbioru walidacyjnego nie wpływa na uzupełnianie braków treningowych.

To był stan na raport przejściowy. Potem zaczęła się weryfikacja — i wtedy wyszło to, co opisuje Część III.

---

## Część III — Etap 0: wszystkie wykryte problemy

Etap 0 to nie praca programistyczna, tylko **decyzje protokołu**. Każdy problem poniżej został wykryty przez policzenie czegoś w danych, nie przez przypuszczenie. Skrypty: `4-pu-setup/etap0_diagnostyka.py`, `etap0_agregacja.py`, `etap0_sgkf_stabilnosc.py`.

Problemy są ułożone od najgroźniejszego.

---

### Problem 1 (0.2): Kohorty prawie nie pokrywają się w czasie

**Co widzimy.** Rozkład rekordów według roku badania:

| Rok | KOR | NEURO |
|---|---:|---:|
| 2000 | 0 | 1 |
| 2010–2018 | **0** | 3 937 |
| 2019 | 326 | 534 |
| 2020 | 31 321 | 317 |
| 2021 | 39 899 | 358 |
| 2022–2024 | **0** | 1 504 |

KOR istnieje praktycznie wyłącznie w latach **2020–2021** — 71 220 z 71 546 rekordów, czyli 99,5%. NEURO rozciąga się na **24 lata**.

We wspólnym oknie dat (2019-12-29 – 2021-12-26) mieści się tylko **680 z 6 651 rekordów NEURO** (10,2%) i **271 z 1 823 pacjentów**. Pozostałe 90% pochodzi z lat, w których KOR nie ma ani jednego rekordu.

**Dlaczego to groźne.** Wyobraźmy sobie, że model dostaje dwa profile krwi: jeden z 2014 roku, drugi z 2021. Jeśli aparatura, odczynniki albo sposób raportowania zmieniły się przez te lata, model może nauczyć się rozpoznawać **rok badania**, a nie chorobę. A ponieważ prawie wszyscy chorzy są „starzy", a prawie wszyscy nieoznaczeni „nowi", rozpoznanie roku jest równoznaczne z rozpoznaniem etykiety. Model osiągnąłby świetny wynik, nie wiedząc nic o tętniakach.

**Jak sprawdziliśmy skalę.** Porównaliśmy mediany cech w trzech grupach: KOR, NEURO z okna czasowego KOR i NEURO spoza tego okna. Różnica między dwiema grupami NEURO to różnica między okresami przy tej samej chorobie. Różnica KOR vs NEURO w tym samym oknie to różnica między kohortami przy tym samym okresie.

| Cecha | KOR | NEURO w oknie | NEURO poza | Różnica między okresami | Różnica między kohortami |
|---|---:|---:|---:|---:|---:|
| WBC | 8,78 | 7,25 | 8,48 | 1,23 | 1,53 |
| NEUT | 6,06 | 4,46 | 5,71 | 1,25 | 1,60 |
| GLU | 107,00 | 99,00 | 104,00 | 5,00 | 8,00 |
| Na | 139,67 | 141,00 | 139,25 | **1,75** | 1,33 |
| K | 4,30 | 4,30 | 4,10 | **0,20** | 0,00 |
| RDW | 13,60 | 13,50 | 13,63 | **0,13** | 0,10 |
| MCV | 88,93 | 90,45 | 90,00 | 0,45 | 1,52 |
| KREA | 0,88 | 0,81 | 0,79 | 0,02 | 0,07 |

Dla większości cech różnica między okresami jest **tego samego rzędu** co różnica między kohortami. Dla sodu, potasu i RDW jest **większa**.

Dochodzi jeszcze jeden czynnik: KOR to niemal w całości okres pandemii COVID-19, a większość NEURO pochodzi sprzed 2020 roku.

**Ważne zastrzeżenie:** to porównanie median nie jest formalnym dowodem. Dwie grupy NEURO to nie te same osoby mierzone dwa razy, więc różnica między nimi może wynikać z epoki, ale też ze zmian w praktyce klinicznej czy z innego składu pacjentów. Skala różnic wystarcza jednak, żeby traktować problem poważnie.

**Warianty rozwiązania:**

| Wariant | Na czym polega | Koszt |
|---|---|---|
| **A** | tylko rekordy z okna 2019–2021 | zostaje **271 pozytywnych pacjentów** zamiast 1 823 |
| **B** | modelujemy na całości, ograniczenie idzie do raportu | wyniki trudne do obrony |
| **C** | dopasowanie albo ważenie kohort po czasie i źródle | więcej pracy, zachowuje dane |

**Rekomendacja: wariant C, z wariantem A jako analizą wrażliwości.** Samo dodanie roku jako zwykłej cechy **nie rozwiązuje problemu** — przeciwnie, ułatwia modelowi rozpoznanie kohorty.

**Argument liczbowy przeciwko wariantowi A jako scenariuszowi głównemu:** przy 271 pozytywnych pacjentach i ukrywaniu 40% na fold testowy przypadłyby ~22 ukryte osoby. Rozdzielczość głównej metryki spadłaby do 4,5 punktu procentowego, czyli jeden trafiony pacjent zmieniałby wynik o 4,5 pp. Przy pełnym zbiorze jest to 0,7 pp.

**Status: DO DECYZJI ZESPOŁU I KONSULTACJI.**

---

### Problem 2 (0.2): Nie wiemy, kiedy pobrano krew chorym

**Co widzimy.** W danych nie ma kolumny z datą rozpoznania tętniaka. Mamy tylko datę badania laboratoryjnego.

Jest za to wskazówka pośrednia. Spośród 1 823 pacjentów NEURO **1 215 ma więcej niż jeden rekord**, a u nich:

- mediana rozpiętości badań: **35 dni**
- w ciągu 30 dni mieści się **48,9%**
- w ciągu 90 dni mieści się **58,2%**

To wzorzec **pojedynczego epizodu hospitalizacji**, a nie wieloletniej obserwacji ambulatoryjnej.

**Dlaczego to groźne.** Chcemy modelu, który spojrzy na morfologię osoby z przychodni i powie: „ten profil wygląda na podwyższone ryzyko". Żeby się tego nauczył, wyniki chorych musiałyby pochodzić sprzed rozpoznania.

Jeśli pochodzą z hospitalizacji, w trakcie której pacjenta diagnozowano i operowano, model nauczy się czegoś innego: rozpoznawać **osobę, która właśnie leży na oddziale neurologicznym**. Po zabiegu, po podaniu kontrastu, w stresie okołooperacyjnym, z podwyższonymi markerami zapalnymi. Taki model na danych przesiewowych z przychodni nie zadziała, bo tam nikt nie ma profilu pooperacyjnego.

**Założenie przyjęte przez zespół (20.09.2026):** wyniki NEURO pochodzą z hospitalizacji, w trakcie której rozpoznano i leczono tętniaka.

**Co to zmienia:** nic w kodzie. Protokół, foldy, metryki zostają te same. Zmienia się **to, co wolno napisać w raporcie**. Nie piszemy „model przesiewowy ryzyka", tylko uczciwie: model rozpoznający profil pacjenta hospitalizowanego z rozpoznanym tętniakiem. Ta różnica jest fundamentalna dla wniosków i musi być w raporcie widoczna.

**Status: ZAŁOŻENIE PRZYJĘTE, wymaga potwierdzenia u prowadzącego lub właściciela danych.**

---

### Problem 3 (0.1): 63 pacjentów jest w obu kohortach naraz

**Co widzimy.** 63 pacjentów ma rekordy zarówno w KOR, jak i w NEURO — łącznie 264 rekordy (145 KOR + 119 NEURO). Relacja czasowa jest wymowna:

| Układ | Pacjentów |
|---|---:|
| wszystkie rekordy KOR **przed** NEURO | **55** |
| wszystkie NEURO przed KOR | 6 |
| przeplatane w czasie | 2 |

Dla tych 55 osób odstęp między ostatnim badaniem w KOR a pierwszym w NEURO wynosi **medianę 546 dni** (od 35 do 1 365).

**Dlaczego to problem.** Grupujemy dane po pacjencie, więc jedna osoba nie może być jednocześnie przypadkiem pozytywnym i nieoznaczonym. Trzeba wybrać.

**Czym to NIE jest.** Początkowo uznałem te 55 osób za cenne znalezisko — pomiary sprzed rozpoznania u ludzi faktycznie chorych, czyli idealny materiał walidacyjny. **To była nadinterpretacja.** Bez daty diagnozy wiemy tylko, że pacjent pojawił się najpierw w jednym źródle, a potem w drugim. Tętniak mógł być rozpoznany dużo wcześniej, a pobyt na neurologii nastąpić dopiero po latach. Kolejność źródeł nie jest dowodem kolejności zdarzeń medycznych.

**Warianty:** nadać status pozytywny z osobną flagą pochodzenia rekordu, wyłączyć jako niejednoznacznych, albo analizować osobno. Pozostałych 8 przypadków (6 + 2) trzeba obejrzeć ręcznie.

**Skala wpływu jest mała** — 63 osoby na 40 924, czyli 0,15%. Decyzja musi jednak zapaść świadomie i zostać opisana.

**Status: DO DECYZJI ZESPOŁU.**

---

### Problem 4 (0.4): Selekcja cech użyła etykiety na całym zbiorze

**Co widzimy.** CRP, MONO i %MONO usunięto na podstawie korelacji z `label`, liczonej na skonsolidowanym zbiorze — czyli także na danych, które później trafią do walidacji. Formalnie jest to wyciek informacji o etykiecie do etapu przygotowania danych.

**Jaka jest realna skala.** Sprawdziliśmy korelacje usuniętych cech:

| Cecha | Korelacja Spearmana z `label` | Braki |
|---|---:|---:|
| CRP | +0,0085 | 21,1% |
| MONO | −0,0183 | 3,7% |
| %MONO | +0,0146 | 3,7% |

Dla porównania najsilniejsze cechy w zbiorze: WPT 0,114, BAZO 0,091, K 0,089.

Czyli **odrzucono szum**, a nie silne predyktory. Wpływ na wyniki jest zapewne żaden.

**Ale jest drugi, poważniejszy powód, żeby to naprawić.** Kolumna `label` odróżnia KOR od NEURO, czyli **kohorty**, a nie chorych od zdrowych. Selekcja po korelacji z etykietą zatrzymuje więc cechy, które najlepiej rozdzielają kohorty — a jak pokazuje Problem 1, kohorty różnią się też epoką i sposobem pozyskania danych. Ta selekcja działa dokładnie w kierunku, którego nie chcemy.

**Rekomendacja: 38 cech jako wariant główny, 35 jako analiza wrażliwości.** Koszt zmiany to zero — plik `data/imputation-inputs/aneurysm_concatted.csv` (78 197 × 42) po usunięciu trzech kolumn jest **co do wartości identyczny** z obecnym plikiem produkcyjnym. Zweryfikowane.

**Status: REKOMENDACJA GOTOWA, czeka na zatwierdzenie.**

---

### Problem 5 (0.5): Wartości niemożliwe fizjologicznie i mieszane jednostki

**Co widzimy.**

| Cecha | Braki | Mediana | p99 | Maksimum | Wartości nierealne |
|---|---:|---:|---:|---:|---|
| WBC | 2 648 | 8,75 | 25,43 | 253,06 | 7 wartości > 200 |
| Na | 12 514 | 139,67 | 149,00 | 177,25 | 8 wartości < 100 (min 74) |
| K | 12 747 | 4,30 | 6,00 | 28,32 | **86 wartości > 9** u 78 pacjentów |
| KREA | 13 643 | 0,87 | 19,56 | 196,15 | patrz niżej |

Potas powyżej 9 mmol/l to stan bezpośredniego zagrożenia życia, a wartości rzędu 26–28 są **fizjologicznie niemożliwe**. To niemal na pewno błędy wpisu albo hemoliza próbki.

**Osobna sprawa: kreatynina ma prawdopodobnie pomieszane jednostki.** Mediana 0,87 wskazuje na mg/dl, ale:

| Próg | Liczba wartości | Udział |
|---|---:|---:|
| > 15 (wartość skrajna dla mg/dl) | 767 | 1,19% |
| > 50 (poza zakresem mg/dl, możliwe µmol/l) | 210 | 0,33% |

Problem dotyczy około 1% rekordów i występuje w obu kohortach w zbliżonej proporcji (1,00% w KOR, 0,78% w NEURO).

**Dlaczego nie da się tego naprawić samemu.** Kusi, żeby podzielić wszystko powyżej progu przez 88,4 (przelicznik µmol/l na mg/dl). Ale kreatynina 200 µmol/l i kreatynina 200 mg/dl to dwie różne historie — pierwsza to niewydolność nerek, druga to błąd. Automatyczna korekta **skasowałaby prawdziwe przypadki ciężkiej niewydolności**, czyli akurat tych pacjentów, którzy są klinicznie najciekawsi.

Potrzebna jest informacja, czy w systemie źródłowym zapisana jest jednostka konkretnego oznaczenia.

**Status: DO KONSULTACJI KLINICZNEJ.** W kodzie jest już przełącznik zamieniający wartości spoza zakresu przeżycia na braki (i pozwalający imputacji je odtworzyć), domyślnie wyłączony.

---

### Problem 6 (0.3): Wybór reguły agregacji realnie zmienia dane

**Co widzimy.** 41,3% pacjentów ma więcej niż jeden rekord, więc trzeba z kilku wyników zrobić jeden profil. Ale punktem wyjścia jest nierównowaga, która przesądza sprawę:

| Klasa | Pacjentów | Mediana rekordów | Średnia | Maksimum | Ma >1 rekord |
|---|---:|---:|---:|---:|---:|
| KOR | 39 164 | 1 | 1,83 | 11 | 40% |
| NEURO | 1 823 | 2 | **3,65** | 38 | 67% |

**Chorzy są badani dwa razy częściej niż nieoznaczeni** — bo leżą w szpitalu.

**Dlaczego to groźne.** Jeśli wybierzemy regułę wrażliwą na liczbę pomiarów, model dostanie sygnał pochodzący z **częstości badania**, a nie ze stanu zdrowia. Maksimum z dziesięciu pomiarów jest statystycznie wyższe niż maksimum z jednego, niezależnie od tego, czy pacjent jest chory.

Sprawdziliśmy to wewnątrz samego KOR, żeby efekt choroby nie mieszał się do wyniku — korelacja liczby rekordów z wartością cechy:

| Reguła | Mediana \|ρ\| | Maks. \|ρ\| | Cech z \|ρ\| > 0,2 |
|---|---:|---:|---:|
| **mediana** | 0,079 | 0,257 | **2** |
| ostatni | 0,083 | 0,296 | 1 |
| średnia | 0,125 | 0,284 | 6 |
| **maksimum** | 0,190 | 0,379 | **17** |

Maksimum przemyca liczbę badań do **17 cech na 35**.

Do tego wciąga wartości nierealne z Problemu 5 — liczba pacjentów poza zakresem przeżycia:

| Cecha | Mediana | Średnia | Ostatni | Maksimum |
|---|---:|---:|---:|---:|
| K | 20 | 24 | 31 | **78** |
| WBC | 1 | 1 | 2 | **7** |

A pozorna siła sygnału jest przy tym praktycznie identyczna (mediana \|AUC−0,5\|: 0,0440 dla średniej, 0,0438 dla mediany, 0,0467 dla maksimum). **Maksimum nie kupuje nic w zamian za te wady.**

**Rekomendacja: mediana jako reguła główna, średnia jako analiza wrażliwości.**

Maksimum odpada całkowicie, nie jako wariant zapasowy. Ostatni pomiar też odpada, ale z innego powodu: przy założeniu o hospitalizacji „ostatni" oznacza u chorych pomiar **po leczeniu**, a u KOR zwykły wynik kontrolny. Reguła, która znaczy co innego w każdej klasie, jest bezużyteczna.

**Status: REKOMENDACJA GOTOWA, czeka na zatwierdzenie.**

---

### Problem 7 (0.6): Podział danych nie był tasowany

**Co widzimy.** W `3-sgkf-split/aneurysm_sgkf_mice_pipeline.py` podział powstaje jako `StratifiedGroupKFold(n_splits=5)`. Domyślnie `shuffle=False`, więc **nie ma tasowania**, a stała `RANDOM_STATE = 42` nie ma na podział żadnego wpływu — działa wyłącznie wewnątrz MICE.

Plik danych jest posortowany po etykiecie: pierwsze 71 546 wierszy to KOR, potem NEURO.

**Czy to zaszkodziło.** Sprawdziliśmy — **nie**. Mediana roku badania w grupie NEURO wynosi 2017 w każdym z pięciu foldów, a udział NEURO sprzed 2019 mieści się w przedziale 0,56–0,61. Stratyfikacja wystarcza, żeby wyrównać foldy mimo braku tasowania.

**Dlaczego mimo to naprawiamy.** Trzy powody:

1. **Kruchość.** Ktokolwiek przesortuje plik albo dopisze wiersze, dostanie inny podział i nawet tego nie zauważy.
2. **Raport wprowadza w błąd.** Napisane jest „ziarno 42", a podział jest w rzeczywistości powtarzalny z kolejności pliku, nie z ziarna.
3. **Nie da się zmierzyć rozrzutu.** Bez działającego ziarna nie powtórzymy walidacji na kilku podziałach, żeby pokazać, że wynik nie jest dziełem przypadku.

**Wyniki siatki stabilności** (2 warianty etykiety × `n_splits` 3/5/10 × shuffle × 5 ziaren, na dwóch poziomach podziału):

- Rozstęp udziału klasy pozytywnej między foldami: maksymalnie **0,25 punktu procentowego**.
- Rozstęp rozmiaru foldu: **0,49%**.
- Zero wspólnych pacjentów między train i test we wszystkich przebiegach.
- Rozkład lat: udział rekordów z 2020–2021 różni się między foldami najwyżej o **2,6 pp**.
- Podział **po pacjentach jest wyraźnie równiejszy** niż po rekordach: przy 5 foldach rozstęp 0,0001 wobec 0,0009, przy 10 foldach 0,0002 wobec 0,0025.

**Najważniejsze odkrycie:** włączenie `shuffle` nie poprawia balansu, ale **całkowicie zmienia skład foldów**. Adjusted Rand Index między podziałem bez tasowania a podziałami z ziarnami wynosi 0,00, i tyle samo między dowolnymi dwoma ziarnami. Każde ziarno daje faktycznie inny podział — więc konkretne ziarno trzeba wybrać, zapisać i już nie ruszać.

**Rekomendacja:** podział po pacjentach, `shuffle=True`, 5 foldów zewnętrznych, ziarno 42, wszystko zapisane w konfiguracji.

**Status: LICZBY GOTOWE, wybór do zamrożenia.**

---

### Problem 8 (przekrojowy): Założenie SCAR jest wątpliwe

**O co chodzi.** Metody PU zwykle zakładają **SCAR** (Selected Completely At Random): że znani pozytywni są losową próbką wszystkich pozytywnych. Czyli że pacjenci z NEURO niczym się nie różnią od niezdiagnozowanych chorych ukrytych w KOR — poza tym, że akurat ich wykryto.

**Dlaczego to u nas nie jest prawda.** NEURO to kohorta neurologiczna. Ci ludzie trafili na oddział, bo mieli objawy — ból głowy, krwawienie, deficyt neurologiczny. Niewykryty tętniak w populacji ogólnej jest z definicji bezobjawowy. To nie są porównywalne grupy.

**Konsekwencja.** Model uczy się odróżniać chorych **objawowych** od reszty. Przeniesienie tego na przesiew osób bezobjawowych jest ekstrapolacją, której nasze dane nie potwierdzają.

**Status: OGRANICZENIE DO OPISANIA W RAPORCIE.** Nie da się go naprawić, da się go tylko uczciwie nazwać.

---

### Problem 9 (przekrojowy): Nie zmierzymy rzeczywistego odsetka pomyłek

**O co chodzi.** Ukrywanie pozytywnych mierzy, czy model potrafi odzyskać **znanych** chorych. Nie mówi nic o tym, ilu pacjentów KOR jest naprawdę zdrowych.

**Konsekwencja praktyczna.** Nie możemy powiedzieć „model myli się w X% przypadków". Możemy powiedzieć „model odzyskuje X% ukrytych chorych w puli Y pacjentów, których da się skierować na badanie". To jest uczciwa miara i to właśnie raportujemy.

**To dlatego odrzuciliśmy metrykę ważoną** z ręczną karą za false positives, o której była mowa na spotkaniu. Nie da się ustalić kary za pomyłkę, której nie potrafimy rozpoznać.

**Status: OGRANICZENIE WBUDOWANE W PROBLEM.**

---

## Część IV — Co zbudowaliśmy: punkty 1–5

Implementacja jest gotowa, przetestowana i uruchomiona na prawdziwych danych. Branch `pu-pipeline-1-5-lk`, 750 linii kodu plus 240 linii testów. Szczegóły techniczne w `4-pu-setup/DOKUMENTACJA_PIPELINE.md`.

**Żaden model jeszcze się nie uczy** — to wciąż infrastruktura. Pierwszy model powstaje w punkcie 6.

### Punkt 1: foldy i imputacja wewnątrz foldu

Kolejność nienaruszalna: **podział → kontrola kohort → scaler → MICE → agregacja**. Wszystko po słowie „podział" uczy się wyłącznie na treningu.

Nowość w stosunku do poprzedniej wersji: **walidacja zagnieżdżona**. Tych samych foldów zewnętrznych nie wolno użyć naraz do strojenia i do raportowania wyniku, bo to zawyża jakość. Foldy zewnętrzne służą do oceny, wewnętrzne do strojenia, a preprocessing dopasowuje się od nowa w każdym foldzie wewnętrznym.

Jest też **puste gniazdo `hak_kontroli_kohort`** — jeśli zespół wybierze wariant C dla Problemu 1, wchodzi dokładnie tam, bez przebudowy pipeline'u.

### Punkt 2: ukrywanie pozytywnych

Losujemy **całych pacjentów**, nigdy pojedynczych rekordów. Trzymamy trzy kolumny: `true_label` (prawda), `observed_label` (to, co widzi model), `is_hidden` (flaga kontrolna). Każda maska ma zapisany udział i ziarno, więc jest w pełni odtwarzalna. Wygenerowane: 3 udziały (20%, 40%, 60%) × 5 ziaren = 15 kombinacji.

### Punkt 3: terminologia

Bez kodu — to ustalenie językowe. Wynik modelu nazywamy **risk score**, nigdy „prawdopodobieństwem tętniaka", dopóki nie jest skalibrowany. Metryki liczone z KOR jako klasą negatywną to **metryki P-vs-U**, a nie skuteczność wykrywania choroby.

### Punkt 4: metryki pacjentowe

Wszystkie metryki działają na pacjentach, nie na rekordach — pacjent z 38 pomiarami nie może ważyć 38 razy więcej niż pacjent z jednym.

| Metryka | Co mierzy |
|---|---|
| **RecallHidden@q** | **główna** — ilu ukrytych chorych odzyskujemy w puli do skierowania |
| Udział ukrytych w top-k | uzupełnienie; **nie jest** estymatorem precision |
| Lift@q | ile razy lepiej od losowania |
| RecallKnown | czy nie gubimy oczywistych przypadków |
| ROC-AUC / PR-AUC P-vs-U | pomocnicze, z zastrzeżeniem |
| ROC-AUC / PR-AUC kontrolowana | ukryci vs reszta puli nieoznaczonej |

### Punkt 5: funkcja celu

```
S(q) = alfa · RecallHidden@q + (1 − alfa) · RecallKnown
```

`q` to odsetek pacjentów, których realnie da się skierować na diagnostykę. **Nie jest hiperparametrem** — strojenie `q` oznaczałoby dobieranie definicji sukcesu pod wynik. Musi zostać zamrożone przed treningiem, najlepiej na podstawie rzeczywistej przepustowości badań obrazowych.

### Co zostało sprawdzone

**17 testów, wszystkie przechodzą.** Najważniejsze: fold wewnętrzny nigdy nie sięga po pacjenta z outer-test, metryka daje ten sam wynik przy przetasowanych wierszach, fold bez ukrytych zwraca brak wartości zamiast zera.

**Przebieg kontrolny na 3 000 pacjentach** z prawdziwym MICE w każdym foldzie i **losowym** scoringiem dał funkcję celu 0,079 przy oczekiwanych 0,05 — mieści się w szumie przy 8–14 ukrytych na fold. Metryki są podpięte poprawnie.

---

## Część V — Co jest osiągalne

Liczby dla scenariusza rekomendowanego (38 cech, mediana, 5 foldów, 40% ukrywania, `q = 5%`):

| Wielkość | Na fold testowy |
|---|---:|
| pacjentów | 8 185 |
| pozytywnych (prawdziwie) | ~365 |
| **ukrytych** | **~146** |
| znanych pozytywnych | ~219 |
| pula rankingowa U | ~7 966 |
| **pacjentów skierowanych przy `q = 5%`** | **~399** |

Model ma wskazać 399 osób, wśród których ukrytych chorych jest 146. Maksymalny możliwy `RecallHidden@q` to 1,0 — zadanie jest wykonalne. Rozdzielczość metryki: 1/146 ≈ 0,7 punktu procentowego.

Rozkład pozytywnych między foldami: 0,0445 / 0,0446 / 0,0446 / 0,0446 / 0,0445. Równo.

---

## Część VI — Decyzje do podjęcia

| Nr | Decyzja | Rekomendacja | Kto decyduje | Koszt zmiany później |
|---|---|---|---|---|
| 0.1 | status 63 pacjentów | pozytywni z flagą | zespół | niski |
| 0.2a | kontrola czasu | wariant C + A jako wrażliwość | zespół + prowadzący | **średni** |
| 0.2b | pochodzenie rekordów NEURO | założenie: hospitalizacja | potwierdzenie kliniczne | tylko wnioski |
| 0.3 | reguła agregacji | mediana | zespół | niski |
| 0.4 | zestaw cech | 38 cech | zespół | niski |
| 0.5 | wartości skrajne i KREA | czeka na jednostki | konsultacja kliniczna | niski |
| 0.6 | konfiguracja podziału | pacjenci, shuffle, ziarno 42, 5 foldów | zespół | niski |
| — | wartość `q` | z przepustowości diagnostyki | zespół + prowadzący | **zmienia definicję sukcesu** |

Koszt zmiany decyzji to w większości nie kod, lecz **czas obliczeń**: pełna imputacja to ~23 minuty na fold, a przy walidacji zagnieżdżonej wielokrotność tego. Dlatego decyzje powinny zapaść **przed** uruchomieniem pełnego treningu.

---

## Część VII — Co dalej

Po zamrożeniu decyzji wchodzimy w `03_PLAN_MODELOWANIE.md`:

- **Punkt 6:** strojenie hiperparametrów Optuną w walidacji zagnieżdżonej, z `RecallHidden@q` jako funkcją celu.
- **Punkt 7:** porównanie modeli — regresja logistyczna, Random Forest, jeden boosting (XGBoost albo LightGBM, nie oba) oraz **co najmniej jedna prawdziwa metoda PU**: PU Bagging i korekta Elkana-Noto.
- **Punkt 8:** analiza pacjentów wysokiego ryzyka, wyłącznie na predykcjach out-of-fold.
- **Punkt 9:** artefakty, model card, pełna konfiguracja i wersje bibliotek.

Do środowiska trzeba będzie doinstalować bibliotekę boostingową — obecnie nie ma ani `xgboost`, ani `lightgbm`.

---

## Część VIII — Uwagi prowadzącego: co z nimi zrobiliśmy

| Uwaga ze spotkania | Gdzie trafiła |
|---|---|
| Uzupełnianie braków per fold | punkt 1, rozszerzone o walidację zagnieżdżoną |
| Ważna metryka do optymalizacji, bo brak grupy kontrolnej | punkt 5 |
| Losujemy chorych, wstawiamy do KOR i ukrywamy etykietę | punkt 2, kolumny `true_label` / `observed_label` / `is_hidden` |
| Wśród false positives będą prawdziwe jedynki | punkt 3, terminologia |
| Optuna do strojenia pod odzyskiwanie ukrytych | punkt 6 |
| False positives idą na badania obrazowe | definiuje `q` — ilu pacjentów da się realnie przebadać |

**Dwa odstępstwa, które wymagają omówienia:**

1. **Metryka ważona z ręczną karą za FP — odrzucona w tej formie.** Powód w Problemie 9: nie da się ustalić kary za pomyłkę, której nie potrafimy rozpoznać. Idea ważenia przetrwała jako parametr `alfa` w funkcji celu, ale z jawnie zadeklarowaną wagą zamiast ukrytej intuicji.
2. **Accuracy odpada, ROC-AUC schodzi do roli pomocniczej.** Przy 4,45% pozytywnych accuracy jest myląca, a ROC-AUC liczone z KOR jako klasą negatywną mierzy separację pozytywnych od nieoznaczonych — nie wykrywanie choroby.

Oba odstępstwa wynikają z recenzji metodologicznej, nie z preferencji.

---

## Podsumowanie w pięciu zdaniach

1. Dane są przygotowane, opisane i zweryfikowane: 78 197 rekordów, 40 924 pacjentów, 4,45% z potwierdzonym tętniakiem.
2. Infrastruktura do modelowania PU jest zbudowana i przetestowana — foldy z imputacją wewnątrz foldu, walidacja zagnieżdżona, ukrywanie pozytywnych, metryki pacjentowe i funkcja celu.
3. Etap 0 wykrył dziewięć problemów, z których **najpoważniejszy jest rozjazd czasowy kohort** — model może rozpoznawać epokę zamiast choroby.
4. Cztery problemy mają gotowe rekomendacje oparte na liczbach, trzy wymagają decyzji zespołu lub konsultacji klinicznej, dwa są nieusuwalne i idą do ograniczeń raportu.
5. Nic nas nie blokuje technicznie — wszystkie decyzje są parametrami konfiguracji, więc ich zmiana kosztuje czas obliczeń, a nie przepisywanie kodu.
