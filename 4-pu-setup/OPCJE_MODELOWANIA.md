# Opcje modelowania po etapie 3: mapa możliwości i plan dalszych prac

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Status:** materiał do dyskusji w zespole i z prowadzącym. Dokument opisuje przestrzeń możliwych rozwiązań i sposób wyboru między nimi. Nie zatwierdza żadnej opcji i nie zawiera wyników modeli.
**Stan na:** 24.10.2026 (branch `pu-modeling-setup-lk`).
**Podstawa:** wyniki etapów 1–3 (`1026_REALIZACJA_DO_SGKF.md`, `3-sgkf-split/RAPORT_SGKF_MICE.md`, pliki w `3-sgkf-split/sgkf/results/`). Nowe liczby o wykonalności opcji pochodzą z `4-pu-setup/opcje_fakty.py` (wynik `4-pu-setup/results/opcje_fakty.json`, log `4-pu-setup/results/logi/opcje_fakty.log`). Skrypt nie trenuje modeli i nie zmienia danych.

**Relacja do `PLAN_MODELOWANIA.md`.** Plan modelowania rozpisuje szczegółowo jedną spójną ścieżkę: profil pacjenta z imputacji MICE i mediany, regresja logistyczna, PU Bagging i model drzewiasty, ocena przez odzyskiwanie ukrytych chorych obok modeli kontrolnych. Tutaj ta ścieżka jest jedną z kilku (S1 w sekcji 11). Ten dokument niczego w planie nie zmienia. Zasady wspólne dla wszystkich opcji obowiązują bez zmian:

- foldy zewnętrzne wyłącznie z `podzial.wczytaj_podzial()`;
- każde dopasowanie (scaler, imputer, selekcja cech, model) tylko na treningu;
- maski ukrywania na poziomie pacjentów, identyczne dla porównywanych modeli;
- modele kontrolne na tych samych maskach.

**Jak czytać:**

- 5 minut: sekcja 0 i mapa decyzji w sekcji 2.
- Najważniejsze dla sensu wyniku: sekcja 3 (jakie pytanie zadajemy), sekcja 7 (co robimy ze skrótami) i sekcja 10 (jak oceniamy).
- Przegląd metod: sekcje 4–9. Każda opcja ma identyfikator (np. C4), używany w ścieżkach i w planie.
- Plan pracy: sekcje 11–13.

Podstawowe pojęcia (PU, SCAR, OOF, `RecallHidden@q`, skrót) objaśnia słownik w `PLAN_MODELOWANIA.md`. Nowe pojęcia z tego dokumentu:

| Pojęcie | Znaczenie |
|---|---|
| π, class prior | Rzeczywisty udział chorych w puli U. Nie znamy go; 4,45% oznaczonych w danych nie jest jego estymatą |
| SAR (*selected at random*) | Założenie słabsze niż SCAR: prawdopodobieństwo, że chory zostanie oznaczony, zależy od jego cech, np. od okresu lub zestawu zleconych badań |
| Blok braków | Grupa cech, których brakuje zwykle razem, np. cały koagulogram. 8 bloków wyznaczono z samej macierzy braków, bez etykiety (`opcje_fakty.py`) |
| Wzorzec bloków | Ciąg 8 znaczników „blok zmierzony / niezmierzony” dla pacjenta. Opisuje, jakie badania zlecano |
| Kanał skrótu | Konkretne źródło różnic między kohortami niezwiązane z chorobą: okres, wzorzec zleceń, ślad przetwarzania, liczba rekordów |
| Ocena warunkowa | Miara liczona w warstwach o wyrównanym kanale skrótu, np. ukryci chorzy porównywani tylko z pacjentami KOR o tym samym wzorcu bloków |
| Benchmark półsyntetyczny | Zadanie na prawdziwych pacjentach KOR ze sztuczną „chorobą” o znanej regule. Pozwala ocenić metody tam, gdzie znamy prawdziwy status każdego pacjenta |
| Neutralizacja | Kanał skrótu wchodzi do modelu w treningu, ale przy ocenie ustawiamy go wszystkim na tę samą wartość, żeby nie wpływał na ranking |

---

## 0. W skrócie

1. **Uczymy się na znanych chorych z jednego źródła i nieoznaczonych pacjentach z innego.** Kohorty różnią się nie tylko chorobą. Różnią się okresem (NEURO 2000–2024, KOR praktycznie tylko 2020–2021), zestawem zleconych badań, liczbą rekordów i śladami przetwarzania. Etap 3 to zmierzył: 38 cech rozdziela kohorty z AUC 0,93, a sam wzorzec braków z AUC 0,82. Sam wzorzec braków odzyskuje też około 52% ukrytych chorych w górnych 5% rankingu.
2. **Test z ukrywaniem z definicji nagradza rozpoznawanie kohorty.** Ukryci są losową częścią NEURO. Najlepszym możliwym rankingiem w tym teście jest więc prawdopodobieństwo przynależności do kohorty NEURO, łącznie ze wszystkimi skrótami (sekcja 10.1). Model, który ograniczy korzystanie ze skrótów, dostanie w tym teście *niższy* wynik, nawet jeśli jest bliżej sygnału choroby. Wybór modelu wyłącznie według `RecallHidden@q` systematycznie faworyzuje skróty, dlatego ocena wymaga dodatkowych schematów (sekcja 10).
3. **Algorytm jest decyzją drugorzędną.** Gdyby oznaczeni chorzy byli losową próbą chorych z tej samej populacji co pula U (założenie SCAR), zwykły klasyfikator P-vs-U porządkowałby pacjentów tak samo jak idealny klasyfikator chory-vs-zdrowy. Metody PU zmieniają przy tym głównie skalę wyniku, próg i odporność elastycznych modeli na zanieczyszczoną pulę U, a nie optymalny ranking. Większy wpływ na sens wyniku mają: ujęcie zadania (oś A), reprezentacja pacjenta (B), braki (C), cechy (D) i sposób traktowania skrótów (E).
4. **Dane pozwalają na więcej opcji, niż zakłada obecny plan.** Przykłady z `opcje_fakty.py`:
   - profil z samej morfologii, wieku i płci nie wymaga imputacji i obejmuje 99% pacjentów w obu kohortach;
   - wzorzec zleconych badań 85,5% chorych występuje także u co najmniej 50 pacjentów KOR, więc można go wyrównać ważeniem lub dopasowaniem;
   - 16 z 36 cech laboratoryjnych to przeliczenia innych cech, a przeliczenie PT na INR różni się między okresami.
5. **Proponowany plan jest etapowy** (sekcja 12):
   1. wspólna infrastruktura, niezależna od metody;
   2. szeroki przegląd wielu opcji na stałych ustawieniach — pytanie brzmi „które decyzje zmieniają wynik”, a nie „który model wygrywa”;
   3. bramka decyzyjna;
   4. krótka lista i strojenie;
   5. analizy odporności i raport.
6. **Wynik negatywny też jest wynikiem.** Może się okazać, że żaden wariant nie przewyższa kontroli braków w ocenie, która wyklucza znane skróty. Poprawny wniosek brzmi wtedy: w tych danych sygnał laboratoryjny nie daje się oddzielić od pochodzenia kohort. To odpowiedź na pytanie badawcze, a nie porażka projektu.

---

## 1. Punkt wyjścia: co z etapów 1–3 ogranicza wybór

| # | Fakt | Co z tego wynika dla opcji |
|---|---|---|
| 1 | 40 924 pacjentów, w tym 1 823 znanych chorych (4,45%), 38 cech, 78 052 rekordy tygodniowe. 5 zamrożonych foldów po ~8 185 pacjentów i ~365 chorych (`1026_SPOTKANIE_PROWADZACY.md`; `PLAN_MODELOWANIA.md`, sekcja 5) | To dane tabelaryczne średniej wielkości, więc każdy klasyczny algorytm jest obliczeniowo wykonalny. Ograniczeniem jest liczba chorych: przy 40% ukrywania wypada ~146 ukrytych na fold, a rozdzielczość `RecallHidden@q` to ~0,7 pp. Opcje zmniejszające liczbę chorych (okno dat, kompletne panele) od razu mają gorszą rozdzielczość |
| 2 | Jeden rekord ma 60,0% pacjentów KOR i 33,4% NEURO. Cztery i więcej rekordów: 9,6% wobec 27,0%. Maksimum: 11 wobec 38 (`opcje_fakty`, A) | Sama liczba rekordów odróżnia kohorty. Każda reprezentacja wrażliwa na liczbę pomiarów wymaga kontroli: maksimum, cechy przebiegu, trening na rekordach bez wag. Modele sekwencyjne mają mało materiału, bo większość pacjentów KOR ma jeden tydzień badań |
| 3 | Brakujące komórki 36 cech laboratoryjnych (KOR / NEURO): wszystkie rekordy 13,0% / 27,5%; pierwszy rekord pacjenta 10,7% / 16,3%; mediana pacjenta przed imputacją 8,7% / 8,9% (`opcje_fakty`, B) | Asymetria braków zależy od reprezentacji. Przy agregacji przed imputacją ilość braków prawie się wyrównuje. Nie wyrównuje się jednak ich wzorzec (wiersz 4), a kompletność profilu zaczyna zależeć od liczby rekordów |
| 4 | Braki występują blokami. Na poziomie pacjenta 85,5% chorych ma wzorzec bloków obecny u co najmniej 50 pacjentów KOR, a 12,3% (225 osób) ma wzorzec, którego nie ma żaden pacjent KOR. Przykład: rozmaz rozszerzony (NEUT, NRBC, IG) nigdy niezmierzony przy pozostałych blokach zmierzonych — 91 chorych, 0 pacjentów KOR. Ważenie KOR do rozkładu wzorców NEURO zostawia efektywną liczebność 17 556 z 39 101 (`opcje_fakty`, C) | Wzorzec zleconych badań da się wyrównać (ważenie, dopasowanie, ocena w warstwach) dla większości chorych (E2, H2). Dla ~12% chorych wzorca nie da się oddzielić od kohorty, więc trzeba ich opisać osobno |
| 5 | KOR obejmuje tylko okno 2019-12-29 – 2021-12-26. Pierwszy rekord NEURO: przed oknem 1 048 pacjentów, w oknie 134, po oknie 641. Co najmniej jeden rekord w oknie ma 271 chorych (`opcje_fakty`, E; `1026_REALIZACJA_DO_SGKF.md`, sekcja 6.1) | Restrykcja do okna nadaje się na analizę wrażliwości (~22 ukrytych na fold), a nie na scenariusz główny. Ważenie względem okresu jest niemożliwe, bo poza oknem nie ma pacjentów KOR. Możliwa jest natomiast ocena w warstwach okresu chorych (H6) |
| 6 | Epokę widać w wynikach samych chorych: AUC 0,83 po imputacji. Na tych samych rekordach wzorzec braków daje 0,77–0,78, zmierzone wartości panelu 0,57–0,65, a 38 cech z NaN 0,84–0,87 (`RAPORT_SGKF_MICE.md`, sekcja 5.2) | Okres siedzi przede wszystkim w tym, które badania zlecano, ale również w wartościach. Opcje „tylko zmierzone wartości wspólnego panelu” ograniczają oba źródła, lecz ich nie usuwają |
| 7 | Przeliczenie PT na INR i WPT zależy od okresu. Przy PT 11,5–12,5 s mediana INR wynosi 1,00 w KOR i 0,99 u chorych z okna (tylko 48 rekordów), a 1,04 i 1,07 u chorych sprzed okna i po nim. Mediana WPT: 96 i 95 wobec 99 (`opcje_fakty`, G) | Para cech wyliczanych z tego samego pomiaru może zdradzać laboratorium lub okres; prawdopodobną przyczyną jest zmiana odczynnika. Elastyczny model, który widzi jednocześnie PT i INR, może z tego korzystać. To argument za zestawem bez cech pochodnych (D3) |
| 8 | 16 z 36 cech laboratoryjnych to przeliczenia innych (`opcje_fakty`, G):<br>• MCV, MCH, MCHC z HGB, HCT i RBC (mediana błędu 0,03–0,07%);<br>• wartości bezwzględne rozmazu z WBC i procentów;<br>• %MONO z pozostałych procentów;<br>• eGFR z KREA, wieku i płci (wzór MDRD odtwarza zapisane wartości z medianą błędu 0,93% poniżej progu cenzurowania);<br>• WAPTT = APTT / 33,3;<br>• INR i WPT z PT.<br>eGFR-MDRD wynosi dokładnie 60 w 69,3% pomiarów, eGFRCKD 90 w 40,5% | Niezależnych pomiarów laboratoryjnych jest około 20, plus wiek i płeć. Ma to znaczenie dla modeli liniowych (współliniowość), dla interpretacji ważności (rozkłada się na cechy pochodne) i dla doboru cech |
| 9 | Przy brakach całymi panelami MICE jest niewiele lepsze od średniej: błąd to 98% błędu średniej w KOR i 89% w NEURO. Uzupełnione NRBC nigdy nie jest zerem i przenosi informację o braku badania (AUC 0,73, jak sam wskaźnik braku) (`RAPORT_SGKF_MICE.md`, sekcje 4.3 i 5.1) | Wartość dodana MICE dla rankingu jest niepewna. Prosta imputacja, wskaźniki braków, natywna obsługa NaN i modele bez imputacji są pełnoprawnymi opcjami, a nie tylko analizami wrażliwości |
| 10 | KOR ma profil szpitalny: CRP > 5 mg/l w 61% zmierzonych wyników. NEURO wygląda na hospitalizację: mediana rozpiętości badań 35 dni u 1 215 chorych z więcej niż jednym rekordem. Czyszczenie z etapu 1 ustawiano osobno dla kohort, m.in. limit historii 11 rekordów w KOR i 38 w NEURO (`1026_REALIZACJA_DO_SGKF.md`, sekcje 6.1–6.2; `RAPORT_SGKF_MICE.md`, sekcje 5.3 i 7) | Interpretacja każdego modelu zależy od odpowiedzi na pytania o pochodzenie danych (sekcja 13). Żadna opcja modelowania nie usuwa skutków wcześniejszego doboru kohort |

---

## 2. Mapa decyzji

Każdy model w tym projekcie to zestaw decyzji na ośmiu osiach. Osie można łączyć, choć nie każde połączenie ma sens (sekcja 11). Kolejność osi odpowiada mniej więcej ich wpływowi na to, *co* wynik oznacza.

| Oś | Pytanie | Opcje | Sekcja |
|---|---|---|---|
| A. Ujęcie zadania | Na jakie pytanie odpowiada model? | A1 ranking PU w dostępnych kohortach · A2 profil hospitalizacji z tętniakiem · A3 wcześniejsze wykrywanie · A4 rozdzielność kohort jako diagnostyka · A5 podobieństwo do znanych chorych · A6 podgrupy chorych bez nadzoru | 3 |
| B. Jednostka i reprezentacja | Czym jest jeden przykład uczący? | B1 profil pacjenta · B2 pierwszy rekord · B3 rekordy z wagami i agregacją predykcji · B4 cechy przebiegu · B5 uczenie wieloinstancyjne · B6 punkt odniesienia w czasie | 4 |
| C. Braki danych | Co robimy z brakującymi wynikami? | C1 MICE w foldzie · C2 prosta imputacja · C3 imputacja i wskaźniki braków · C4 natywna obsługa NaN · C5 tylko wartości zmierzone · C6 imputacja wielokrotna · C7 agregacja przed imputacją · C8 inne imputery z etapu 2 | 5 |
| D. Cechy | Które informacje i w jakiej postaci? | D1 38 cech · D2 bez NRBC i %NRBC · D3 tylko pomiary niezależne · D4 wspólny panel · D5 cechy pochodne kliniczne · D6 transformacje · D7 wybór cech w walidacji | 6 |
| E. Skróty | Co robimy z informacją niezwiązaną z chorobą? | E0 tylko mierzymy · E1 ograniczamy dane · E2 wyrównujemy · E3 modelujemy jawnie i neutralizujemy · E4 uczymy niezmienniczości · E5 modelujemy mechanizm oznaczania | 7 |
| F. Strategia PU | Jak traktujemy pulę U? | F1 naiwny P-vs-U · F2 ważenie kosztów · F3 bagging PU · F4 Elkan–Noto · F5 ryzyko nieobciążone lub nieujemne · F6 metody dwuetapowe · F7 PU poza SCAR · F8 estymacja class prior | 8 |
| G. Algorytm | Jaka rodzina modeli? | G1 regresja logistyczna · G2 modele addytywne · G3 lasy losowe · G4 boosting · G5 SVM · G6 podobieństwo i kNN · G7 sieci neuronowe · G8 zespoły | 9 |
| H. Ocena i walidacja | Jak sprawdzamy, co model umie? | H1 `RecallHidden@q` z kontrolami · H2 ocena warunkowa · H3 ukrywanie zależne od cech · H4 benchmark półsyntetyczny · H5 testy placebo · H6 warstwy okresu · H7 fold zablokowany · H8 walidacja zewnętrzna lub kliniczna · H9 interpretacja i stabilność | 10 |

---

## 3. Oś A: na jakie pytanie odpowiada model

To najważniejsza decyzja, bo zmienia znaczenie wszystkich pozostałych. Przykład: wzorzec zlecanych badań jest skrótem w zadaniu przesiewowym (A1, A3), ale może być częścią opisu w zadaniu dotyczącym hospitalizacji (A2).

| Opcja | Pytanie | Warunki | Co wolno potem twierdzić |
|---|---|---|---|
| **A1. Eksploracyjny ranking PU w dostępnych kohortach** | Czy da się uszeregować pacjentów tak, żeby celowo ukryci znani chorzy trafiali wysoko? Ile z tego zostaje po wykluczeniu znanych skrótów? | Wykonalne teraz; to założenie obecnego planu | „Model odzyskuje X% ukrytych znanych chorych w tym zbiorze, przewaga nad kontrolą braków wynosi Y”. Bez twierdzeń o wykrywaniu tętniaka |
| **A2. Profil hospitalizacji z tętniakiem** | Jak bardzo wyniki pacjenta przypominają wyniki pacjentów hospitalizowanych z tętniakiem? | Potwierdzenie, że NEURO to hospitalizacja, najlepiej z informacją, czy tętniaki były pęknięte; porównywalna pula hospitalizowanych | Opis profilu klinicznego, nie ryzyko przesiewowe. Zmiany typowe dla ostrej fazy, np. po krwotoku podpajęczynówkowym, mogą dominować sygnał |
| **A3. Wcześniejsze wykrywanie** | Czy wyniki sprzed rozpoznania pozwalają wskazać osoby z późniejszym rozpoznaniem? | Daty rozpoznania i rekordy sprzed niego; ustalony moment predykcji; tylko wyniki dostępne przed tym momentem | Jedyne ujęcie zbliżone do „oceny ryzyka”. Na obecnych danych niewykonalne |
| **A4. Rozdzielność kohort jako diagnostyka** | Co i jak silnie odróżnia kohorty? | Wykonalne teraz; częściowo zrobione w etapie 3 | Mapa skrótów i górna granica rozpoznawalności kohorty, a nie model choroby |
| **A5. Podobieństwo do znanych chorych** | Jak bardzo pacjent przypomina znanych chorych, bez traktowania U jako negatywów? | Wykonalne teraz | Ranking podobieństwa. Ukryci chorzy są podobni do widocznych także przez skróty, więc to ujęcie nie usuwa problemu |
| **A6. Podgrupy chorych bez nadzoru** | Czy wśród chorych są wyraźne podgrupy profili, np. według okresu, ostrego stanu albo zestawu badań? | Wykonalne teraz, bez etykiety U | Materiał do decyzji w osiach A, D i E, np. czy dominuje podgrupa „ostra”. To nie jest model |

**Wniosek dla planu.** Bez nowych informacji wykonalne są A1, A4, A5 i A6, a ujęciem głównym pozostaje A1. A2 i A3 zależą od odpowiedzi na pytania z sekcji 13. A4 i A6 są tanie i zasilają decyzje w pozostałych osiach.

---

## 4. Oś B: jednostka i reprezentacja pacjenta

| Opcja | Na czym polega | Za | Przeciw / ryzyko | Koszt |
|---|---|---|---|---|
| **B1. Profil pacjenta** | Jeden wiersz na pacjenta. Cechy agregowane regułą: mediana, średnia, pierwsza lub ostatnia dostępna wartość, minimum, maksimum, liczba wyników poza zakresem | Każdy pacjent ma jeden głos, zgodnie z oceną na pacjentach. Mediana słabiej niż średnia i maksimum wiąże się z liczbą badań (`etap0_agregacja.py`) | Agregacja łączy różne fazy pobytu, np. przyjęcie i leczenie. Reguły typu maksimum przemycają liczbę badań | niski |
| **B2. Pierwszy rekord** | Jeden tydzień badań na osobę: najwcześniejszy | Stały punkt w czasie. Asymetria braków maleje: 10,7% / 16,3% zamiast 13,0% / 27,5%. Jeśli NEURO to hospitalizacja, pierwszy tydzień jest najbliżej przyjęcia, czyli przed leczeniem | Odrzuca informację z kolejnych rekordów. Pierwszy rekord nie oznacza „przed rozpoznaniem”. Wariant „losowy rekord” wymaga powtórzeń losowania | niski |
| **B3. Rekordy z wagami 1/n i agregacją predykcji** | Trening na tygodniach z wagą odwrotnie proporcjonalną do liczby rekordów pacjenta. Wynik pacjenta to agregat wyników jego rekordów | Więcej przykładów; widać zmienność wewnątrz pacjenta | Agregacja predykcji wchodzi w interakcję z liczbą rekordów: losowy wynik rekordów z medianą daje 3,2%, a losowy ranking pacjentów 4,2% przy q = 5% (oczekiwane 5%). Kontrole muszą być pacjentowe | niski |
| **B4. Cechy przebiegu** | Zmiana między pierwszym a ostatnim pomiarem, rozrzut, nachylenie, czas trwania serii | Mogą nieść informację o ostrym stanie lub o leczeniu | Istnieją tylko przy co najmniej dwóch rekordach (40% KOR, 67% NEURO), więc samo ich istnienie jest skrótem. Sens mają tylko w podgrupie o porównywalnej liczbie rekordów | średni |
| **B5. Uczenie wieloinstancyjne** | Pacjent to „worek” rekordów; model uczy się, które rekordy niosą sygnał | Naturalnie opisuje relację rekord–pacjent | Złożone przy małej liczbie rekordów na pacjenta. B3 jest jego prostym przybliżeniem | wysoki |
| **B6. Punkt odniesienia w czasie** | Profil tylko z badań sprzed ustalonego momentu, np. rozpoznania | Warunek konieczny dla A3 | Wymaga dat rozpoznania | zależy od danych |

**Do przeglądu szerokiego:** B1 z medianą jako punkt odniesienia, B2 i B3 jako warianty. B4 tylko w podgrupie pacjentów z co najmniej dwoma rekordami.

---

## 5. Oś C: braki danych i imputacja

| Opcja | Na czym polega | Za | Przeciw / ryzyko | Koszt |
|---|---|---|---|---|
| **C1. MICE w foldzie** (obecnie) | `IterativeImputer` z ExtraTrees (7 iteracji / 78 drzew / głębokość 10), dopasowany na treningu, deterministyczny | Sprawdzony w etapie 3. Zapisane foldy zewnętrzne można wykorzystać w przebiegu bez strojenia | Przy brakach panelowych działa prawie jak średnia; zostawia podpis NRBC. Trwa ~3 min na fold. Przy strojeniu trzeba go dopasować od nowa w każdym foldzie wewnętrznym | średni |
| **C2. Prosta imputacja** | Mediana lub średnia z treningu w miejsce braku | Bardzo tania; odpowiada na pytanie, czy MICE w ogóle pomaga rankingowi | W modelu liniowym stała wartość działa jak ukryty wskaźnik braku | niski |
| **C3. Imputacja i jawne wskaźniki braków** | Do cech dochodzi znacznik 0/1 „zmierzono” dla każdego bloku lub cechy | Skrót staje się jawny i mierzalny przez współczynniki i ważność. Umożliwia neutralizację przy ocenie (E3) | W literaturze predykcyjnej wskaźniki braków bywają zalecane, o ile braki powstają tak samo przy tworzeniu i przy stosowaniu modelu (Sperrin i in., 2020). U nas kodują jednak kohortę, więc podnoszą `RecallHidden@q` z niewłaściwego powodu | niski |
| **C4. Natywna obsługa NaN** | HistGradientBoosting, Random Forest (scikit-learn od 1.4), LightGBM i XGBoost same uczą się, w którą stronę kierować brak | Brak imputera i śladu imputacji; mniej kroków | Model uczy się wzorca braków wprost: w kontroli z 17.10, przeliczonej 31.10, 38 cech z NaN dało 69,2% przy q = 5%. Działa tylko z modelami drzewiastymi | niski |
| **C5. Tylko wartości zmierzone** | Profil z kompletnych pomiarów wspólnego panelu | Bez imputacji. Morfologia z wiekiem i płcią obejmuje 99,0% pacjentów w obu kohortach | Szersze panele wycinają pacjentów nierówno: z CRP zostaje 78,1% KOR i 60,3% NEURO. Kompletność też zależy od liczby rekordów | niski |
| **C6. Imputacja wielokrotna** | Kilka losowych uzupełnień (np. `IterativeImputer(sample_posterior=True)`), wyniki uśrednione | Pokazuje, jak bardzo ranking zależy od niepewności uzupełnień | Obecne MICE jest deterministyczne, więc potrzebny inny estymator. Koszt rośnie z liczbą uzupełnień | wysoki |
| **C7. Agregacja przed imputacją** | Najpierw mediana surowych wyników pacjenta, potem imputacja na profilach | Braki spadają do 8,7% / 8,9% komórek; taniej | Kompletność profilu zależy od liczby rekordów: więcej tygodni to więcej zmierzonych paneli. Benchmark etapu 2 był rekordowy, więc jakość imputacji profili wymaga osobnej oceny | niski |
| **C8. Inne imputery z etapu 2** | KNN i MissForest (`2-imputation/knn-am`, `2-imputation/missforest-dm`) uruchomione w foldach | Wykorzystuje istniejącą pracę zespołu. Benchmark etapu 2 mierzył jakość uzupełnień, a nie ich wpływ na ranking | Te same ograniczenia co C1 przy brakach panelowych | średni |

**Mechanizm braków.** W tych danych brak wyniku jest w dużej części decyzją kliniczną: badania nie zlecono. Każda opcja inaczej rozstrzyga, czy ta informacja trafia do modelu. C3 i C4 ją ujawniają, C1 i C2 częściowo ukrywają w wartościach, a C5 i E2 ją ograniczają. Wybór należy więc w równym stopniu do osi E, co do jakości imputacji.

**Do przeglądu szerokiego:** C2 jako tani punkt odniesienia, C1 z zapisanych foldów, C3, C4 (z modelami drzewiastymi), C5 i C7.

---

## 6. Oś D: zestaw i postać cech

| Opcja | Na czym polega | Za | Przeciw / ryzyko |
|---|---|---|---|
| **D1. 38 cech** | Pełne wejście etapu 3 | Punkt odniesienia zgodny z etapem 3 | Zawiera cechy pochodne i podpis NRBC |
| **D2. 36 cech bez NRBC i %NRBC** | Usunięcie obu cech przed MICE | Usuwa najwyraźniejszy ślad imputacji; cechy mają niską wartość kliniczną u dorosłych | Wymaga nowego dopasowania MICE (~15 min na 5 foldów) |
| **D3. Tylko pomiary niezależne** | Po jednej reprezentacji każdego pomiaru, razem 20 cech laboratoryjnych oraz wiek i płeć:<br>• morfologia: HGB, RBC, HCT, RDW, PLT, MPV, WBC;<br>• rozmaz: cztery z pięciu procentów, %IG, %NRBC;<br>• biochemia: KREA, Na, K, GLU, CRP;<br>• koagulologia: PT, APTT | Mniej współliniowości i czytelniejsza ważność. Usuwa ślady przeliczeń zależnych od laboratorium (INR, WPT; wiersz 7 w sekcji 1) | Wybór reprezentacji jest arbitralny, np. procent czy wartość bezwzględna, PT czy INR. Trzeba go zapisać przed porównaniem. eGFR niesie dodatkowo nieliniowe połączenie KREA z wiekiem i płcią |
| **D4. Wspólny panel** | Cechy mierzone prawie u wszystkich w obu kohortach: morfologia, wiek, płeć (99% pacjentów) | Minimalny ślad zleceń i imputacji | Mało informacji. Ewentualny sygnał w biochemii lub koagulogramie znika |
| **D5. Cechy pochodne kliniczne** | Ilorazy (np. neutrofile/limfocyty, płytki/limfocyty), liczba wyników poza zakresem referencyjnym | Modele liniowe nie zbudują ilorazów same; łatwo je opisać | Wskaźniki zapalne opisują też stan szpitalny, a KOR ma profil zapalny |
| **D6. Transformacje** | Logarytm dla rozkładów skośnych (CRP, GLU, KREA, WBC); przekształcenie log-ilorazowe procentów rozmazu, które sumują się do 100% (Aitchison, 1986); standaryzacja; harmonizacja wartości w obrębie okresu | Lepsze warunki dla modeli liniowych i odległościowych. Harmonizacja ogranicza dryf analityczny | Harmonizacja dotyczy praktycznie tylko NEURO i usuwa też prawdziwe różnice między okresami, więc nadaje się wyłącznie do analizy wrażliwości |
| **D7. Wybór cech w walidacji** | L1, selekcja stabilnościowa, ważność permutacyjna — wewnątrz foldów treningowych | Mniejszy, prostszy model | Etykieta oznacza kohortę, więc selekcja wybiera cechy odróżniające kohorty. To ta sama pułapka co w etapie 1 (`RAPORT_SGKF_MICE.md`, sekcja 2.6) |

**Do przeglądu szerokiego:** D1 jako odniesienie, D2, D3 i D4. D5–D7 dopiero po bramce 1, jeśli wynik okaże się zależny od cech.

---

## 7. Oś E: co robimy ze skrótami

Skrót to informacja odróżniająca kohorty z powodów niezwiązanych z chorobą. W naszych danych znamy cztery kanały:

- okres badań;
- zestaw zleconych badań, czyli wzorzec braków;
- ślady przetwarzania: uzupełnienia NRBC, przeliczenia PT→INR;
- liczbę rekordów.

Mogą istnieć kanały niemierzalne, np. typ oddziału.

**To, czy coś jest skrótem, zależy od ujęcia zadania** (oś A). Przykład: koagulogram zlecony przed zabiegiem jest skutkiem rozpoznania. W zadaniu przesiewowym (A1, A3) to informacja „z przyszłości”, a w opisie hospitalizacji (A2) część profilu. Usunięcie skrótu może więc usunąć także sygnał związany z chorobą. Każda z opcji E1–E5 zmienia pytanie i trzeba to w opisie wyniku nazwać.

| Opcja | Na czym polega | Wykonalność u nas | Za | Przeciw | Koszt |
|---|---|---|---|---|---|
| **E0. Tylko mierzymy** | Model bez zmian; obok niego kontrole (losowa, demografia, wzorzec braków) na tych samych maskach | tak, to obecny plan | Nie zmienia zadania; prosta interpretacja przewagi nad kontrolą | Model nadal korzysta ze skrótów. Przewaga nad kontrolą braków nie wyklucza innych kanałów | niski |
| **E1. Ograniczamy dane** | Wspólne okno dat, wspólny panel, pierwszy rekord | tak; w oknie 271 chorych | Usuwa kanał u źródła | Mniej danych; w oknie ~22 ukrytych na fold. Nie usuwa kanałów niemierzalnych | niski |
| **E2. Wyrównujemy** | Ważymy U tak, żeby rozkład wzorca bloków był taki jak u P. Alternatywy: dopasowana podpróbka U albo model warunkowy w warstwach wzorca. To odpowiednik „target-group background” z modelowania występowania gatunków, gdzie tło dobiera się tak, by miało to samo obciążenie doboru co obserwacje pozytywne (Phillips i in., 2009), oraz ważenia z pracy Makar i in. (2022) | tak dla wzorca zleceń: 85,5% chorych ma wzorzec obecny u co najmniej 50 pacjentów KOR, efektywna liczebność KOR po ważeniu 17 556. **Nie** dla okresu | Model nie zyskuje na samym wzorcu i zachowuje pełną informację o wartościach | Chorzy z wzorcem nieobecnym w KOR (12,3%) wypadają albo wymagają osobnego opisu. Wartości w obrębie wzorca nadal mogą nieść okres | średni |
| **E3. Modelujemy jawnie i neutralizujemy** | Kanał (wskaźniki braków, liczba rekordów) wchodzi do modelu w treningu. Przy ocenie ustawiamy go wszystkim na wspólną wartość albo uśredniamy wynik po jego rozkładzie (Pope i Sydnor, 2011) | tak dla modeli addytywnych (G1, G2); dla drzew przez uśrednianie | Współczynniki wartości laboratoryjnych są szacowane po odjęciu kanału, zamiast przejmować go przez korelację | W modelach z interakcjami neutralizacja jest przybliżona. Działa tylko dla zmierzonych kanałów | niski–średni |
| **E4. Uczymy niezmienniczości** | Uczymy reprezentację, z której nie da się przewidzieć kanału, np. okresu wśród chorych: sieci z adwersarzem (Ganin i in., 2016) albo regularyzacja warunkowej niezależności (Makar i in., 2022) | ograniczona: mało chorych, a okres jest zmienny praktycznie tylko w NEURO | Działa także dla kanałów rozproszonych po wielu cechach | Duża złożoność i dużo strojenia. Łatwo usunąć razem ze skrótem także sygnał choroby | wysoki |
| **E5. Modelujemy mechanizm oznaczania** | Metody PU poza SCAR (F7): prawdopodobieństwo oznaczenia chorego zależy od wybranych cech (okres, wzorzec), a klasyfikator od pozostałych | eksperymentalne | Jedyna rodzina, która formalnie opisuje sytuację „oznaczeni nie są losową próbą chorych” | Identyfikowalność wymaga mocnych założeń. Okres nie pasuje do tego schematu, bo poza oknem nie ma KOR | wysoki |

**Wniosek dla planu.** E0 obowiązuje zawsze jako minimum. E1 i E2 są tanie i dobrze uzasadnione danymi (wiersze 4–5 w sekcji 1), więc powinny trafić do przeglądu szerokiego. E3 jest tanie przy modelach addytywnych. E4 i E5 to opcje badawcze na później, jeśli przegląd pokaże, że E1–E3 zmieniają wnioski.

---

## 8. Oś F: strategia PU

### 8.1 Co metody PU mogą zmienić, a czego nie

**Jeśli zachodzi SCAR.** Załóżmy, że oznaczeni chorzy są losową próbą wszystkich chorych z populacji, z której pochodzi pula U. Wtedy prawdopodobieństwo bycia oznaczonym jest proporcjonalne do prawdopodobieństwa choroby (Elkan i Noto, 2008). Zwykły klasyfikator P-vs-U daje więc ranking taki jak idealny klasyfikator chory-vs-zdrowy, przynajmniej przy dużej próbie i dobrze dobranym modelu. Metody PU zmieniają wtedy dwie rzeczy:

- skalę wyniku, czyli kalibrację i próg;
- w skończonej próbie sposób uczenia elastycznych modeli, które bez korekty uczą się spychać w dół chorych obecnych w puli U.

**Jeśli SCAR nie zachodzi.** U nas oznaczenie zależy od źródła, okresu i sposobu zlecania badań. Klasyfikator P-vs-U uczy się wtedy mieszanki choroby i różnic kohort. Większość metod PU (F2–F6, F8) zakłada SCAR i tego nie naprawia. Problem adresują jedynie metody poza SCAR (F7) oraz opcje osi E.

**Ile chorych jest w puli U.** W teście ukrywania treningowa pula U zawiera średnio 583 ukrytych chorych na 31 864 osoby (1,83%) przy 40% ukrywania. Przy 20% to 0,92%, przy 60% 2,72% (`opcje_fakty`, F). Do tego dochodzą nieznani chorzy w KOR. Ich częstości nie znamy. W metaanalizie Vlak i in. (2011) częstość niepękniętych tętniaków wynosi 3,2% wśród dorosłych bez chorób współistniejących. Gdyby w KOR była podobna, nierozpoznanych chorych w treningowym KOR byłoby rzędu tysiąca. To tylko rząd wielkości, bo KOR jest populacją szpitalną. Wniosek: zanieczyszczenie puli U jest małe, rzędu 1–3%, zarówno przez ukrytych, jak i przez nieznanych chorych. Przy tak małym zanieczyszczeniu różnice w rankingu między F1 a metodami PU mogą być niewielkie. Sprawdzi to benchmark półsyntetyczny (H4), w którym prawdę znamy.

### 8.2 Opcje

| Opcja | Na czym polega | Założenia | Zmienia ranking względem F1? | Za / przeciw u nas | Koszt |
|---|---|---|---|---|---|
| **F1. Naiwny P-vs-U** | Klasyfikator z U jako etykietą 0 | przy SCAR ranking poprawny | — | Punkt odniesienia; każda inna metoda musi się z nim zmierzyć | niski |
| **F2. Ważenie kosztów** | U jako negatywy z mniejszą wagą albo wagi klas, np. „biased SVM” (Liu i in., 2003) | brak jawnego π | tak, zmienia dopasowanie | Tanie; wagi strojone wewnątrz walidacji | niski |
| **F3. Bagging PU** | Wiele modeli uczonych na widocznych P i losowych podpróbkach U; wynik uśredniony (Mordelet i Vert, 2014) | odporny na małe zanieczyszczenie U | tak | Tani, stabilny, z dowolnym modelem składowym. Wersja transdukcyjna ocenia pacjentów U z treningu modelami, które ich nie widziały — przydatne, gdyby celem był ranking całej puli KOR | niski |
| **F4. Elkan–Noto** | (a) korekta wyniku przez stałą c; (b) estymacja c na odłożonych P; (c) ponowny trening z wagami | SCAR | (a) nie; (c) tak | (a) nie zmieni `RecallHidden@q`. (c) to osobna metoda. Przy braku SCAR c i π są obciążone | niski |
| **F5. Ryzyko nieobciążone lub nieujemne (uPU, nnPU)** | Funkcja straty, która przy znanym π koryguje obecność chorych w U. Wersja nieujemna zapobiega przeuczeniu elastycznych modeli (du Plessis i in., 2015; Kiryo i in., 2017) | SCAR, znane π | tak | Wymaga π z estymacji (F8) albo z literatury, najlepiej jako zakres w analizie wrażliwości. Dla regresji logistycznej problem pozostaje wypukły. Naturalna dla G1 i G7 | średni |
| **F6. Metody dwuetapowe** | Najpierw „wiarygodne negatywy” z U, np. techniką szpiegów (Liu i in., 2002; 2003), potem P przeciw tym negatywom, ewentualnie iteracyjnie | wyraźna rozdzielność klas | tak | **Ryzyko u nas:** najpewniejsze negatywy to pacjenci najbardziej typowi dla KOR pod względem okresu i zleceń, więc metoda wzmacnia skróty. Technika szpiegów to w istocie nasze ukrywanie i ma tę samą słabość | średni |
| **F7. PU poza SCAR** | Prawdopodobieństwo oznaczenia zależy od wskazanych cech: założenie SAR (Bekker i in., 2019). Inne założenie: oznaczenie jest tym bardziej prawdopodobne, im typowszy przypadek (Kato i in., 2019) | wskazanie cech wpływających na oznaczenie | tak | Odpowiada naszej sytuacji, ale wymaga decyzji, co jest „cechą oznaczenia”: wzorzec zleceń tak, okres nie pasuje. Kod autorów: `ML-KULeuven/SAR-PU`. Założenie Kato i in. jest u nas wątpliwe, bo oznaczenie zależy od okresu, a nie od typowości choroby | wysoki |
| **F8. Estymacja class prior** | Szacowanie udziału chorych w U: c z Elkan–Noto, TIcE (Bekker i Davis, 2018), KM (Ramaswamy i in., 2016), AlphaMax (Jain i in., 2016) | zwykle SCAR | — (wejście dla F4b i F5) | Przy braku SCAR obciążona. Mamy jednak test kontrolny: przy 20%, 40% i 60% ukrywania wiemy, o ile rośnie udział chorych w U. Estymator powinien rosnąć zgodnie z tym przyrostem | niski–średni |

**Wniosek dla planu.** Minimum to F1 i F3, a F2 i F4c to tanie warianty. F5 ma sens z zakresem wartości π. F6 warto uruchomić raczej jako ostrzeżenie niż jako kandydata. F7 i F8 są najciekawsze poznawczo; ich działanie najlepiej sprawdzić najpierw na benchmarku półsyntetycznym (H4).

---

## 9. Oś G: rodzina algorytmów

| Opcja | Mocne strony u nas | Słabe strony u nas | Braki danych | Interpretacja | Koszt |
|---|---|---|---|---|---|
| **G1. Regresja logistyczna** (L2, L1, elastic net; opcjonalnie splajny) | Szybka i stabilna przy 1 823 chorych. Naturalna dla F2, F4, F5 i E3; czytelne współczynniki | Zależności liniowe, co łagodzą splajny. Współliniowość cech pochodnych wymaga D3 albo regularyzacji. Potrzebne skalowanie | wymaga imputacji albo C3 | wysoka | niski |
| **G2. Modele addytywne** (GAM, Explainable Boosting Machine) | Nieliniowe, a nadal czytelne: każda cecha ma własną krzywą wpływu, więc widać np. skok na typowych wartościach uzupełnień. Dobre do E3 (Lou i in., 2013) | Interakcje tylko dla wybranych par; dodatkowa biblioteka (`interpret`) | zależnie od implementacji | wysoka | średni |
| **G3. Lasy losowe, ExtraTrees** | Mało strojenia; odporne na skalę i cechy pochodne | Słaba kalibracja, bez znaczenia dla rankingu. Wbudowana ważność jest obciążona, lepiej liczyć permutacyjną | NaN natywnie (Random Forest od scikit-learn 1.4) | średnia | niski |
| **G4. Boosting** (HistGradientBoosting, LightGBM, XGBoost) | Zwykle najmocniejsze na danych tabelarycznych (Grinsztajn i in., 2022). HistGradientBoosting jest już używany w diagnostyce | Najszybciej wykorzystuje skróty, zwłaszcza przy NaN; wymaga strojenia | NaN natywnie | średnia (SHAP) | średni |
| **G5. SVM** (liniowy, z jądrem) | Historyczna baza metod PU: biased SVM, bagging SVM | Wersja z jądrem słabo się skaluje powyżej kilkunastu tysięcy przykładów; w baggingu podpróbki mają tylko ~1 750 wierszy. Potrzebne skalowanie i imputacja | wymaga imputacji | niska–średnia | średni |
| **G6. Podobieństwo, kNN** | Najprostszy ranking „jak bardzo przypomina chorych”, zgodny z A5. Zespół zna KNN z etapu 2 | Wrażliwe na skalę i cechy pochodne (D3, D6). W ponad 30 wymiarach odległości tracą kontrast | wymaga imputacji | średnia (najbliżsi sąsiedzi) | niski |
| **G7. Sieci neuronowe** (MLP) | Naturalne środowisko dla nnPU (F5) i adwersarza (E4) | Przy 40 tys. wierszy danych tabelarycznych zwykle nie lepsze od boostingu; dużo strojenia | wymaga imputacji | niska | wysoki |
| **G8. Zespoły, stacking** | Mogą łączyć mocne strony innych modeli | Trudniejsze do interpretacji; zysk zwykle niewielki. Stacking dokłada ryzyko wycieku | — | niska | średni |

Kontrole (losowa, demografia, wzorzec braków) powinny używać tej samej rodziny co porównywany model (`PLAN_MODELOWANIA.md`, sekcja 14.1). Inaczej przewaga mierzy różnicę algorytmów, a nie informację zawartą w wartościach badań.

**Wniosek dla planu.** W przeglądzie szerokim wystarczą trzy rodziny o różnym charakterze: G1 (liniowa), G2 lub G3 (nieliniowa i odporna) oraz G4 (najmocniejsza i najbardziej podatna na skróty). G5–G8 tylko z konkretnym uzasadnieniem.

---

## 10. Oś H: ocena, walidacja i interpretacja

### 10.1 Dlaczego sam test z ukrywaniem nie wystarcza

W teście ukrywania etykietę 1 mają widoczni chorzy, a etykietę 0 ukryci chorzy i pacjenci KOR. Ukryci są losową częścią NEURO. Klasyfikator uczony na takich etykietach szacuje więc P(etykieta 1 | x) = (1 − h) · P(NEURO | x), gdzie h to udział ukrywania. Ranking według P(NEURO | x) jest jednocześnie najlepszym możliwym rankingiem do odzyskiwania losowo ukrytych pacjentów NEURO. Nie ma przy tym znaczenia, czy przynależność do NEURO rozpoznajemy dzięki chorobie, okresowi czy wzorcowi zleceń.

Wniosek praktyczny: wyższy `RecallHidden@q` oznacza lepsze rozpoznawanie kohorty. Model, który ograniczy skróty (E1–E5), w tym teście straci, nawet jeśli lepiej opisuje chorobę.

W tych danych nie ma miary, która bezpośrednio sprawdzałaby wykrywanie choroby. Możemy natomiast:

- wykluczać znane kanały skrótów (H2);
- sprawdzać działanie metod tam, gdzie prawdę znamy (H4);
- badać odporność na inny mechanizm oznaczania (H3);
- szukać błędów w samym procesie (H5).

Wykrywanie choroby wprost sprawdza tylko H8.

### 10.2 Opcje oceny

| Opcja | Na czym polega | Na jakie pytanie odpowiada | Koszt |
|---|---|---|---|
| **H1. `RecallHidden@q` z kontrolami** | Obecny protokół (`PLAN_MODELOWANIA.md`, sekcje 12–14): wynik dla kilku q, Lift, kontrole na tych samych maskach | Jak dobrze model odzyskuje ukrytych i ile z tego dają same skróty | niski; obowiązkowe |
| **H2. Ocena warunkowa względem skrótów** | `RecallHidden@q` lub AUC liczone w warstwach. Warstwy to wzorzec bloków (dla każdego ukrytego dobieramy pacjentów KOR o tym samym wzorcu), okres albo tercyle wyniku kontroli braków. Raportujemy też najgorszą warstwę (por. Sagawa i in., 2020) | Czy model odróżnia ukrytych od KOR także wtedy, gdy znany skrót jest wyrównany? W puli dopasowanej po wzorcu bloków model widzący tylko ten wzorzec daje wszystkim ten sam wynik, więc nie może przewyższyć losowania | niski–średni |
| **H3. Ukrywanie zależne od cech** | Chorych ukrywamy z prawdopodobieństwem zależnym od cech, np. częściej tych z okna dat albo z wzorcem typowym dla KOR. Taki schemat SAR stosuje się w benchmarkach PU (Bekker i in., 2019) | Czy model odnajduje chorych bardziej podobnych do KOR niż typowy chory NEURO, czyli bliższych sytuacji prawdziwych nierozpoznanych | niski |
| **H4. Benchmark półsyntetyczny** | Tylko pacjenci KOR, z ich prawdziwymi korelacjami i brakami. „Chorobę” określa ukryta reguła na wynikach badań o ustalonej sile. Część chorych oznaczamy losowo (SCAR) albo zależnie od wzorca zleceń (SAR) | Która strategia PU (oś F) i jaka estymacja π działa, gdy znamy prawdziwy status wszystkich pacjentów. Można wtedy policzyć też precyzję i kalibrację | średni |
| **H5. Testy placebo** | Losowa grupa pacjentów KOR dostaje etykietę „chorzy”, a potem przechodzi przez pełny pipeline z ukrywaniem. Wynik powinien być bliski losowemu | Czy pipeline nie ma wycieku ani błędu w ocenie | niski |
| **H6. Warstwy okresu** | `RecallHidden@q` osobno dla ukrytych sprzed okna, z okna i po oknie. Wariant: uczenie na chorych sprzed okna, ocena na chorych po oknie | Czy model działa podobnie dla chorych z różnych okresów, czy głównie dla tych, których okres najbardziej odróżnia od KOR | niski |
| **H7. Fold zablokowany do końca** | Jeden fold zewnętrzny odkładamy do końcowego porównania; przegląd i strojenie robimy tylko na pozostałych | Czy wnioski z przeglądu przetrwają na danych, na których niczego nie wybieraliśmy | niski; kosztem mniejszej liczby ukrytych w przeglądzie i szerszego przedziału w ocenie końcowej |
| **H8. Walidacja zewnętrzna lub kliniczna** | Niezależna kohorta zebrana w podobny sposób albo weryfikacja części wskazań | Czy model wskazuje rzeczywistych chorych | zależy od możliwości wskazanych przez prowadzącego |

H7 jest alternatywą lub uzupełnieniem zamrożenia protokołu (`PLAN_MODELOWANIA.md`, sekcja 3.6). Oba rozwiązania chronią przed dopasowywaniem decyzji do wyników testu.

### 10.3 Interpretacja i stabilność (H9)

- Wyjaśnienia liczymy na predykcjach OOF: współczynniki (G1), krzywe wpływu (G2), SHAP lub ważność permutacyjna (G3, G4). Ważność warto liczyć grupami (blok badań albo pomiar razem z cechami pochodnymi), bo cechy pochodne dzielą ją między siebie. Przy skorelowanych cechach wykresy ALE są bezpieczniejsze od wykresów zależności cząstkowej (Apley i Zhu, 2020).
- Sprawdzamy, czy model opiera się na cechach o wysokim udziale uzupełnień, na wzorcu zleceń albo na cechach pochodnych zależnych od laboratorium.
- Sprawdzamy stabilność listy najwyżej ocenionych pacjentów U między ziarnami i modelami, np. odsetek wspólnych pacjentów w top-q. Niestabilna lista nie nadaje się do opisu „kandydatów do diagnostyki”.
- Ocena, czy kierunki wpływu są wiarygodne klinicznie, wymaga konsultacji z klinicystą. Bez niej opisujemy działanie modelu, a nie mechanizm choroby.

**Wniosek dla planu.** H1, H2, H5 i H6 są tanie i niezależne od wybranej metody, więc powinny wejść do wspólnej funkcji oceny od początku. H3 i H4 wymagają generatora masek lub etykiet syntetycznych, ale bez nich nie da się sensownie porównać osi E i F. H7 to decyzja organizacyjna, którą trzeba podjąć przed przeglądem.

---

## 11. Jak łączyć opcje: przykładowe spójne ścieżki

| Ścieżka | Skład | Na jakie pytanie odpowiada | Główne ryzyko |
|---|---|---|---|
| **S1. Klasyczny ranking PU** (obecny `PLAN_MODELOWANIA.md`) | A1 · B1 (MICE → mediana) · C1 · D1 lub D2 · E0 · F1, F3 · G1, G3 · H1 | Ilu ukrytych chorych odzyskuje typowy pipeline PU i jaka jest jego przewaga nad kontrolami | Wynik zdominowany przez rozpoznawanie kohorty; trudno oddzielić, ile wnoszą same wartości badań |
| **S2. Minimum skrótów** | A1 · B1 (mediana bez imputacji) lub B2 · C5 · D4 (morfologia, wiek, płeć) · E1 · F1, F3 · G1, G2 · H1, H2, H6 | Ile rozróżnialności zostaje na wspólnym, prawie kompletnym panelu bez imputacji | Za mało informacji, żeby cokolwiek pokazać. Niski wynik może jednak być prawdziwy |
| **S3. Skróty jawnie pod kontrolą** | A1 · B1 · C3 · D3 · E2 lub E3 · F1, F3 · G1, G2, G4 · H1, H2, H3 | Czy wartości badań wnoszą coś ponad wyrównany wzorzec zleceń | Kanały, których nie wyrównano (okres w wartościach); poprawność ważenia |
| **S4. Górna granica rozpoznawalności** | A4 · B1 · C4 · D1 · E0 · F1 · G4 ze strojeniem · H1 | Jak dobrze w ogóle da się rozpoznać kohortę. To punkt odniesienia, a nie kandydat na model | Pokusa przedstawiania tego wyniku jako skuteczności |
| **S5. Mechanika PU w kontrolowanych warunkach** | H4 · F1–F8 · G1, G4 | Które metody PU i estymatory π działają przy SCAR i przy SAR, gdy znamy prawdę | Reguła syntetyczna jest arbitralna, więc wynik dotyczy metod, a nie tętniaka |

Ścieżki się nie wykluczają. Najwięcej informacji da zestawienie S1, S2 i S3 na tych samych maskach. S4 wyznacza wtedy górny punkt odniesienia, a kontrole dolny.

**Połączenia, które wymagają uwagi:**

- C4 (natywne NaN) działa tylko z G3 i G4 i utrudnia E3, bo nie ma jawnych wskaźników do zneutralizowania.
- F5 wymaga π (z F8 albo jako zakres z literatury) i treningu gradientowego (G1, G7).
- E2 dla wzorca zleceń wymaga zdefiniowania bloków przed porównaniem; propozycja to 8 bloków z `opcje_fakty.py`. Dla okresu E2 jest niewykonalne.
- B2 (pierwszy rekord) wyklucza B4 (cechy przebiegu).
- E1 w wersji z oknem dat daje ~22 ukrytych na fold. Wystarcza to do oceny kierunku wniosku, a nie do porównywania modeli.
- Każdy wariant, który zmienia wejście imputera (D2, D3, C7, okno dat), wymaga nowego dopasowania MICE. Warianty z C2, C4 i C5 nie potrzebują MICE wcale.
- Do przeglądu bez strojenia wystarczą zapisane foldy zewnętrzne. Strojenie (faza 3 w sekcji 12) wymaga preprocessingu dopasowanego od nowa w foldach wewnętrznych (`PLAN_MODELOWANIA.md`, sekcja 15).

---

## 12. Plan dalszych prac

Zasada: najpierw budujemy to, co jest wspólne dla wszystkich opcji. Potem tanio sprawdzamy wiele opcji, a dopiero na końcu stroimy kilka wybranych. Każda faza kończy się konkretnym rezultatem, a tam, gdzie trzeba, decyzją.

### Faza 0: ramy (przed kodowaniem modeli)

- Odpowiedzi prowadzącego na pytania z sekcji 13 albo zapis, że wobec ich braku obowiązuje ujęcie A1.
- Wybór wersji danych: obecna z opisanymi ograniczeniami czy odtworzenie etapu 1.
- Stałe oceny: wartości q (np. 1%, 5%, 10%), udział ukrywania, ziarna masek.
- Decyzja o H7: fold zablokowany do końca czy tylko zamrożony protokół.
- **Rezultat:** jedno zdanie celu, lista opcji dopuszczonych do przeglądu i lista opcji wykluczonych z uzasadnieniem.

### Faza 1: wspólna infrastruktura, niezależna od metody

- Odczyt danych i przydziału z kontrolą spójności (`PLAN_MODELOWANIA.md`, sekcja 11).
- Generatory:
  - masek losowych (H1) i zależnych od cech (H3);
  - etykiet półsyntetycznych (H4);
  - etykiet placebo (H5).
- Jedna funkcja oceny dla wszystkich modeli: `RecallHidden@q` dla kilku q, Lift, kontrole, ocena warunkowa (H2), warstwy okresu (H6), liczniki i mianowniki.
- Wspólny format predykcji OOF i konfiguracji. Identyfikatory opcji z tego dokumentu mogą być polami konfiguracji, np. `reprezentacja = "B2"`.
- **Rezultat:**
  - kontrola losowa i jeden prosty model przechodzą przez cały przepływ;
  - test placebo daje wynik bliski losowemu;
  - ręczny przykład metryki zgadza się z kodem.

### Faza 2: przegląd szeroki bez strojenia

- Tani punkt odniesienia, np. B1 (mediana) · C2 · D1 · E0 · F1 · G1, i zmiana jednej osi naraz:
  - B: B2, B3;
  - C: C1 (zapisane foldy), C3, C4 (z G3 lub G4), C5, C7;
  - D: D2, D3, D4;
  - E: E1 (okno, panel), E2 (ważenie po wzorcu), E3;
  - F: F2, F3, F4c oraz F5 z kilkoma wartościami π;
  - G: G2 lub G3 oraz G4.
- Kilka połączeń, które same się narzucają: ścieżki S2 i S3 w całości oraz S4 jako górna granica.
- Równolegle S5: oś F i estymacja π (F8) na benchmarku półsyntetycznym.
- Wszystkie ustawienia zapisujemy przed uruchomieniem. Wszystkie warianty liczymy na tych samych maskach i oceniamy według H1, H2 i H6, a wybrane także według H3.
- Koszt: większość wariantów to dopasowanie klasyfikatora na ~32 tys. profili treningowych. Czas wyznaczają warianty z nowym MICE (około 3 min na fold według etapu 3).
- **Rezultat:** tabela „oś → jak zmienia `RecallHidden@q`, przewagę nad kontrolą braków i wynik warunkowy H2”. Celem jest wiedza, które decyzje mają znaczenie, a nie wyłonienie zwycięzcy.

### Bramka decyzyjna 1 (kryteria zapisane przed fazą 2)

- **Jeśli żaden wariant nie przewyższa kontroli braków** w ocenie warunkowej H2 wyraźniej niż wynosi rozrzut między ziarnami i w porównaniu sparowanym: głównym wynikiem projektu jest ustalenie metodologiczne. Strojenie nie ma wtedy sensu; zostają analizy odporności i raport.
- **Jeśli przewaga istnieje:** wybieramy 2–3 konfiguracje reprezentujące różne podejścia, np. po jednej z S1, S2 i S3, a nie trzy warianty tej samej.
- Osie, które w przeglądzie nie zmieniały wyników, ustalamy na najprostszej opcji.

### Faza 3: krótka lista i strojenie

- Walidacja zagnieżdżona 5 × 3 z preprocessingiem dopasowywanym w foldach wewnętrznych i z równym budżetem strojenia dla każdej konfiguracji (`PLAN_MODELOWANIA.md`, sekcja 15).
- Funkcja celu ustalona przed startem: `RecallHidden@q` (H1) albo miara warunkowa (H2). Miara warunkowa mniej nagradza skróty, ale jest bardziej zaszumiona.
- **Rezultat:** porównanie strojonych konfiguracji z kontrolami, predykcje OOF i niepewność z bootstrapu pacjentów.

### Faza 4: analizy odporności (lista zamknięta przed startem)

- H3 (ukrywanie zależne od cech) i H6 (warstwy okresu).
- E1: okno dat z preprocessingiem dopasowanym w oknie.
- D2 oraz warianty imputacji (C1, C2, C4).
- H5 na pełnym pipeline.
- Wariant bez 63 pacjentów obecnych w obu kohortach.
- **Rezultat:** wiadomo, które wnioski są stabilne, a które zależą od wariantu.

### Faza 5: interpretacja i raport

- Interpretacja (H9) i profile najwyżej ocenionych pacjentów U, liczone wyłącznie z OOF.
- Ograniczenia i opis wszystkich zaplanowanych porównań, także tych ze spadkiem wyniku.
- Raport według listy kontrolnej dla modeli predykcyjnych, np. TRIPOD+AI (Collins i in., 2024), w zakresie, w jakim pasuje do badania eksploracyjnego.
- H8, jeśli prowadzący wskaże taką możliwość.

### Co można robić równolegle

- W fazie 1 są trzy niezależne części: dane i preprocessing; maski i ocena; generator półsyntetyczny.
- W fazie 2 pracę można podzielić po osiach. Każda osoba odpowiada za jedną lub dwie osie i uruchamia swoje warianty przez wspólny pipeline. Przykładowy podział:
  - braki i imputacja — tu można wykorzystać implementacje KNN i MissForest z etapu 2;
  - reprezentacja i cechy;
  - strategie PU razem z benchmarkiem półsyntetycznym;
  - ocena, kontrole i skróty.
- Przez cały czas trwa tor informacyjny: pytania do prowadzącego (sekcja 13). Odpowiedzi mogą otworzyć ujęcia A2 i A3.

---

## 13. Pytania, od których zależy wybór opcji

| Pytanie | Jeśli odpowiedź… | …to w planie |
|---|---|---|
| Skąd pochodzi KOR: oddział, kryteria włączenia, związek z pandemią? | populacja szpitalna z konkretnego oddziału | A1 z opisem populacji. Zastosowanie przesiewowe odpada. E2 można rozszerzyć o dopasowanie kontekstu klinicznego, jeśli pojawią się takie dane |
| Kiedy wykonano badania NEURO względem rozpoznania i leczenia? Czy są daty rozpoznania? | badania z hospitalizacji, bez dat | A2 jako uczciwy opis zadania; B2 (pierwszy rekord) zyskuje na znaczeniu |
| | daty dostępne i są rekordy sprzed rozpoznania | A3 i B6 stają się wykonalne. To jedyna droga do „oceny ryzyka” w ścisłym sensie |
| Czy NEURO obejmuje tętniaki pęknięte (krwotok podpajęczynówkowy), niepęknięte, czy jedne i drugie? *Tego pytania nie ma jeszcze w dotychczasowych dokumentach* | głównie pęknięte | Profil może opisywać ostrą fazę krwotoku i leczenie, a nie samego tętniaka. Warto sprawdzić podgrupy (A6) i ostrożnie traktować cechy zapalne oraz elektrolity |
| | mieszane, z oznaczeniem typu | Analizy osobno w podgrupach; ukrywanie w warstwach typu tętniaka |
| Czy istnieje jakakolwiek wiedza o częstości nierozpoznanych tętniaków w KOR? | brak | π dla F4b i F5 jako zakres w analizie wrażliwości, np. wokół 3,2% z metaanalizy Vlak i in. (2011) |
| Jaki odsetek pacjentów można realnie skierować na obrazowanie? | konkretna liczba | q wyprowadzone z przepustowości, a nie przyjęte jako propozycja badawcza |
| Czy możliwa jest niezależna kohorta albo weryfikacja kliniczna części wskazań? | tak | H8; plan oceny warto wtedy podporządkować tej walidacji |
| Czy można uzyskać dodatkowe zmienne: rozpoznania współistniejące, nadciśnienie, palenie, wywiad rodzinny? | tak | Nowe opcje w osi D. W literaturze jako czynniki ryzyka tętniaków opisuje się m.in. płeć żeńską, wiek, nadciśnienie, palenie, obciążający wywiad rodzinny i wielotorbielowatość nerek. W obecnych danych są tylko wiek i płeć |

---

## 14. Czego żadna opcja modelowania nie rozwiąże

- Bez walidacji zewnętrznej lub klinicznej nie zmierzymy skuteczności wśród rzeczywistych nierozpoznanych chorych w KOR.
- Bez dat rozpoznania nie powstanie model „sprzed choroby”. Każdy model opisuje profil z okresu, z którego pochodzą badania.
- Choroba i pochodzenie danych są w pełni sprzężone, bo wszyscy znani chorzy pochodzą z jednego źródła. Opcje osi E ograniczają znane kanały skrótów, ale nie mogą wykluczyć niemierzalnych, np. typu oddziału.
- Historyczne czyszczenie z etapu 1, ustawione osobno dla kohort (kwantyle, limit historii 11/38), pozostaje, dopóki nie odtworzymy tego etapu.
- Wynik pozostaje rankingiem. Prawdopodobieństwo choroby wymagałoby znanego π i spełnionego SCAR, a ten jest wątpliwy.

---

## 15. Literatura i narzędzia

**Uczenie PU: podstawy i przeglądy**

- Elkan C., Noto K. (2008). Learning classifiers from only positive and unlabeled data. KDD. [PDF](https://cseweb.ucsd.edu/~elkan/posonly.pdf)
- Bekker J., Davis J. (2020). Learning from positive and unlabeled data: a survey. *Machine Learning* 109, 719–760. [arXiv:1811.04820](https://arxiv.org/abs/1811.04820)
- Mordelet F., Vert J.-P. (2014). A bagging SVM to learn from positive and unlabeled examples. *Pattern Recognition Letters* 37, 201–209. [arXiv:1010.0772](https://arxiv.org/abs/1010.0772)
- Liu B., Lee W.S., Yu P.S., Li X. (2002). Partially supervised classification of text documents. ICML.
- Liu B., Dai Y., Li X., Lee W.S., Yu P.S. (2003). Building text classifiers using positive and unlabeled examples. ICDM.
- du Plessis M.C., Niu G., Sugiyama M. (2015). Convex formulation for learning from positive and unlabeled data. ICML.
- Kiryo R., Niu G., du Plessis M.C., Sugiyama M. (2017). Positive-unlabeled learning with non-negative risk estimator. NeurIPS. [arXiv:1703.00593](https://arxiv.org/abs/1703.00593)

**Class prior i założenia o oznaczaniu**

- Bekker J., Davis J. (2018). Estimating the class prior in positive and unlabeled data through decision tree induction. AAAI.
- Ramaswamy H., Scott C., Tewari A. (2016). Mixture proportion estimation via kernel embeddings of distributions. ICML.
- Jain S., White M., Radivojac P. (2016). Estimating the class prior and posterior from noisy positives and unlabeled data. NeurIPS.
- Bekker J., Robberechts P., Davis J. (2019). Beyond the selected completely at random assumption for learning from positive and unlabeled data. ECML PKDD. [arXiv:1809.03207](https://arxiv.org/abs/1809.03207); kod: [ML-KULeuven/SAR-PU](https://github.com/ML-KULeuven/SAR-PU)
- Kato M., Teshima T., Honda J. (2019). Learning from positive and unlabeled data with a selection bias. ICLR.

**Dane obserwacyjne, skróty i zakłócenia**

- Agniel D., Kohane I.S., Weber G.M. (2018). Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. *BMJ* 361:k1479. [BMJ](https://www.bmj.com/content/361/bmj.k1479). Samo zlecenie badania i jego czas bywały silniej związane z przeżyciem niż wynik badania.
- Geirhos R. i in. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2, 665–673.
- Zech J.R. i in. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine* 15(11): e1002683. Model rozpoznawał m.in. szpital, z którego pochodziło zdjęcie.
- DeGrave A.J., Janizek J.D., Lee S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence* 3, 610–619.
- Phillips S.J., Dudík M., Elith J. i in. (2009). Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. *Ecological Applications* 19(1), 181–197. Tło o takim samym obciążeniu doboru jak obserwacje pozytywne.
- Ward G., Hastie T., Barry S., Elith J., Leathwick J.R. (2009). Presence-only data and the EM algorithm. *Biometrics* 65(2), 554–563.
- Makar M., Packer B., Moldovan D., Blalock D., Halpern Y., D'Amour A. (2022). Causally motivated shortcut removal using auxiliary labels. AISTATS, PMLR 151. [arXiv:2105.06422](https://arxiv.org/abs/2105.06422)
- Pope D.G., Sydnor J.R. (2011). Implementing anti-discrimination policies in statistical profiling models. *American Economic Journal: Economic Policy*.
- Ganin Y. i in. (2016). Domain-adversarial training of neural networks. *JMLR* 17(59), 1–35.
- Sagawa S., Koh P.W., Hashimoto T.B., Liang P. (2020). Distributionally robust neural networks for group shifts: on the importance of regularization for worst-case generalization. ICLR.

**Braki danych w predykcji**

- Sperrin M., Martin G.P., Sisk R., Peek N. (2020). Missing data should be handled differently for prediction than for description or causal explanation. *Journal of Clinical Epidemiology* 125, 183–187.
- Josse J., Chen J.M., Prost N., Scornet E., Varoquaux G. (2019). On the consistency of supervised learning with missing values. [arXiv:1902.06931](https://arxiv.org/abs/1902.06931)
- Twala B., Jones M.C., Hand D.J. (2008). Good methods for coping with missing data in decision trees. *Pattern Recognition Letters* 29(7), 950–956.

**Modele i interpretacja**

- Lou Y., Caruana R., Gehrke J., Hooker G. (2013). Accurate intelligible models with pairwise interactions. KDD.
- Grinsztajn L., Oyallon E., Varoquaux G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS, Datasets and Benchmarks.
- Apley D.W., Zhu J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *Journal of the Royal Statistical Society B* 82(4), 1059–1086.
- Aitchison J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.

**Dziedzina i raportowanie**

- Vlak M.H.M., Algra A., Brandenburg R., Rinkel G.J.E. (2011). Prevalence of unruptured intracranial aneurysms, with emphasis on sex, age, comorbidity, country, and time period: a systematic review and meta-analysis. *Lancet Neurology* 10(7), 626–636. [PubMed](https://pubmed.ncbi.nlm.nih.gov/21641282/). Częstość 3,2% (95% CI 1,9–5,2) w populacji bez chorób współistniejących, średni wiek 50 lat.
- Collins G.S. i in. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 385:e078378.

**Narzędzia.** Przed użyciem trzeba sprawdzić zgodność wersji; na Macu jest scikit-learn 1.6.1, na Windowsie 1.8.

- scikit-learn: `LogisticRegression`, `HistGradientBoostingClassifier` (NaN natywnie), `RandomForestClassifier` (NaN od wersji 1.4), `IterativeImputer(sample_posterior=True)` dla C6.
- [`pulearn`](https://pypi.org/project/pulearn/): `ElkanotoPuClassifier`, `WeightedElkanotoPuClassifier`, `BaggingPuClassifier`. Nazwy argumentów i kodowanie etykiet różnią się między wersjami pakietu.
- SAR-PU (F7): kod autorów, [ML-KULeuven/SAR-PU](https://github.com/ML-KULeuven/SAR-PU).
- nnPU (F5): istnieją implementacje referencyjne w PyTorch; wersja dla regresji logistycznej to kilkadziesiąt linii własnego kodu.
- EBM (G2): pakiet `interpret`.
