# Plan modelowania Positive-Unlabeled — od SGKF do wyników (propozycja)

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Status:** **propozycja do uzgodnienia przez zespół i prowadzącego.** Dokument łączy wcześniejsze ustalenia z planem wykonania etapu 4. Wskazane niżej wartości startowe są rekomendacjami, a nie potwierdzonymi decyzjami zespołu. Opis przyszłego kodu nie oznacza, że został on już zaimplementowany.
**Stan na:** 18.10.2026. Zachowano decyzje dotyczące danych z 04.10 oraz uzupełnienia z weryfikacji 16–17.10.2026 (`1026_SPOTKANIE_PROWADZACY.md`, sekcja 9); dodano szczegółową kolejność dalszych prac, objaśnienia i kryteria ukończenia.

To główny dokument planistyczny w `4-pu-setup/`. Aktualna wersja wejścia, podział i imputacja w foldach zostały wykonane i sprawdzone technicznie (etapy 1–3, `1026_REALIZACJA_DO_SGKF.md`). Przed porównaniem modeli trzeba potwierdzić zakres eksperymentu i zapisać protokół, czyli wspólne zasady jego wykonania. Zmiana danych może wymagać nowego zamrożenia artefaktów; sam rozwój kodu modelowania nie wymaga tworzenia nowych foldów zewnętrznych.

**Jak czytać ten plan:**

- Jeśli dołączasz do etapu modelowania: zacznij od słownika poniżej, sekcji 1 i mapy prac w sekcji 9.
- Jeśli przygotowujesz spotkanie zespołu: sekcje 7 i 10 zbierają decyzje oraz proponowane wartości startowe.
- Jeśli implementujesz: sekcje 11–15 opisują dane, ukrywanie, metryki, modele i strojenie.
- Jeśli odpowiadasz za interpretację i raport: sekcje 8 i 16–17 opisują ograniczenia, analizy wrażliwości i wyniki.
- Sekcje 18–19 pozwalają rozdzielić pracę i sprawdzić, czy pierwszy eksperyment jest ukończony.

Sekcje 1–8 zachowują dotychczasowe uzasadnienia i historię. Sekcje 9–19 rozwijają je w plan wykonania. Nie trzeba znać wszystkich algorytmów, żeby zrozumieć kolejność prac i ocenić, czy są zakończone.

### Słownik potrzebny do dalszej części

| Pojęcie | Co oznacza w naszym projekcie |
|---|---|
| Cecha | Informacja podawana modelowi, np. hemoglobina, kreatynina, wiek. Identyfikator pacjenta służy do organizacji danych i nie jest cechą |
| Etykieta | Informacja, z którą model się uczy. U nas wskazuje znanego pacjenta NEURO albo osobę nieoznaczoną; nie jest pełną diagnozą wszystkich pacjentów |
| PU: Positive–Unlabeled | Uczenie na znanych pozytywnych P i nieoznaczonych U. W U mogą być zarówno zdrowi, jak i niewykryci chorzy |
| Rekord / profil pacjenta | Rekord to jeden tydzień badań. Profil to jeden wiersz opisujący pacjenta, uzyskany np. przez medianę jego wyników |
| Model / trening | Model wylicza wynik na podstawie cech. Trening to dopasowanie jego reguł do danych treningowych |
| Preprocessing | Przygotowanie cech przed modelem: m.in. uzupełnienie braków i skalowanie |
| Imputacja / MICE | Oszacowanie brakujących wartości. U nas „MICE” oznacza deterministyczny `IterativeImputer` z drzewami ExtraTrees; nie tworzymy wielu możliwych wersji braków ani nie mierzymy w ten sposób całej ich niepewności |
| Skalowanie | Przeliczenie cech na porównywalne skale. Nie zmienia jednostek w danych źródłowych, tylko sposób reprezentacji na potrzeby uczenia |
| Fold | Jedna z części danych. W pięciu przebiegach każda część kolejno służy do testowania, a pozostałe do treningu |
| Walidacja / test | Walidacja pomaga wybierać ustawienia podczas prac. Test służy do oceny ustalonego wariantu; jego wyników nie używamy do poprawiania tego samego wariantu |
| Hiperparametr / strojenie | Ustawienie ustalane przed treningiem, np. głębokość drzewa. Strojenie porównuje takie ustawienia na walidacji |
| Baseline, czyli model bazowy | Prosty punkt odniesienia. Pozwala sprawdzić, czy bardziej złożona metoda wnosi korzyść |
| OOF, czyli out-of-fold | Wynik pacjenta uzyskany przez model, który nie uczył się na tym pacjencie. Każdy pacjent ma taki wynik z własnego foldu testowego |
| Seed, czyli ziarno losowania | Liczba pozwalająca odtworzyć losowanie. Osobno zapisujemy losowanie ukrytych chorych i losowość algorytmu |
| Wyciek informacji | Wykorzystanie przy uczeniu informacji z części, która miała służyć do niezależnej oceny. Może dotyczyć także imputera i scalera |
| Skrót | Łatwa informacja pozwalająca rozpoznać kohortę, np. zestaw zleconych badań. Dobry wynik dzięki niej nie dowodzi rozpoznania choroby |
| Analiza wrażliwości | Powtórzenie porównania po ustalonej zmianie, np. ograniczeniu okresu. Sprawdza, od czego zależy wniosek |
| Zamrożenie protokołu | Zapisanie zasad przed porównaniem modeli. Późniejsza zmiana jest dozwolona, ale otrzymuje nową wersję i uzasadnienie |

---

## 1. Na czym polega zadanie i dlaczego ocena jest nietypowa

Mamy klasę **P** (NEURO, potwierdzony tętniak) i klasę **U** (KOR: w danych nie ma u nich rozpoznania tętniaka, ale nie wiemy, czy i u kogo wykonano obrazowanie; profil laboratoryjny wskazuje raczej na pacjentów szpitalnych niż na populację ogólną, sekcja 2). `label = 0` znaczy „nieoznaczony”, a nie „zdrowy”. Konsekwencje:

1. **Nie zmierzymy liczby pomyłek.** Wysoki wynik pacjenta KOR może być błędem albo niewykrytym chorym, więc „false positive” traci sens.
2. **Accuracy jest bezużyteczna.** Przy 4,45% pozytywnych model mówiący zawsze „0” ma 95,5% trafności.
3. **Oceniamy kontrolowanym testem odzyskiwania** (pomysł prowadzącego). Część znanych chorych ukrywamy w puli U i sprawdzamy, czy model wypchnie ich na górę rankingu. Ten test nie usuwa skrótów. Ukryci zachowują cechy kohorty NEURO, a model na samym wzorcu braków odzyskuje około 52% z nich w górnych 5% (sekcja 2). Dlatego każdy wynik porównujemy z modelami kontrolnymi na tych samych maskach (sekcje 3.4 i 4.2).

Cały protokół podporządkowany jest pytaniu: *czy wynik mówi coś o tętniakach, czy o tym, jak powstały dane?*

---

## 2. Wejście: co jest gotowe

| Element | Gdzie | Stan |
|---|---|---|
| dane po decyzjach z 04.10.2026 (38 cech, przed imputacją) | `data/processed/aneurysm_sgkf_input.csv` (część A etapu 3) | **gotowe** |
| przydział pacjent → fold (5 foldów, stratyfikacja etykieta × okno × liczba rekordów) | `3-sgkf-split/sgkf/results/pacjent_fold.csv` | **zamrożone** |
| MICE wewnątrz foldu (ExtraTrees 7/78/10) | `3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py` | gotowe |
| zaimputowane foldy zewnętrzne (38 cech) | `3-sgkf-split/sgkf/results/foldy_imputowane_38/` (lokalnie, poza gitem) | **policzone**, odtwarzalne jednym poleceniem |

**Zasada nadrzędna:** foldy wyłącznie z `podzial.wczytaj_podzial()`, nigdy liczone od nowa. Każde dopasowanie (scaler, imputer, selekcja cech, model, próg) odbywa się tylko na części treningowej.

---

## 3. Przygotowanie: protokół przed pierwszym modelem

### 3.0 Założenia wejściowe: co już rozstrzygnięto, a co zostaje

**Rozstrzygnięte 04.10.2026** (wdrożone w etapie 3, szczegóły w `1026_REALIZACJA_DO_SGKF.md`, sekcja 6):

| Decyzja | Co to oznacza dla modelowania |
|---|---|
| 63 pacjentów w obu kohortach: pozytywni, rekordy KOR usunięte | nic do zrobienia; lista tych pacjentów jest w `3-sgkf-split/przygotowanie/results/usuniete_rekordy.csv`, gdyby trzeba było ich wykluczyć w analizie wrażliwości |
| wartości niemożliwe (KREA, K, Na) → brak, uzupełniane w foldzie | nic do zrobienia |
| zestaw cech: **38, jeden wariant** (selekcja z etapu 1 cofnięta) | ewentualny wybór cech tylko **wewnątrz** walidacji krzyżowej; MONO i %MONO są redundantne (rozmaz sumuje się do 100%), co jest istotne dla modeli liniowych |
| rozjazd czasowy kohort: **bez korekty**, opisany jako ograniczenie | **nie dodawać roku ani daty jako cechy**, bo ułatwia rozpoznanie kohorty. Diagnostyka pokazała, że epokę widać w wynikach samych chorych (AUC 0,83), więc **analiza wrażliwości we wspólnym oknie dat jest konieczna**, a nie tylko „ewentualna” (decyzja do potwierdzenia z prowadzącym) |
| asymetria braków (NEURO 26%, KOR 12% komórek uzupełnianych): **bez korekty** | sam wzorzec braków odróżnia kohorty z AUC 0,82, a imputacja zostawia ślad (podpis NRBC). Zalecana analiza wrażliwości: model na cechach o niskim odsetku braków w obu kohortach |

**Co pokazała analiza wyników SGKF** (`3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcje 4.3 i 5), czyli rzeczy, które plan musi uwzględnić:

| Ustalenie | Konsekwencja dla modelowania |
|---|---|
| kohorty rozdzielne na cechach po imputacji z AUC **0,93** (pacjenci), a we wspólnym oknie dat z AUC 0,80 (bez imputacji 0,83) | wysokie AUC P-vs-U **nie jest** dowodem wykrywania tętniaka; raportujemy je wyłącznie jako metrykę P-vs-U |
| sam wzorzec braków (które badania zlecono) odróżnia kohorty z AUC **0,82** | analiza wrażliwości na cechach z niskim odsetkiem braków w obu kohortach (morfologia, wiek, płeć); porównanie z modelem na pełnym zestawie |
| **ukryci chorzy zachowują cechy kohorty**: model na samym wzorcu braków odzyskuje **52%** ukrytych w górnych 5% rankingu; losowy ranking 5%, sama demografia 14%, 38 cech 66–70% (`3-sgkf-split/sgkf/kontrola_skrotow.py`, protokół orientacyjny) | `RecallHidden@q` raportujemy zawsze obok modeli kontrolnych na tych samych maskach; liczy się przewaga nad nimi, a nie sama wartość; wynik opisujemy jako odzyskiwanie znanych chorych w tym zbiorze |
| **podpis imputacji NRBC / %NRBC**: zmierzone to w 78% dokładne zera, uzupełnione nigdy nie są zerem; NRBC brakuje w 48% rekordów NEURO i 3% KOR, więc 63% rekordów z uzupełnionym NRBC to NEURO (w całych danych 8,5%); samo NRBC po imputacji odróżnia kohorty z AUC 0,73, tak jak sam wskaźnik braku | przed modelowaniem rozstrzygnąć: wykluczyć NRBC i %NRBC (niska wartość kliniczna u dorosłych) albo sprawdzić ważność tych cech w modelu; inaczej model wykorzysta informację o braku badania |
| przy brakach całymi panelami (koagulogram, glukoza: ~50% rekordów) MICE jest niewiele lepsze od średniej z treningu (błąd 98% błędu średniej w KOR, 89% w NEURO), a uzupełnienia skupiają się wokół średniej; przy skośnych rozkładach są wyższe od typowej zmierzonej wartości (CRP, GLU) | ważność tych cech interpretować ostrożnie; rozważyć wskaźniki braków jako jawną informację zamiast ukrytego śladu |
| epoka rozpoznawalna w wynikach NEURO z AUC **0,83** (po imputacji); przed imputacją, na tych samych rekordach, wzorzec braków daje 0,76–0,77, a zmierzone wartości paneli 0,56–0,65 | analiza wrażliwości we wspólnym oknie dat (patrz wyżej) i na wartościach zmierzonych |
| KOR ma profil szpitalny (CRP > 5 mg/l w 61% zmierzonych wyników; CRP zmierzono w 81% rekordów KOR) | interpretacja klasy U i zastosowania „przesiewowego” zależy od odpowiedzi prowadzącego, skąd pochodzi KOR |

**Zostaje do potwierdzenia lub decyzji przy planie:**

| Kwestia | Stan | Wpływ |
|---|---|---|
| pochodzenie kohorty KOR | profil laboratoryjny wskazuje na pacjentów szpitalnych z lat 2020–2021, a nie populację ogólną | interpretacja klasy U; sens zastosowania przesiewowego |
| pochodzenie wyników NEURO i cel predykcji | założenie robocze: hospitalizacja (mediana rozpiętości badań 35 dni u 1 215 pacjentów z >1 rekordem); nie wiadomo, czy badania są sprzed rozpoznania, z jego czasu, czy po leczeniu | interpretacja: model rozpoznaje pacjenta **hospitalizowanego** z tętniakiem, nie ryzyko przesiewowe; bez odpowiedzi bezpieczny zakres to eksploracyjny ranking PU w dostępnych kohortach |
| czyszczenie z etapu 1 | progi kwantylowe i limit historii (11 rekordów KOR, 38 NEURO) ustalone osobno dla kohort, na całych danych przed podziałem; na razie ograniczenie | do decyzji: ujednolicić reguły przy ponownym zamrożeniu danych (wspólny limit 11 usunąłby 943 rekordy 94 chorych) |
| walidacja zewnętrzna | brak niezależnej kohorty | bez niej wynik opisujemy jako odzyskiwanie znanych chorych w tym zbiorze |
| NRBC i %NRBC | rekomendacja: wykluczyć z modelowania albo kontrolować (podpis imputacji) | zestaw cech modelu |
| reguła agregacji rekordów do pacjenta | rekomendacja: **mediana**, średnia jako wrażliwość | parametr pipeline'u |

**Uzasadnienie reguły agregacji** (`4-pu-setup/etap0_agregacja.py`, wyniki w `4-pu-setup/results/`; od 16.10.2026 na obecnym wejściu etapu 3, 38 cech; pierwsza wersja z 09.2026 na danych sprzed przygotowania dała te same wnioski). 41% pacjentów ma więcej niż jeden rekord, a NEURO ma ich średnio dwa razy więcej niż KOR (3,65 wobec 1,83). Może to odzwierciedlać hospitalizację lub inny sposób zbierania badań; pochodzenie danych wymaga potwierdzenia. Reguła wrażliwa na liczbę pomiarów tworzy sygnał z samej częstości badania. Korelacja liczby rekordów z wartością cechy, liczona wśród pacjentów KOR z więcej niż jednym rekordem:

| Reguła | Mediana \|ρ\| | Maks. \|ρ\| | Cech z \|ρ\| > 0,2 |
|---|---:|---:|---:|
| **mediana** | 0,079 | 0,257 | **2** |
| ostatnia dostępna wartość | 0,086 | 0,296 | 1 |
| średnia | 0,124 | 0,284 | 7 |
| maksimum | 0,204 | 0,379 | **19** |

Pozorna siła sygnału jest przy tym zbliżona dla średniej, mediany i maksimum (mediana \|AUC − 0,5\|: 0,0429, 0,0415, 0,0503). Maksimum nie jest rekomendowane jako reguła główna, bo wiąże się z liczbą badań w 19 z 38 cech i wciąga skrajne wartości (K poza pomocniczym zakresem 1,5–9 mmol/l u 57 pacjentów wobec 10 przy medianie; nie oznacza to automatycznie błędów pomiaru). „Ostatnia dostępna wartość” daje najsilniejszy pozorny sygnał (0,0522), ale także nie jest rekomendowana jako reguła główna. W skrypcie to `groupby.last()`, czyli ostatnia niepusta wartość każdej cechy osobno, więc profil może łączyć różne badania, a nie pochodzić z jednego ostatniego badania. Ponadto przy hospitalizacji ostatnie wyniki mogłyby być związane z leczeniem, inaczej niż w KOR. Jest to hipoteza do sprawdzenia po uzyskaniu informacji o pochodzeniu danych, a nie potwierdzone wyjaśnienie silniejszego sygnału.

### 3.1 Foldy z walidacją zagnieżdżoną

Tych samych foldów zewnętrznych nie wolno użyć naraz do strojenia i do raportowania wyniku, bo to zawyża jakość. Układ docelowy:

```
dla każdego foldu zewnętrznego (ocena):
    foldy wewnętrzne na części treningowej (podzial.podzial_wewnetrzny, ta sama stratyfikacja)
    ├─ scaler + MICE dopasowywane OD NOWA w każdym foldzie wewnętrznym
    ├─ strojenie hiperparametrów na foldach wewnętrznych
    └─ najlepsza konfiguracja trenowana na całym treningu foldu zewnętrznego
    jedna ocena na foldzie zewnętrznym
```

Nie wolno zaimputować całego outer-train raz i dopiero potem dzielić go na foldy wewnętrzne, bo to wyciek przez preprocessing.

### 3.2 Ukrywanie części pozytywnych

- losujemy **całych pacjentów** z NEURO, nigdy pojedyncze rekordy,
- trzy kolumny: `true_label` (pochodzenie: 1 = znany chory z NEURO), `observed_label` (to, co widzi model), `is_hidden` (flaga kontrolna). `true_label = 0` nie oznacza zdrowego: prawdziwy status KOR jest nieznany, dlatego w kodzie lepiej nazwać tę kolumnę `original_label`,
- scenariusz główny, np. 40% ukrytych, oraz 20% i 60% jako wrażliwość, każdy na 5–10 ziarnach,
- **identyczne maski dla wszystkich modeli**,
- maski są deterministyczne i wspólne dla modeli. Zapisujemy parametry, dokładny algorytm oraz małą tabelę ukrytych **pacjentów** z odciskiem; nie trzeba powielać flag dla wszystkich rekordów. Szczegóły w sekcji 12.

### 3.3 Terminologia

| Sytuacja | Jak mówimy |
|---|---|
| ukryty chory z wysokim wynikiem | sukces (TP względem `true_label`), nie „false positive” |
| pacjent KOR z wysokim wynikiem | „wysoko sklasyfikowany pacjent U”, „kandydat do diagnostyki”, nigdy „false positive” |
| wynik modelu | **ranking (risk score)**, nie „prawdopodobieństwo tętniaka”. Sama kalibracja tego nie uzasadnia, bo nie znamy statusu KOR i nie mamy podstaw, by przyjąć losowy dobór oznaczonych chorych, czyli założenie SCAR |
| wysoki `RecallHidden@q` | „odzyskiwanie znanych chorych w tym zbiorze”, a nie „skuteczność wykrywania tętniaka” |

### 3.4 Metryki: na pacjentach, nie na rekordach

Pacjent NEURO ma średnio 3,65 rekordu, a KOR 1,83, więc metryki wierszowe premiowałyby chorych.

| Metryka | Rola |
|---|---|
| **RecallHidden@q**: odsetek ukrytych chorych w górnych `q` puli U | **główna** |
| udział ukrytych w top-`q` | uzupełnienie; **nie jest** estymatorem precision |
| Lift@q | czy model jest lepszy od losowania |
| **modele kontrolne**: losowy ranking pacjentów, sama demografia, sam wzorzec braków, liczone na tych samych maskach | **obowiązkowy punkt odniesienia** dla `RecallHidden@q`: pokazuje, ile wyniku dają same skróty |
| RecallKnown | czy nie gubimy oczywistych przypadków |
| ROC-AUC / PR-AUC P-vs-U i kontrolowane (ukryci vs reszta U) | pomocnicze, opisywane jako P-vs-U, a nie wykrywanie choroby |

Losowy punkt odniesienia liczymy na pacjentach. Losowy wynik rekordów zagregowany medianą do pacjenta daje przy `q` = 5% tylko 2,8% zamiast 5%, bo pacjenci z wieloma rekordami rzadziej trafiają na skraj rankingu (`kontrola_skrotow.py`). Agregacja predykcji wchodzi więc w interakcję z liczbą rekordów pacjenta. To kolejny powód, żeby jednostkę treningu i sposób agregacji ustalić przed porównaniami (pytania 8–9 w sekcji 7).

### 3.5 Funkcja celu

```
S(q) = α · RecallHidden@q + (1 − α) · RecallKnown        (α = 1 → czysty RecallHidden@q)
```

`q` to odsetek pacjentów, których realnie da się skierować na obrazowanie. **`q` i `α` zamrażamy przed treningiem.** Strojenie `q` oznaczałoby dobieranie definicji sukcesu pod wynik. „Metryka ważona” z ręczną karą za FP została odrzucona, bo nie da się ustalić kary za pomyłkę, której nie potrafimy rozpoznać.

### 3.6 Zamrożenie przed treningiem

Foldy (już zamrożone), maski ukrywania, metryki z `q` i `α`, scenariusz główny (kontrola czasu, cechy, udział ukrywania), wersje bibliotek i ziarna. Każda późniejsza zmiana trafia do raportu jako zmiana protokołu.

---

## 4. Modelowanie

### 4.1 Strojenie

Optuna w walidacji zagnieżdżonej (3.1), z funkcją celu `RecallHidden@q`. Strojenie na scenariuszu głównym; pozostałe udziały i ziarna oceniają wrażliwość zamrożonej konfiguracji. Wyniki preprocessingu można cache'ować per split, bo nie zależą od hiperparametrów klasyfikatora.

### 4.2 Porównanie modeli

| Model | Rola |
|---|---|
| regresja logistyczna P-vs-U | baseline liniowy |
| Random Forest | baseline drzewiasty |
| XGBoost **albo** LightGBM | jeden boosting (do doinstalowania) |
| **PU Bagging** | właściwa metoda PU |
| modele kontrolne: losowy ranking, wiek i płeć, sam wzorzec braków | punkt odniesienia dla każdego wyniku; te same maski i foldy (kontrola skrótów) |
| **Elkan–Noto** / estymacja class prior | korekta PU; prior szacowany tylko na treningu. Sama korekta dzieli wynik przez stałą `c`, więc nie zmienia rankingu ani `RecallHidden@q` względem modelu bazowego; inne wyniki daje dopiero wariant z ważonym ponownym treningiem. Trzeba wskazać, który porównujemy |

Same baseline'y z KOR jako klasą 0 to klasyfikacja na zaszumionych etykietach, a nie PU learning. Potrzebna jest co najmniej jedna prawdziwa metoda PU.

### 4.3 Analiza pacjentów wysokiego ryzyka

Tylko na **predykcjach out-of-fold**. Dla górnej części rankingu: wynik i percentyl, stabilność między ziarnami i modelami, liczba pomiarów, najważniejsze cechy (SHAP), braki i anomalie w profilu. To lista kandydatów do diagnostyki, nie rozpoznanie.

### 4.4 Artefakty

Tabela foldów, parametry masek, predykcje OOF każdego modelu, wyniki per ziarno z przedziałami ufności (bootstrap grupowy po pacjentach, nie test t na foldach), model card, pełna konfiguracja i wersje bibliotek.

---

## 5. Co jest osiągalne przy obecnych liczbach

Scenariusz: 5 foldów, 40% ukrywania, `q = 5%`. Na fold testowy przypada:

| pacjentów | pozytywnych | ukrytych | pula U | skierowanych przy `q = 5%` |
|---:|---:|---:|---:|---:|
| 8 185 | ~365 | **~146** | ~7 966 | ~399 |

Rozdzielczość `RecallHidden@q` wynosi 1/146 ≈ 0,7 pp. Gdyby ograniczyć dane do wspólnego okna dat, zostałoby ~22 ukrytych na fold i rozdzielczość spadłaby do 4,5 pp. Dlatego wspólne okno nadaje się na analizę wrażliwości (czy wnioski są zgodne co do kierunku), a nie na scenariusz główny.

---

## 6. Prototyp z września: czego się nauczyliśmy

We wrześniu powstała robocza implementacja punktów 3.1–3.5: moduły konfiguracji, danych, foldów, ukrywania, metryk i funkcji celu, około 750 linii plus 17 testów. Leżała na branchu `pu-pipeline-1-5-lk` (commit `2cb6d1a`), który usunięto przy porządkach 04.10.2026. Nowa implementacja powstanie według ustalonego planu. Wnioski warte zachowania:

1. **Przebieg kontrolny z losowym scoringiem** (3000 pacjentów, prawdziwe MICE w foldach) dał `RecallHidden@q` 0,079 przy oczekiwanych 0,05. To mieści się w szumie przy 8–14 ukrytych na fold. Podpróbka może sprawdzić wykonanie kodu, ale nie wystarcza do wiarygodnego porównania jakości modeli, bo przy małej liczbie ukrytych szum dominuje.
2. **Remisy w rankingu** trzeba rozstrzygać deterministycznie (np. po `patient_id`), inaczej wynik zależy od kolejności wierszy.
3. **Fold bez ukrytych ma zwracać brak wartości, nie zero**, bo uśrednianie zer zaniża wynik.
4. **Wszystkie decyzje etapu 0 jako parametry konfiguracji**, a nie logika w kodzie. Zmiana decyzji kosztuje wtedy czas obliczeń, nie przepisywanie.
5. **Gniazdo na kontrolę kohort** w pipeline foldu: funkcja `(train, test) → (train, test, wagi)` wpinana między podział a scaler. Przy decyzji „bez korekty” zostaje puste, ale pozwala później dodać np. ważenie po czasie bez przebudowy.
6. Prototyp liczył **własny podział** stratyfikowany tylko po etykiecie. Nowa implementacja ma czytać zamrożony przydział z `3-sgkf-split/sgkf/results/pacjent_fold.csv` przez `podzial.wczytaj_podzial()`.

---

## 7. Do ustalenia z prowadzącym

1. **Skąd pochodzi kohorta KOR?** Profil laboratoryjny (wśród zmierzonych wyników CRP > 5 mg/l w 61%, WBC > 10 G/l w 37%) wskazuje na pacjentów szpitalnych z lat 2020–2021, a nie na populację ogólną. Od tego zależy interpretacja klasy U.
2. **Pochodzenie wyników NEURO i cel predykcji:** czy badania wykonano przed rozpoznaniem tętniaka, w trakcie hospitalizacji, w której go rozpoznano, czy po leczeniu? Od tego zależy, czy badamy wcześniejsze wykrywanie choroby, czy rozpoznawanie profilu pacjentów już leczonych. Bez odpowiedzi bezpieczny zakres to eksploracyjny ranking PU w dostępnych kohortach.
3. **Rozjazd czasowy:** epokę widać w wynikach chorych z AUC 0,83. Czy przyjmujemy „bez korekty, z analizą wrażliwości we wspólnym oknie dat”?
4. **NRBC i %NRBC:** wykluczyć z modelowania (podpis imputacji, niska wartość kliniczna u dorosłych) czy zostawić i kontrolować?
5. Wartość `q`: z jakiej przepustowości diagnostyki obrazowej ją wyprowadzić?
6. Czy 40% jako główny udział ukrywania jest uzasadnione, skoro zmienia proporcję klas widzianą przez model?
7. Stratyfikacja foldów po prawdziwym statusie pacjenta (element konstrukcji benchmarku, model go nie widzi): akceptowalna?
8. Kolejność: imputacja rekordów, potem agregacja do pacjenta (zgodne z benchmarkiem imputacji), czy odwrotnie (taniej, mniej braków)?
9. Jednostka treningu: jeden profil na pacjenta czy rekordy tygodniowe z wagami 1/n?
10. Które metody PU: PU Bagging i Elkan–Noto wystarczą?
11. **Walidacja:** czy jest szansa na niezależną kohortę zebraną w podobny sposób albo na kliniczną weryfikację części wskazań?
12. **Czyszczenie z etapu 1:** zostawiamy reguły ustalone osobno dla kohort jako ograniczenie, czy ujednolicamy je przy ponownym zamrożeniu danych?

---

## 8. Ograniczenia, które trafią do raportu niezależnie od wyników

1. Nie mierzymy rzeczywistego odsetka fałszywych alarmów w KOR. Ukrywanie testuje tylko odzyskiwanie znanych przypadków.
2. Założenie SCAR jest wątpliwe, bo NEURO to chorzy objawowi.
3. ROC-AUC i PR-AUC z KOR jako klasą negatywną to metryki P-vs-U.
4. Kohorty są rozdzielone czasowo i źródłowo (KOR 2020–2021, NEURO 2000–2024); model może rozpoznawać epokę. Decyzja: bez korekty, opisane wprost.
5. Brak daty diagnozy ogranicza interpretację predykcyjną.
6. Przy założeniu hospitalizacji model rozpoznaje profil pacjenta hospitalizowanego, a nie ryzyko przed rozpoznaniem; nie wolno go opisywać jako modelu przesiewowego.
7. Asymetria braków: MICE odtwarza 26% profilu NEURO wobec 12% KOR. Do tego czyszczenie z etapu 1 ustawiano osobno dla kohort i na całych danych przed podziałem: kwantyle usunęły 3,2% rekordów NEURO i 0,8% KOR, a limit historii wynosi 11 rekordów w KOR i 38 w NEURO.
8. `RecallHidden@q` mierzy odzyskiwanie znanych chorych w tym zbiorze. Ukryci zachowują cechy kohorty, więc część wyniku dają skróty (sam wzorzec braków: około 52% w górnych 5%). Skuteczność wśród niewykrytych chorych KOR wymagałaby niezależnej walidacji.
9. Model daje ranking, a nie prawdopodobieństwo tętniaka. Ani metoda PU, ani kalibracja nie usuwają skutków doboru kohort.
