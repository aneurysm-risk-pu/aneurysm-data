# Plan modelowania Positive-Unlabeled — od SGKF do wyników (propozycja)

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Status:** **propozycja do uzgodnienia przez zespół i prowadzącego.** Dokument łączy wcześniejsze ustalenia z planem wykonania etapu 4. Wskazane niżej wartości startowe są rekomendacjami, a nie potwierdzonymi decyzjami zespołu. Opis przyszłego kodu nie oznacza, że został on już zaimplementowany.
**Stan na:** 18.10.2026. Zachowano decyzje dotyczące danych z 04.10 oraz uzupełnienia z weryfikacji 16–17.10.2026 (`1026_SPOTKANIE_PROWADZACY.md`, sekcja 9); dodano szczegółową kolejność dalszych prac, objaśnienia i kryteria ukończenia. 31.10.2026: podział przebudowany na `StratifiedGroupKFold`, a liczby zależne od foldów przeliczone (`1026_SPOTKANIE_PROWADZACY.md`, sekcja 9.1).

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
| q / RecallHidden@q / Lift@q | q to część puli wybierana z góry rankingu, np. 5%. RecallHidden mówi, jaką część ukrytych znaleźliśmy. Lift porównuje to z losowaniem; przykład w sekcji 13 |
| AUC | W dalszym tekście samo AUC oznacza ROC-AUC: 0,5 to losowa kolejność, 1 idealne rozdzielenie wskazanych grup. W diagnostyce bywają to kohorty albo okresy, nie status choroby. Odrębną miarę PR-AUC wyjaśnia sekcja 13.3 |
| SCAR | Założenie, że znani chorzy zostali oznaczeni losowo spośród wszystkich chorych, niezależnie od ich cech. W naszych danych nie możemy go po prostu przyjąć |
| Wyciek informacji | Wykorzystanie przy uczeniu informacji z części, która miała służyć do niezależnej oceny. Może dotyczyć także imputera i scalera |
| Skrót | Łatwa informacja pozwalająca rozpoznać kohortę, np. zestaw zleconych badań. Dobry wynik dzięki niej nie dowodzi rozpoznania choroby |
| Analiza wrażliwości | Powtórzenie porównania po ustalonej zmianie, np. ograniczeniu okresu. Sprawdza, od czego zależy wniosek |
| Zamrożenie protokołu | Zapisanie zasad przed porównaniem modeli. Późniejsza zmiana jest dozwolona, ale otrzymuje nową wersję i uzasadnienie |

---

## 1. Na czym polega zadanie i dlaczego ocena jest nietypowa

Mamy klasę **P** (NEURO, potwierdzony tętniak) i klasę **U** (KOR: w danych nie ma u nich rozpoznania tętniaka, ale nie wiemy, czy i u kogo wykonano obrazowanie; profil laboratoryjny sugeruje pacjentów szpitalnych, co wymaga potwierdzenia, sekcja 3.0). `label = 0` znaczy „nieoznaczony”, a nie „zdrowy”. Konsekwencje:

1. **Nie policzymy rzeczywistych fałszywych alarmów w KOR.** Wysoki wynik pacjenta KOR może być błędem albo wskazaniem niewykrytego chorego. Bez sprawdzenia jego statusu nie potrafimy rozstrzygnąć, czy jest to „false positive”.
2. **Accuracy jest bezużyteczna.** Przy 4,45% pozytywnych model mówiący zawsze „0” ma 95,5% trafności.
3. **Oceniamy kontrolowanym testem odzyskiwania** (pomysł prowadzącego). Część znanych chorych ukrywamy w puli U i sprawdzamy, czy model wypchnie ich na górę rankingu. Ten test nie usuwa skrótów. Ukryci zachowują cechy kohorty NEURO, a model na samym wzorcu braków odzyskuje około 52% z nich w górnych 5% w wariancie rekordowym i 56% w pacjentowym (sekcje 3.0 i 14.1). Dlatego każdy wynik porównujemy z modelami kontrolnymi na tych samych maskach.

Cały protokół podporządkowany jest pytaniu: *czy wynik mówi coś o tętniakach, czy o tym, jak powstały dane?*

---

## 2. Wejście: co jest gotowe

| Element | Gdzie | Stan |
|---|---|---|
| dane po decyzjach z 04.10.2026 (38 cech, przed imputacją) | `data/processed/aneurysm_sgkf_input.csv` (część A etapu 3) | **gotowe** |
| przydział pacjent → fold (`StratifiedGroupKFold`, 5 foldów, stratyfikacja etykieta × okno × liczba rekordów) | `3-sgkf-split/sgkf/results/pacjent_fold.csv` | **zamrożone** |
| MICE wewnątrz foldu (ExtraTrees 7/78/10) | `3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py` | gotowe |
| zaimputowane foldy zewnętrzne (38 cech) | `3-sgkf-split/sgkf/results/foldy_imputowane_38/` (lokalnie, poza gitem) | **policzone**, odtwarzalne jednym poleceniem |

**Zasada nadrzędna:** foldy zewnętrzne wyłącznie z `podzial.wczytaj_podzial()`, nigdy liczone od nowa w etapie 4 dla tej samej wersji danych. Każde dopasowanie (scaler, imputer, selekcja cech, model, próg uczony na etykietach) odbywa się tylko na części treningowej. Wybór top-q z rankingu testowego realizuje ustalony wcześniej limit miejsc, bez dobierania q na podstawie etykiet testowych (sekcja 13).

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
| kohorty rozdzielne na cechach po imputacji z AUC **0,93** (pacjenci), a we wspólnym oknie dat z AUC 0,78 (bez imputacji 0,825) | wysokie AUC P-vs-U **nie jest** dowodem wykrywania tętniaka; raportujemy je wyłącznie jako metrykę P-vs-U |
| sam wzorzec braków (które badania zlecono) odróżnia kohorty z AUC **0,82** | analiza wrażliwości na cechach z niskim odsetkiem braków w obu kohortach (morfologia, wiek, płeć); porównanie z modelem na pełnym zestawie |
| **ukryci chorzy zachowują cechy kohorty**: model na samym wzorcu braków odzyskuje **52%** ukrytych w górnych 5% rankingu; losowy ranking 4% (oczekiwane 5%), sama demografia 12%, 38 cech 66–69% (`3-sgkf-split/sgkf/kontrola_skrotow.py`, protokół orientacyjny) | `RecallHidden@q` raportujemy zawsze obok modeli kontrolnych na tych samych maskach; liczy się przewaga nad nimi, a nie sama wartość; wynik opisujemy jako odzyskiwanie znanych chorych w tym zbiorze |
| **podpis imputacji NRBC / %NRBC**: zmierzone to w 78% dokładne zera, uzupełnione nigdy nie są zerem; NRBC brakuje w 48% rekordów NEURO i 3% KOR, więc 63% rekordów z uzupełnionym NRBC to NEURO (w całych danych 8,5%); samo NRBC po imputacji odróżnia kohorty z AUC 0,73, tak jak sam wskaźnik braku | przed modelowaniem rozstrzygnąć: wykluczyć NRBC i %NRBC (niska wartość kliniczna u dorosłych) albo sprawdzić ważność tych cech w modelu; inaczej model wykorzysta informację o braku badania |
| przy brakach całymi panelami (koagulogram, glukoza: ~50% rekordów) MICE jest niewiele lepsze od średniej z treningu (błąd 98% błędu średniej w KOR, 89% w NEURO), a uzupełnienia skupiają się wokół średniej; przy skośnych rozkładach są wyższe od typowej zmierzonej wartości (CRP, GLU) | ważność tych cech interpretować ostrożnie; rozważyć wskaźniki braków jako jawną informację zamiast ukrytego śladu |
| epoka rozpoznawalna w wynikach NEURO z AUC **0,83** (po imputacji); przed imputacją, w porównaniach na identycznych rekordach danego kompletnego panelu, wzorzec braków daje 0,77–0,78, a wartości panelu 0,57–0,65; różne panele mają różne liczebności | analiza wrażliwości we wspólnym oknie dat (patrz wyżej) i na wartościach zmierzonych |
| KOR ma profil szpitalny (CRP > 5 mg/l w 61% zmierzonych wyników; CRP zmierzono w 81% rekordów KOR) | interpretacja klasy U i zastosowania „przesiewowego” zależy od odpowiedzi prowadzącego, skąd pochodzi KOR |

**Zostaje do potwierdzenia lub decyzji przy planie:**

| Kwestia | Stan | Wpływ |
|---|---|---|
| pochodzenie kohorty KOR | profil laboratoryjny wskazuje na pacjentów szpitalnych z lat 2020–2021, a nie populację ogólną | interpretacja klasy U; sens zastosowania przesiewowego |
| pochodzenie wyników NEURO i cel predykcji | założenie robocze: hospitalizacja (mediana rozpiętości badań 35 dni u 1 215 pacjentów z >1 rekordem); nie wiadomo, czy badania są sprzed rozpoznania, z jego czasu, czy po leczeniu | jeśli hospitalizacja się potwierdzi, model może opisywać profil pacjentów hospitalizowanych; obecnie zakres to eksploracyjny ranking PU w dostępnych kohortach |
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

Losowy punkt odniesienia liczymy na pacjentach. Losowy wynik rekordów zagregowany medianą do pacjenta daje przy `q` = 5% tylko 3,2%, a losowy ranking pacjentów 4,2% (oczekiwane 5%), bo pacjenci z wieloma rekordami rzadziej trafiają na skraj rankingu (`kontrola_skrotow.py`). Agregacja predykcji wchodzi więc w interakcję z liczbą rekordów pacjenta. To kolejny powód, żeby jednostkę treningu i sposób agregacji ustalić przed porównaniami (pytania 8–9 w sekcji 7).

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

Optuna w walidacji zagnieżdżonej (3.1), z funkcją celu `RecallHidden@q`. Strojenie dotyczy scenariusza głównego i ustalonego zestawu masek, proponowanego jako ziarna 0–4. Jedna Optuna na rodzinę modelu i outer-fold wybiera parametry według średniej wyników inner dla tych masek. Inne udziały ukrywania i dodatkowe ziarna służą analizie wrażliwości; szczegóły w sekcji 15. Wyniki preprocessingu można cache'ować per podział, jeżeli nie zależą od maski ani hiperparametrów klasyfikatora.

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
2. Założenie SCAR jest wątpliwe: nie wiemy, czy oznaczeni pacjenci NEURO są reprezentatywną, losowo wybraną częścią wszystkich chorych. Hospitalizacja i objawowość mogą wpływać na wybór, lecz wymagają potwierdzenia.
3. ROC-AUC i PR-AUC z KOR jako klasą negatywną to metryki P-vs-U.
4. Kohorty są rozdzielone czasowo i źródłowo (KOR 2020–2021, NEURO 2000–2024); model może rozpoznawać epokę. Decyzja: bez korekty, opisane wprost.
5. Brak daty diagnozy ogranicza interpretację predykcyjną.
6. Przy założeniu hospitalizacji model rozpoznaje profil pacjenta hospitalizowanego, a nie ryzyko przed rozpoznaniem; nie wolno go opisywać jako modelu przesiewowego.
7. Asymetria braków: MICE odtwarza 26% profilu NEURO wobec 12% KOR. Do tego czyszczenie z etapu 1 ustawiano osobno dla kohort i na całych danych przed podziałem: kwantyle usunęły 3,2% rekordów NEURO i 0,8% KOR, a limit historii wynosi 11 rekordów w KOR i 38 w NEURO.
8. `RecallHidden@q` mierzy odzyskiwanie znanych chorych w tym zbiorze. Ukryci zachowują cechy kohorty, więc część wyniku dają skróty (sam wzorzec braków: około 52% w górnych 5%). Skuteczność wśród niewykrytych chorych KOR wymagałaby niezależnej walidacji.
9. Model daje ranking, a nie prawdopodobieństwo tętniaka. Ani metoda PU, ani kalibracja nie usuwają skutków doboru kohort.

---

## 9. Od SGKF do pierwszego wyniku — kolejność prac

**Pierwszy cel etapu 4:** jeden odtwarzalny eksperyment, który dla pięciu zamrożonych foldów ukrywa część znanych chorych, przygotowuje dane bez korzystania z testu, uruchamia modele kontrolne i pierwszą metodę PU oraz zapisuje wyniki na poziomie pacjentów.

Nie wyznaczamy na tym etapie wymaganego wyniku, np. „AUC powyżej 0,9”. Wysoki wynik może pochodzić ze skrótów, a niski może ujawnić rzeczywiste ograniczenie danych. Sukcesem pierwszego przebiegu jest poprawność i możliwość wyjaśnienia, co dokładnie zmierzyliśmy.

| Kolejność | Pakiet z podziału prac na spotkanie | Co robimy | Co musi powstać, żeby przejść dalej |
|---|---|---|---|
| 1 | 0: wspólny stan projektu | Przegląd zmian, wskazanie obowiązującej wersji danych i środowiska | Zespół pracuje na tej samej wersji kodu, danych i bibliotek |
| 2 | 1: protokół oceny | Uzgodnienie celu, jednostki pacjentowej, masek, cech i metryk | Zapisana konfiguracja oraz lista świadomie przyjętych ograniczeń |
| 3 | 2A: pierwszy pipeline | Przygotowanie cech, ukrywanie, predykcje i ocena bez strojenia | Kontrole techniczne przechodzą, wynik można odtworzyć |
| 4 | 3 i 4: pierwsze modele | Kontrole, regresja logistyczna i PU Bagging; potem model drzewiasty | Jedna wspólna tabela porównania i predykcje OOF |
| 5 | 2B: walidacja zagnieżdżona | Dodanie podziałów wewnętrznych i strojenia | Ustawienia dobierane wyłącznie wewnątrz treningu |
| 6 | 5: analizy wrażliwości | Powtórzenie ustalonych porównań po zmianie cech, czasu lub imputacji | Wiadomo, które wnioski są stabilne, a które zależą od wariantu |
| 7 | 6: raport | Zestawienie jakości, niepewności, kosztów i ograniczeń | Wynik zrozumiały i możliwy do odtworzenia przez inną osobę |

Podział pakietu 2 na 2A i 2B jest organizacyjny. Umożliwia uruchomienie prostego eksperymentu bez czekania na pełne strojenie. Nie pozwala stroić na zewnętrznym teście. Historyczne diagnozy i pierwszy przebieg nadal mają charakter eksploracyjny — obecne dane były już wielokrotnie oglądane.

## 10. Co ustalamy wspólnie przed pierwszym porównaniem

### 10.1 Cel i wersja danych

Najpierw zapisujemy jedno zdanie określające zadanie. Przy obecnej wiedzy proponowane brzmienie to: **„Porównujemy metody tworzenia rankingu pacjentów PU przez odzyskiwanie celowo ukrytych, znanych przypadków NEURO w dostępnych kohortach”.** Nie zawiera ono obietnicy przewidywania przyszłego zachorowania ani skuteczności badań przesiewowych.

Jeśli uzyskamy daty rozpoznania i informacje o leczeniu, można opracować zadanie wcześniejszego wykrywania. Wówczas trzeba ustalić moment predykcji i dopuścić tylko wyniki dostępne do tego momentu. Mediana z całej historii, obejmującej późniejsze leczenie, nie byłaby poprawnym wejściem takiego modelu.

Przy historycznym czyszczeniu wybieramy jedną z dwóch dróg i zapisujemy uzasadnienie:

- **Zachowujemy obecną wersję do badań eksploracyjnych.** Korzystamy z istniejącego podziału, jawnie opisując osobne kwantyle, limity 11/38 i utracony rekord po kolizji klucza. To umożliwia kontynuację bez odtwarzania całego etapu 1.
- **Poprawiamy proces i zamrażamy nową wersję danych.** Odtwarzamy odpowiednie kroki od źródła, kontrolujemy liczebności i ponawiamy zależne artefakty etapu 3. Samo obcięcie historii NEURO do 11 rekordów nie usuwa wcześniejszej selekcji kwantylowej ani wszystkich różnic źródłowych.

Zmiana cech wyłącznie w eksperymencie modelowym nie wymaga losowania nowego zewnętrznego przydziału pacjentów. Zmiana wejściowego pliku etapu 3 wymaga jego ponownej kontroli i wersjonowania zgodnie z mechanizmem odcisków.

### 10.2 Proponowane ustawienia startowe i ich uzasadnienie

Poniższa tabela daje zespołowi konkretny punkt wyjścia. **Nie oznacza akceptacji tych ustawień przez zespół lub prowadzącego.** Po uzgodnieniu zapisujemy przy każdym polu wartość, datę i osobę odpowiedzialną za decyzję.

| Kwestia | Propozycja na start | Dlaczego właśnie tak | Co pozostaje do uzgodnienia |
|---|---|---|---|
| Jednostka klasyfikatora | Jeden profil na pacjenta | Każda osoba ma równy udział w treningu i ocenie | Potwierdzenie, czy celem jest ranking osób, a nie pojedynczych tygodni |
| Kolejność przygotowania | Imputacja rekordów w foldzie, potem mediana cech pacjenta | Zachowuje związek z przetestowanym etapem 3 i pozwala wykorzystać zgodne outer-parquet | Agregacja przed MICE jako osobny wariant, nie cicha zmiana |
| Agregacja | Mediana; średnia w analizie wrażliwości | Na obecnych danych mediana słabiej wiąże profil z liczbą badań niż średnia i maksimum | Potwierdzenie reguły; mediana nie rozwiązuje braku daty diagnozy |
| Zestaw cech | 38 jako dotychczasowy wariant odniesienia; rozstrzygnięcie wariantu bez NRBC/%NRBC przed właściwym porównaniem | Zachowujemy pełne wejście projektu i osobno badamy znany ślad braków | Czy scenariuszem głównym będzie 38 czy 36 cech; wyłączenie przed MICE wymaga nowych uzupełnień |
| Imputacja | Zamrożone MICE 7 iteracji / 78 drzew / głębokość 10 | To istniejący, sprawdzony kompromis; nie dokładamy strojenia imputera do pierwszego eksperymentu | Porównanie z prostą medianą, a dla odpowiedniego modelu z obsługą NaN |
| Udział ukrytych P | 40%; dodatkowo 20% i 60% | Kontynuacja prototypu i kontroli z 17.10; pozostawia liczne widoczne P i ukryte przypadki | To parametr symulacji, nie estymacja liczby nierozpoznanych chorych KOR |
| Główne q | 5%; 1% i 10% jako dodatkowe punkty krzywej | Nawiązuje do istniejącej diagnostyki i ograniczonej liczby miejsc w rankingu | Brak potwierdzonej przepustowości klinicznej; 5% jest propozycją badawczą |
| Ziarna maskowania | 0, 1, 2, 3, 4 | Zgodność z kontrolą z 17.10 i sprawdzenie zależności od losowania | Czy zwiększamy liczbę powtórzeń po ocenie kosztu obliczeń |
| Funkcja celu | `RecallHidden@q`, czyli α = 1 | Jedno kryterium jest łatwiejsze do wyjaśnienia; nie wymaga arbitralnego łączenia dwóch miar | `RecallKnown` na start pomocniczo; inne α tylko z ustalonym uzasadnieniem |
| Podział | Obecne 5 foldów zewnętrznych; 3 wewnętrzne przy strojeniu | Powtarzalna ocena tych samych pacjentów i rozdzielenie wyboru ustawień od oceny | Postępowanie przy małych podzbiorach analiz wrażliwości |

**Dlaczego nie dobierać tych wartości po zobaczeniu wyniku?** Zmiana q z 5% na 10% zwiększa liczbę dostępnych miejsc, a więc także szansę odzyskania ukrytych. Lepszy procent nie musi wówczas wynikać z lepszego modelu. Podobnie zmiana udziału ukrywania zmienia zadanie treningowe. Porównujemy modele przy jednakowych warunkach, a osobno pokazujemy wpływ zmiany warunków.

### 10.3 Powiązanie z siedmioma sprawami do omówienia

| Sprawa | Jak zamykamy ją w planie |
|---|---|
| Cel predykcji i pochodzenie danych | Odpowiedzi od prowadzącego albo zapisany ograniczony cel eksploracyjny; sekcja 10.1 |
| Historyczne filtry zależne od kohorty | Wybór wersji danych i jawne ograniczenie lub odtworzenie procesu; sekcja 10.1 |
| Niedomknięty protokół | Uzupełniona tabela 10.2 i szczegóły sekcji 11–13 |
| Wspólny okres | Pełny wariant ograniczony do okna przed dopasowaniem preprocessingu; sekcja 16 |
| Zbyt mocne opisy wcześniejszych wyników | Korzystamy z korekt 16–17.10 i odróżniamy diagnostykę od nowego eksperymentu; sekcje 14 i 17 |
| Kontrola artefaktów i odtwarzalność | Walidacja przy odczycie, manifest, wersje bibliotek i kodu; sekcje 11 i 17 |
| Znaczenie etykiet i wyników | `original_label` odrębnie od `observed_label`; ranking bez twierdzeń o diagnozie KOR; sekcje 12–13 |

## 11. Wspólny pipeline: jak dane przechodzą do modelu

### 11.1 Jedno wejście i jednoznaczny podział

Punktem wyjścia jest `data/processed/aneurysm_sgkf_input.csv`, czyli 78 052 rekordy, 40 924 pacjentów i 38 cech w obecnej wersji. Zewnętrzny przydział odczytujemy przez `podzial.wczytaj_podzial()`. Nie używamy globalnie zaimputowanych plików z etapu 2 ani przypadkowej kopii `*_shortend.csv`.

Przed obliczeniami sprawdzamy nie tylko odcisk wejścia, ale również sam przydział: jeden wpis na pacjenta, pełne pokrycie danych, numery foldów 0–4, zgodność etykiet i rozłączność train/test. Obecne dane są poprawne; dodanie tej kontroli ma zapobiec użyciu niespójnej przyszłej wersji.

### 11.2 Proponowana ścieżka pierwszego eksperymentu

```text
wejście przed imputacją + zamrożony przydział pacjentów
    → wydzielenie treningu i testu danego foldu
    → scaler imputacji i MICE: dopasowanie tylko na rekordach treningu
    → uzupełnienie treningu i transformacja testu, powrót do oryginalnej skali
    → mediana cech osobno dla każdego pacjenta po każdej stronie podziału
    → przypisanie observed_label według wspólnej maski pacjentów
    → ewentualny scaler klasyfikatora: dopasowanie tylko do profili treningu
    → trening modelu na observed_label
    → jeden wynik dla każdego pacjenta testowego
    → ranking puli U, metryki i zapis OOF
```

**Dlaczego jeden profil?** Jeśli pacjent A ma jedno badanie, a B dziesięć, trening na rekordach daje B dziesięć okazji do wpływania na klasyfikator. Jeden profil daje każdemu pacjentowi jeden wiersz. Nie twierdzimy, że usuwa to całą nierówność: MICE nadal uczy się na rekordach, więc częściej badani mają większy udział w dopasowaniu imputera.

**Dlaczego mediana po MICE?** Zachowujemy sposób imputacji sprawdzony w etapie 3. Mediana cech po uzupełnieniu i mediana cech przed uzupełnieniem mogą być różne. To drugie rozwiązanie jest tańsze i może zostawić mniej braków, ale wymaga osobnej oceny; nie dziedziczy automatycznie wyników benchmarku rekordowego.

**Dwa skalowania służą różnym celom.** MinMax w etapie 3 przygotowuje wejście do imputera, po czym jego działanie jest odwracane. Regresja logistyczna może później dostać `StandardScaler` dopasowany do profili pacjentów treningowych, żeby wielkość jednostki laboratoryjnej nie decydowała o sile regularyzacji. Model drzewiasty nie potrzebuje tego drugiego kroku. Zasada dopasowania transformacji wyłącznie na treningu wynika z ochrony przed wyciekiem; opisuje ją też [dokumentacja scikit-learn 1.6](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).

Przykład: mediana cechy z wyników 10, 12 i 80 wynosi 12, a średnia 34. Mediana ogranicza wpływ skrajnego wyniku, ale może również ukryć istotną zmianę stanu pacjenta. Dlatego jest propozycją opisowego profilu, a nie dowodem najlepszego wyboru klinicznego.

### 11.3 Kiedy wolno wykorzystać już zapisane foldy

W pierwszym eksperymencie bez strojenia można użyć `foldy_imputowane_38/`, jeśli dane, przydział, cechy i parametry odpowiadają manifestowi przebiegu. Następnie agregujemy cechy do pacjentów i trenujemy ustalony z góry model. Dla innego wejścia MICE, np. bez NRBC/%NRBC, potrzebne jest nowe dopasowanie.

Nie wolno użyć tych outer-parquet do strojenia przez ich późniejszy podział na inner-train i inner-validation. Imputer dopasowany do całego outer-train znałby już cechy pacjentów inner-validation. Poprawny przebieg strojenia opisuje sekcja 15.

### 11.4 Umowa między modułami

| Część | Co otrzymuje | Co zwraca |
|---|---|---|
| Odczyt i podział | Dane, konfigurację i zapisany przydział | Rozłączne listy pacjentów oraz odpowiadające im rekordy |
| Preprocessing | Cechy train i validation/test; bez etykiet kontrolnych | Przygotowane cechy, obiekty transformacji i manifest |
| Maskowanie | Listę pacjentów, `original_label`, udział i ziarno | `observed_label`, `is_hidden` oraz odtwarzalną maskę |
| Model | Cechy i `observed_label` treningu | Wyniki dla przekazanych pacjentów |
| Ocena | Wyniki, metadane i maskę danego foldu | Metryki wraz z licznikami, mianownikami i miejscami w rankingu |

Identyfikatory, data, fold, `original_label` i `is_hidden` pozostają w metadanych. Model i imputer nie dostają ich jako cech. Wiek i płeć mogą być cechami; płeć w obecnych danych jest kompletna i stała wewnątrz pacjenta.

**Gotowe, gdy:** dwa różne modele mogą korzystać z tego samego modułu przygotowania i oceny, bez własnych wersji podziału lub metryk.

## 12. Ukrywanie chorych — co dokładnie widzi model

Ukrywanie zmienia etykietę dostępną modelowi, a nie wyniki badań i nie miejsce pacjenta w foldzie. Losujemy pacjentów, ponieważ jedna osoba nie może być jednocześnie widoczna i ukryta w różnych tygodniach.

| Przypadek | `original_label` | `observed_label` | `is_hidden` | Rola |
|---|---:|---:|---|---|
| Pacjent KOR | 0 | 0 | false | Nieoznaczony, rzeczywisty status choroby nieznany |
| Widoczny pacjent NEURO | 1 | 1 | false | Znany pozytywny widoczny w danym scenariuszu |
| Ukryty pacjent NEURO | 1 | 0 | true | Dla modelu U; dla funkcji oceniającej znany cel odzyskania |

**Proponowany sposób wykonania:**

1. Dla ustalonej wersji danych i ziarna tworzymy wspólną listę pacjentów P. Sortujemy ją i losujemy bez zwracania `round(h × liczba P)` osób, gdzie h to udział ukrywania. Zapisujemy algorytm, regułę zaokrąglenia i wynik losowania.
2. Ta sama lista obowiązuje wszystkie modele, foldy i podziały wewnętrzne. W konkretnym train/validation/test bierzemy przecięcie listy z pacjentami tej części. Faktyczny udział w pojedynczym foldzie może się lekko różnić od h; zapisujemy go.
3. W treningu ukryci dostają `observed_label = 0`. Widoczni P dostają 1. Zmiana musi nastąpić przed trenowaniem klasyfikatora i przed tworzeniem podzbiorów PU Bagging.
4. W validation/test cechy wszystkich pacjentów są przetwarzane bez dopasowywania na nich transformacji. Ocena rankingu obejmuje KOR oraz ukrytych NEURO. Widoczni NEURO są poza tą pulą i mogą służyć do pomocniczego `RecallKnown`.
5. `original_label` i `is_hidden` są dostępne funkcji oceniającej, ale nie modelowi. Prawdziwe oznaczenia wolno wykorzystać do technicznej konstrukcji benchmarku i stratyfikacji; trzeba jawnie odnotować ten dostęp.

**Dlaczego ukrywamy także w treningu?** Gdyby ukrywać wyłącznie w teście, model uczyłby się innego zadania: wszystkich znanych P przeciw KOR. Chcemy sprawdzić sytuację, w której w treningowej puli U również znajdują się celowo ukryci znani chorzy.

MICE nie wykorzystuje tych etykiet, więc może korzystać z cech wszystkich pacjentów treningowych, także ukrytych. Nie wolno natomiast dopasowywać go do cech pacjentów testowych pod pretekstem, że są nieoznaczeni. Wybrany protokół ocenia nowych pacjentów; nie jest eksperymentem uczącym się dodatkowo na testowej puli U.

Zapis listy pacjentów jest niewielki i ułatwia kontrolę zgodności. Samo podanie „seed 42” nie wystarcza, jeśli inna kolejność pacjentów lub inna implementacja losowania zmieni wybór. Własnego `hash()` Pythona nie używamy jako trwałego identyfikatora maski.

**Gotowe, gdy:** każdy rekord pacjenta ma ten sam status, modele używają identycznej maski, a przy ustalonej masce i podziale zmiana etykiet przekazanych do funkcji oceniającej test nie wpływa na wytrenowany model.

## 13. Jak liczymy wynik i jak go czytać

### 13.1 Główna miara na prostym przykładzie

Wyobraźmy sobie pulę oceny liczącą 1000 pacjentów: KOR oraz celowo ukrytych NEURO. Wśród nich jest 40 ukrytych znanych chorych. Przy q = 5% wybieramy 50 najwyżej ocenionych osób. Jeśli jest wśród nich 20 ukrytych:

- `RecallHidden@5% = 20 / 40 = 50%`: odzyskaliśmy połowę ukrytych;
- udział ukrytych w wybranej grupie wynosi `20 / 50 = 40%`: pozostałych 30 osób nie wolno automatycznie uznać za fałszywe alarmy;
- losowy wybór 5% puli odzyskuje przeciętnie około 5% ukrytych, więc `Lift@5%` wynosi tutaj około `50% / 5% = 10`.

To trzy różne mianowniki. Nie należy zamieniać czułości na ukrytych z udziałem ukrytych w wybranej grupie ani nazywać tego udziału rzeczywistą precision dla tętniaka.

### 13.2 Reguły wspólne dla wszystkich modeli

W każdym testowym foldzie osobno:

1. tworzymy pulę U: KOR i ukryci NEURO z tego foldu;
2. wyliczamy liczbę miejsc `k = ceil(q × liczba pacjentów w puli)`;
3. sortujemy wynik malejąco, remisy rozstrzygamy według ustalonego porządku `patient_id`;
4. wybieramy dokładnie k osób i liczymy odnalezionych ukrytych;
5. zapisujemy licznik, liczbę wszystkich ukrytych, wielkość puli i k obok procentu.

Losowy ranking nadaje jedną losową liczbę każdemu pacjentowi. Jego dokładna wartość oczekiwana to `k / wielkość puli`, bliska q; pojedyncze losowanie nie musi trafić dokładnie w q. Losowanie liczb dla rekordów i późniejsza mediana nie daje równomiernego losowego rankingu pacjentów — potwierdza to kontrola z 17.10.

**Dlaczego osobno w każdym foldzie?** Pięć modeli zewnętrznych może nadawać wynikom różne skale. Nie łączymy ich surowych score w jeden ranking top-q. W obrębie foldu można zapisać percentyl, czyli pozycję względem innych osób w tej puli. Oznacza on względne miejsce, nie prawdopodobieństwo choroby.

Wybór top-q na podstawie testowych wyników jest realizacją z góry ustalonego limitu miejsc, bez oglądania etykiet. Nie jest strojeniem progu na testowych rozpoznaniach. Inny, stały próg klasyfikacji lub kalibrację można dobierać tylko na treningu/walidacji.

### 13.3 Łączenie foldów, ziarna i dodatkowe metryki

Dla każdego ziarna główny wynik łączny to **suma odnalezionych ukrytych ze wszystkich foldów podzielona przez sumę ukrytych w tych foldach**. Dzięki temu każdy ukryty pacjent ma jeden głos. Obok pokazujemy wartości per fold, żeby widzieć niestabilność. Następnie zestawiamy wyniki ziaren; nie traktujemy powtarzających się pacjentów jako nowych niezależnych osób.

`RecallKnown` proponujemy raportować pomocniczo: odsetek widocznych P z testu, które osiągają regułę odcięcia wyznaczoną przez top-q puli U danego foldu. Obsługa remisów musi odpowiadać regule rankingu. Ta miara nie zwiększa liczby miejsc w głównej puli i nie jest czułością dla wszystkich chorych populacji.

ROC-AUC opisuje, jak często losowo wybrany oznaczony pozytywny ma wyższy wynik niż losowo wybrana osoba z grupy porównawczej; 0,5 odpowiada losowej kolejności, a 1 idealnej w tej konkretnej definicji etykiet. PR-AUC podsumowuje zależność między odzyskiwaniem pozytywnych a ich udziałem w kolejnych częściach rankingu. W raporcie wskazujemy też dokładny sposób liczenia, np. average precision, bo nie każda implementacja pola pod krzywą PR liczy to identycznie.

Obie miary opisują pomocniczo rozróżnianie P i U według jawnie podanej definicji etykiety. PR-AUC zależy również od udziałów klas, dlatego wartości z różnych scenariuszy ukrywania lub podzbiorów nie oznaczają automatycznie lepszego modelu. Accuracy nie jest kryterium wyboru.

**Gotowe, gdy:** ręcznie policzony przykład daje ten sam wynik co kod; remisy są odtwarzalne; fold bez ukrytych zwraca brak miary, nie zero; wszystkie modele mają identyczne pule oceny.

## 14. Pierwsze modele — co porównujemy i po co

### 14.1 Kontrole obowiązkowe

| Model kontrolny | Jakie informacje otrzymuje | Na jakie pytanie odpowiada |
|---|---|---|
| Losowy ranking pacjentów | Żadnych cech ani etykiet treningowych | Czy przewaga nad losowaniem jest widoczna i czy metryka działa poprawnie? |
| Wiek i płeć | Tylko profil demograficzny, `observed_label` treningu | Ile odzyskiwania dają proste różnice demograficzne? |
| Wzorzec braków | Udział brakujących rekordów każdej cechy u pacjenta, liczony **przed imputacją** | Ile wyniku daje sam sposób zlecania i zbierania badań? |

W wariancie pacjentowym kontrola braków powinna również uczyć się na jednym profilu na osobę i korzystać z tej samej listy dopuszczonych cech. Nie zastępujemy jej kontrolą rekordową, bo te zadania dają różne wyniki.

**Jaki algorytm dla kontroli?** Proponujemy kontrolę demografii i braków tą samą rodziną modelu, którą porównujemy na pełnych cechach: np. regresja na brakach obok regresji pełnej, a las na brakach obok lasu pełnego. W pierwszym przebiegu mają z góry ustalone ustawienia; w porównaniu strojonym otrzymują ten sam budżet i wewnętrzne podziały. Dodatkowo zachowujemy HistGradientBoosting z parametrami dotychczasowej diagnostyki jako stały punkt odniesienia dla braków. Porównanie różnych rodzin jest informacyjne, ale nie izoluje samego wkładu wartości badań.

Kontrola orientacyjna z 17.10, przeliczona 31.10 na podziale SGKF, przy 40% ukrywania, pięciu ziarnach i pięciu foldach, dała `RecallHidden@5%`: 4,2% dla losowego rankingu pacjentów, 51,7% dla rekordowego modelu na brakach i 56,1% dla pacjentowego profilu braków. Model pacjentowy na medianach 38 cech z NaN osiągnął 68,5%. Źródło: `3-sgkf-split/sgkf/results/kontrola_skrotow.json`. To wcześniejsza diagnostyka, **nie wynik docelowego protokołu** ani poziom, który przyszły model musi przekroczyć o ustaloną liczbę punktów.

Przewaga nad kontrolą braków sugeruje dodatkową użyteczność cech w tym benchmarku. Nie mierzy „czystego sygnału choroby” — pełny model nadal może korzystać z innych różnic między kohortami.

### 14.2 Kolejność wdrażania modeli właściwych

**Najmniejszy sensowny zestaw:** kontrole, regresja logistyczna i PU Bagging. Następnie dokładamy jeden model drzewiasty. Przed uruchomieniem zapisujemy wszystkie parametry pierwszego przebiegu; nie poprawiamy ich po obejrzeniu outer-test.

| Model | Proste objaśnienie | Uzasadnienie miejsca w planie |
|---|---|---|
| Regresja logistyczna P-vs-U | Łączy cechy w jeden wynik przy pomocy wyuczonych współczynników. Regularyzacja ogranicza nadmiernie duże współczynniki | Szybki i czytelny baseline; odpowiedni do sprawdzenia całego przepływu. KOR jako 0 oznacza tu etykietę roboczą, nie zdrowie |
| PU Bagging | Wiele razy uczy model na widocznych P i losowanej części U, po czym uśrednia wyniki | Pierwsza metoda jawnie uwzględniająca nieoznaczoną pulę przez wielokrotne podpróbki; naturalne porównanie z pojedynczym baseline'em |
| Random Forest | Wiele drzew dzieli dane według warunków na cechach i łączy swoje wyniki | Dodaje zależności nieliniowe bez instalowania nowej biblioteki; porównamy, czy złożoność przynosi stabilną korzyść |
| Jeden boosting, jeśli potrzebny | Kolejne drzewa poprawiają wcześniejsze błędy | Rozszerzenie po sprawdzeniu podstaw. HistGradientBoosting jest już używany w diagnostyce; wybór XGBoost/LightGBM wymaga uzasadnienia i zapisu zależności |

W PU Bagging dobieramy widoczne P i próbkujemy U **wyłącznie z treningu danego podziału**. Nie uczymy żadnego modelu na osobach testowych. Przykładowa propozycja techniczna to 50 powtórzeń i podzbiór U o wielkości liczby widocznych P, z prostą regresją logistyczną jako modelem składowym. Liczbę powtórzeń, wielkość i sposób losowania U zapisujemy przed porównaniem. Nie są one estymatą rzeczywistego udziału chorych.

Pomysł wielokrotnego porównywania P z podpróbkami U pochodzi z pracy [Mordelet i Vert, „A bagging SVM to learn from positive and unlabeled examples”](https://arxiv.org/abs/1010.0772). Oryginalna praca dotyczy SVM; wariant z regresją jest propozycją implementacyjną naszego eksperymentu, nie odtworzeniem wyników tej publikacji. Uśrednianie nie gwarantuje usunięcia skrótów kohortowych.

Korektę Elkan–Noto zostawiamy jako dalszy wariant po ustaleniu założeń. Proste dzielenie score przez stałą nie zmienia kolejności wewnątrz foldu, więc nie da samodzielnej poprawy `RecallHidden@q`. Wariant z ważonym ponownym treningiem jest inną metodą i wymaga osobnej specyfikacji. Nie znamy class prior, czyli rzeczywistego udziału chorych w KOR; 4,45% oznaczonych pacjentów w połączonych danych nie jest jego estymatą. [Elkan i Noto, 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf).

### 14.3 Pierwszy przebieg bez strojenia

Wykonujemy pięć zewnętrznych foldów dla każdego ustalonego ziarna ukrywania i modelu. Losowość samego modelu zapisujemy oddzielnie od losowości masek; na start można ustalić jeden seed algorytmu, a później sprawdzić jego stabilność. Podpróbka lub dane syntetyczne służą do sprawdzenia wykonania, pełny zbiór do porównania jakości.

Nie testujemy od razu dziesiątek kombinacji modeli, imputacji, q i cech. Najpierw porównujemy ustalony mały zestaw. Tabela wyników zawiera również czas obliczeń, liczbę ukrytych i wielkość puli, żeby nie mylić korzyści jakościowej z innym zadaniem lub kosztem.

**Gotowe, gdy:** mamy kompletną tabelę kontroli, baseline'u i pierwszej metody PU, ich predykcje OOF i konfiguracje, a inny członek zespołu potrafi odtworzyć przebieg.

## 15. Strojenie: jak poprawiać ustawienia bez uczenia się testu

### 15.1 Po co drugi poziom podziału

Model może wyglądać dobrze również dlatego, że przetestowaliśmy wiele ustawień i wybraliśmy szczęśliwy wynik. Dlatego część treningu służy do wyboru ustawień, a odłożony outer-test dopiero do ich oceny. Jest to cel zagnieżdżonej walidacji opisany w [przykładzie scikit-learn 1.6](https://scikit-learn.org/1.6/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

W naszym projekcie dla testowanego foldu nr 0 pozostałe foldy 1–4 tworzą outer-train. Tylko wewnątrz tej grupy powstają trzy nowe podziały pacjentów. Imputer w pierwszym inner-train nie może oglądać pacjentów z jego inner-validation ani z foldu nr 0. Dotyczy to również scalera, wyboru cech i ewentualnego progu uczonego na etykietach.

```text
dla każdego z 5 zapisanych foldów zewnętrznych:
    odłóż outer-test
    w outer-train utwórz 3 inner foldy pacjentów
    dla każdego inner foldu:
        dopasuj preprocessing wyłącznie na inner-train
        przygotuj inner-validation bez ponownego fit
        dla porównywanych ustawień i ustalonych masek:
            trenuj na observed_label inner-train
            oceń odzyskiwanie ukrytych w inner-validation
    wybierz ustawienia według ustalonej metryki inner
    dopasuj preprocessing i model na całym outer-train
    oblicz i zapisz wynik outer-test
```

Agregacja medianą wewnątrz jednego pacjenta nie uczy parametrów wspólnych dla populacji. Można ją wykonać oddzielnie po obu stronach podziału. Nie usprawiedliwia to jednak MICE dopasowanego wcześniej do wszystkich pacjentów outer-train.

### 15.2 Zakres strojenia i stabilność masek

Przed startem zapisujemy zakres parametrów każdego modelu, liczbę prób Optuny, ziarna i limit czasu. Proponowany start to nieduża liczba, np. 20 prób na rodzinę modelu i outer-fold, przy ustalonym zestawie masek wewnętrznych. MICE pozostaje na tym etapie zamrożone. Konkretne zakresy wynikają z wybranych modeli i muszą znaleźć się w konfiguracji przed uruchomieniem.

Każde ustawienie oceniamy na tych samych wewnętrznych pacjentach i maskach. Losowanie nowej maski dla każdej próby utrudniałoby rozróżnienie poprawy modelu od łatwiejszego losowania chorych. Te same osoby między ziarnami nie tworzą niezależnych prób.

Proponowana funkcja celu jednej Optuny działa następująco: dla każdego z ziaren 0–4 łączymy liczniki i mianowniki z trzech inner-validation, otrzymując jeden `RecallHidden@q` dla tego ziarna, a następnie uśredniamy pięć wyników. Wybrane parametry są wspólne dla tych masek w danym outer-foldzie. Potem dla każdego ziarna osobno trenujemy model na całym outer-train z odpowiadającym mu `observed_label` i oceniamy outer-test. Nie uruchamiamy pięciu niezależnych poszukiwań tylko po to, aby każda maska dostała inne najlepsze ustawienia.

Lista rodzin modeli także jest elementem protokołu. Jeśli chcemy ocenić automatyczny wybór „logistyczna czy las”, wybór rodziny powinien odbywać się w pętli wewnętrznej. Gdy jedynie pokazujemy kilka uprzednio zaplanowanych porównań OOF, raportujemy wszystkie; najwyższy z nich nie staje się przez to niezależnie potwierdzonym wynikiem zwycięzcy.

Po obejrzeniu outer-test nie zmieniamy q ani ustawień tego samego eksperymentu. Nowy pomysł można sprawdzić, lecz otrzymuje nową wersję protokołu i status dalszej eksploracji. Samo nested CV nie cofa wcześniejszego oglądania danych i wyborów z etapów 1–3.

### 15.3 Jak ograniczamy koszt: cache, czyli zapis gotowego przygotowania

Preprocessing można policzyć raz dla danego podziału, a potem używać przy wielu ustawieniach klasyfikatora. Jest to poprawne, jeśli przygotowanie nie korzysta z etykiet ani wag zależnych od maski. Dla obecnego MICE ta niezależność zachodzi. Jeśli później dodamy etykietową selekcję cech, musi ona być dopasowywana i wersjonowana zgodnie z tym, od czego zależy.

Klucz cache obejmuje odciski danych i przydziału, dokładną listę train/validation/test, listę i kolejność cech, zakres dat, kolejność agregacji i imputacji, parametry, losowość oraz wersje bibliotek. To chroni przed użyciem gotowego wyniku dla innego eksperymentu. Katalog zapisujemy jako ukończony dopiero po zapisaniu wszystkich plików i manifestu; odczyt sprawdza ich zgodność.

Przy schemacie 5 × 3 potrzebujemy 15 dopasowań preprocessingu do podziałów wewnętrznych i 5 do pełnego outer-train na jeden wariant. Wcześniejsze około 3 minut MICE na fold daje orientację kosztu, nie gwarancję czasu. Nowy zestaw cech, inna kolejność agregacji lub okno dat wymagają własnego cache. Nie mnożymy natomiast dopasowań MICE przez wszystkie próby Optuny ani przez maski, jeśli preprocessing pozostaje od nich niezależny.

**Gotowe, gdy:** outer-test nie uczestniczy w wyborze ustawień, a test techniczny potwierdza, że zmiana jego cech nie zmienia dopasowania na treningu.

## 16. Analizy wrażliwości: sprawdzamy, od czego zależy wynik

Nie chodzi o znalezienie wariantu z najwyższą liczbą. Każda zmiana ma odpowiadać na konkretne pytanie. Najpierw definiujemy porównanie i oczekiwany sposób interpretacji, potem je uruchamiamy.

| Analiza | Co zmieniamy | Co pozwala sprawdzić |
|---|---|---|
| Bez NRBC i %NRBC | Usuwamy obie cechy **przed MICE**, dopasowujemy nowy imputer i klasyfikator | Czy wynik zależy od bezpośredniej lub pośredniej informacji tych cech? |
| Często mierzone panele | Ustalona lista cech z małym odsetkiem braków w obu kohortach; osobno wariant wyłącznie ze zmierzonym kompletnym panelem | Czy przewaga utrzymuje się przy mniejszym udziale odtwarzanych wartości? |
| Wspólne okno dat | Wybieramy rekordy z okna przed dopasowaniem preprocessingu, zachowujemy przydział pozostałych pacjentów | Czy wynik utrzymuje się przy ograniczeniu różnicy okresów? |
| Prosta imputacja | Zamiast MICE stosujemy medianę z train, zachowując ten sam poziom przygotowania i agregację | Czy złożoność MICE pomaga w końcowym rankingu? |
| Model obsługujący NaN | Ustalony model drzewiasty dostaje braki bez MICE | Czy wynik zależy od sposobu imputacji? Model nadal może wykorzystywać braki jako skrót |
| Kolejność agregacji | Mediana surowych cech pacjenta przed imputacją zamiast mediany po imputacji rekordów | Czy profil i wynik silnie zależą od kolejności przygotowania? |
| Trening rekordowy | Rekordy z wagami 1/n pacjenta i ustalona agregacja ich predykcji | Czy jednostka treningu zmienia wniosek mimo pacjentowej oceny? |
| Udział ukrywania | 20% i 60%, obok głównych 40% | Czy metoda zachowuje przewagę przy innej liczbie widocznych P i ukrytych przypadków? |
| Bez pacjentów obu kohort | Wykluczenie 63 osób z odpowiednich train i test bez ponownego losowania reszty | Czy sposób rozstrzygnięcia tej grupy ma duży wpływ? |

Usunięcie NRBC tylko po MICE-38 usuwa tę cechę z klasyfikatora, ale jej wartości mogły już posłużyć do uzupełnienia innych cech. Taki dodatkowy wariant może być użyteczny, jednak odpowiada na węższe pytanie i musi być osobno nazwany.

We wspólnym oknie mamy obecnie 680 rekordów NEURO od 271 pacjentów. Stratyfikacyjne kryterium co najmniej połowy historii w oknie spełnia 159 osób — to inna definicja. W analizie ograniczonej rekordowo do okna korzystamy z faktycznie pozostałych rekordów, a nie z pełnych historii tych pacjentów. Mamy około 22 ukrytych na fold przy 40% ukrywania, więc niewielkie różnice procentowe mogą być szumem.

Kontrola z 17.10, przeliczona 31.10, bez imputacji i wyłącznie na rekordach z okna, dała AUC pacjentowe P-vs-U 0,825. Ograniczenie czasu nie usuwa więc całej rozróżnialności kohort. Pełny wariant MICE w oknie nadal wymaga dopasowania imputera do treningu z okna, jeśli ma odpowiadać na pytanie o taki pipeline.

**Ważne przy porównywaniu podzbiorów:** kompletny panel może pozostawić innych pacjentów w każdej kohorcie. Pokazujemy liczbę wykluczonych, udziały P/U i zakres czasu. Jeśli to możliwe, porównujemy pełny i ograniczony model również na tych samych osobach. Wyższy wynik po usunięciu trudnych pacjentów nie jest automatycznie poprawą metody.

Na start rekomendowana jest z góry zapisana lista „częstych cech”, uzasadniona istniejącą diagnostyką braków i zawierająca sensowny panel laboratoryjny. Nie wybieramy jej według wyniku testowego. Jeśli zamiast listy ma działać automatyczny próg, zespół ustala go przed startem i sprawdza, jakie cechy rzeczywiście zostawia. Próg 10% osobno w obu kohortach byłby w obecnych danych zbyt restrykcyjny dla podstawowej morfologii NEURO, gdzie brakuje około 12,3% rekordów.

Selektor oparty na brakach w całym train może działać bez etykiet. Warunek liczony osobno w grupach widocznych P i U zależy już od `observed_label`, a więc również od maski: należy dopasować go tylko na train i uwzględnić maskę w cache. Nie podajemy mu po cichu prawdziwego pochodzenia ukrytych chorych jako dodatkowej wiedzy. Wariant z kompletnym panelem dotyczy wyselekcjonowanych osób i nie gwarantuje działania u wszystkich pozostałych.

Przy zmianie udziału ukrywania model trzeba ponownie wytrenować, bo zmieniają się etykiety treningowe. Można zachować ustawienia dobrane w głównym protokole, aby badać stabilność bez kolejnego strojenia. Inne q można policzyć na tych samych predykcjach, jeśli zostały wcześniej przewidziane jako miary dodatkowe. Zmiana cech lub rodzaju imputacji tworzy nowy pipeline; albo stroimy go wewnętrznie według jednakowego budżetu, albo jawnie pokazujemy test z przeniesionymi ustawieniami.

**Gotowe, gdy:** dla każdej analizy potrafimy powiedzieć, co zmieniono, jaka populacja pozostała i czy kierunek przewagi nad kontrolami się utrzymał. Spadek jakości jest wynikiem badawczym, nie powodem usunięcia analizy z raportu.

## 17. Co zapisujemy i jak powstaje raport

### 17.1 Artefakty każdego eksperymentu

| Plik / zawartość planowana | Po co jest potrzebna |
|---|---|
| Pełna konfiguracja i wersja protokołu | Wiadomo, jakie pytanie i wariant faktycznie sprawdzono |
| Odciski wejścia, przydziału i masek; wersja kodu i bibliotek | Pozwala ustalić, czy dwa przebiegi dotyczą tych samych danych i reguł |
| Maska na poziomie pacjenta | Dowód, że modele miały tych samych ukrytych chorych |
| Predykcje OOF pacjentów | Podstawa obliczeń i analiz; jeden wpis na pacjenta dla danego modelu, scenariusza i zestawu ziaren |
| Wyniki per fold i ziarno z licznikami | Ujawniają liczebności i niestabilność ukrytą przez samą średnią |
| Parametry modeli, zapis strojenia, czasy | Wyjaśniają wybór ustawień i koszt uzyskania wyniku |
| Manifest przygotowania i logi | Pozwalają odtworzyć błąd lub odrzucić niezgodny cache |
| Krótka karta modelu | Zwięzły opis celu, danych, oceny, ograniczeń i niedozwolonych interpretacji |

Planowane kolumny tabeli OOF: identyfikator eksperymentu i scenariusza, model, ziarno maski i modelu, `patient_id`, outer-fold, `original_label`, `observed_label`, `is_hidden`, score, percentyl w puli U, liczba rekordów i udział braków przed imputacją. Metadane pomagają diagnozować wynik, ale nie stają się automatycznie cechami modelu. Tabela ma powstawać z predykcji testowych, bez zastępowania ich predykcjami treningowymi.

### 17.2 Niepewność i uczciwe porównanie

Przy pięciu foldach i pięciu maskach mamy 25 wartości, ale te same osoby wracają między maskami, a treningi zachodzą na siebie. Nie traktujemy tego jako 25 niezależnych prób. Minimum raportowania to wynik łączny dla każdego ziarna, wyniki foldów, liczebności i rozrzut. Zakres minimum–maksimum nie jest przedziałem ufności.

Jeśli liczymy bootstrap, losujemy pacjentów z zastępowaniem i przenosimy razem ich wyniki ze wszystkich modeli i masek. W każdym powtórzeniu odtwarzamy pule i top-q według tej samej reguły. Porównanie jest sparowane: model A i B mają tych samych wylosowanych pacjentów. Nie losujemy pojedynczych tygodni ani każdego wystąpienia pacjenta między ziarnami osobno. Przedział z gotowych OOF opisuje niepewność oceny warunkową dla tych wytrenowanych modeli; nie zawiera całej niepewności ich ponownego trenowania i historycznego doboru danych.

### 17.3 Jak wyjaśniać model

Analizujemy predykcje OOF: które cechy wiążą się z wysokim wynikiem, czy są często imputowane, czy pozycja zależy od liczby badań i czy model wskazuje podobne osoby między maskami. Współczynniki regresji, ważności lub SHAP opisują działanie modelu, a nie przyczynę choroby. Współzależne cechy, np. wyniki morfologii, mogą dzielić między siebie ważność.

Przykład poprawnego opisu: „Przy q = 5% model odzyskał X% celowo ukrytych NEURO, a kontrola braków Y%; wynik zmienił się do Z% we wspólnym okresie”. Nie zastępujemy tego zdaniem „model wykrywa X% tętniaków” ani „pozostali wskazani są fałszywie dodatni”.

Po zakończeniu porównań można dopasować wybrany model do wszystkich dostępnych danych dla dalszych eksperymentów. Nie daje to nowej niezależnej oceny. Wynik opisujemy na podstawie zaplanowanej walidacji OOF, a rzeczywiste zastosowanie kliniczne wymaga odrębnej weryfikacji.

## 18. Jak podzielić pracę w zespole

Poniżej role do obsadzenia, bez przypisywania osób bez wspólnego ustalenia. Jeden członek zespołu może realizować więcej niż jedną rolę. Warto, żeby odbiór części wykonał ktoś inny niż jej autor.

| Rola | Zakres | Co przekazuje innym |
|---|---|---|
| Koordynacja protokołu i danych | Decyzje z prowadzącym, wersja danych, konfiguracja, wspólne środowisko, zgodność dokumentacji | Jedno obowiązujące wejście i zapis zasad |
| Pipeline i preprocessing | Odczyt foldów, manifesty, przygotowanie profili, inner CV i cache | Wspólny interfejs train/predict bez własnych metryk |
| Maskowanie i ocena | Maski pacjentów, pule U, top-q, metryki, OOF, kontrole obliczeń | Jedna funkcja oceny dla wszystkich modeli i ręczny przykład referencyjny |
| Modele bazowe i PU | Implementacje oraz konfiguracje modeli; współpraca z istniejącym pipeline'em | Score pacjentów, parametry i czasy w ustalonym formacie |
| Wrażliwość i raport | Scenariusze, liczebności, porównania sparowane, wykresy i interpretacja | Czytelne zestawienie przewag i ograniczeń |

Najpierw wspólnie uzgadniamy kolumny i format wyników z sekcji 17. Modelami bazowymi i PU można zajmować się równolegle, testując interfejs na małych danych syntetycznych. Pełne porównania czekają na gotowe maskowanie i ocenę. Nie tworzymy osobnego podziału, imputera i wzoru metryki w każdym skrypcie modelowym.

Historia prototypu z commita `2cb6d1a` i kontrola `kontrola_skrotow.py` mogą przyspieszyć pracę. Przeniesiony kod trzeba dopasować do obecnego przydziału i kontraktu; wcześniejsze istnienie funkcji nie oznacza ukończenia nowego pakietu.

## 19. Kryteria ukończenia i pierwszy konkretny rezultat

### 19.1 Przed pierwszym pełnym przebiegiem

- Uzgodniono wersję danych oraz cel albo jawnie ograniczony zakres eksploracyjny.
- Zapisano jednostkę, kolejność przygotowania, listę cech, q, udział ukrywania i ziarna.
- Maski, ręczny przykład metryki, remisy i przypadek bez ukrytych mają sprawdzony wynik.
- Pacjenci są rozłączni, wejście i przydział zgodne, kolumny kontrolne nie trafiają do modelu.
- Kod potrafi odrzucić niezgodny manifest zamiast po cichu wczytać inne pliki.

### 19.2 Pierwszy kamień milowy: eksperyment bez strojenia

**Do pokazania zespołowi:** jedna tabela wyników losowego rankingu, demografii, wzorca braków, regresji logistycznej i PU Bagging, policzona na pięciu zamrożonych foldach i wspólnych maskach. Towarzyszą jej predykcje OOF, konfiguracja, liczebności i krótki opis interpretacji.

Odbiór nie wymaga osiągnięcia wysokiej jakości. Wymaga, żeby każdy pacjent miał dokładnie jedną predykcję testową w danym wariancie i ziarnie, a druga osoba potrafiła odtworzyć wynik. Dopiero wtedy rozszerzamy modelowanie i strojenie.

### 19.3 Przed raportowaniem wyników modeli strojonych

- Preprocessing w każdym inner-foldzie jest dopasowany tylko do inner-train.
- Dobór parametrów i ewentualnej rodziny modelu odbywa się bez outer-test.
- Cache jest zgodny z wariantem danych, cech i czasu.
- Wszystkie zaplanowane modele, kontrole i analizy są raportowane, również te ze spadkiem jakości.
- Porównania uwzględniają powtarzanie tych samych pacjentów i podają właściwy zakres niepewności.
- Rozdzielono odzyskiwanie znanych NEURO od rzeczywistego rozpoznawania choroby u KOR.

Nie ma potrzeby wracać do ogólnego audytu po każdym pakiecie. Kontrolujemy konkretne warunki opisane w planie. Jeśli zmienia się pytanie badawcze, wersja danych lub reguły oceny, zapisujemy nową wersję protokołu i ustalamy, które wyniki trzeba przeliczyć.
