# Plan modelowania Positive-Unlabeled (propozycja)

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Status:** **propozycja do ustalenia z prowadzącym.** Streszcza plany z września (dawne `02_PLAN_PRZED_MODELOWANIEM.md` i `03_PLAN_MODELOWANIE.md`) i wnioski z prototypu kodu. Po spotkaniu powstanie z tego plan obowiązujący, a dopiero potem implementacja.
**Stan na:** 04.10.2026, po decyzjach dotyczących danych

To jedyny dokument w `4-pu-setup/`. Wejście do modelowania jest gotowe i zamrożone: dane po decyzjach z 04.10.2026, podział i imputacja w foldach (etapy 1–3, `1026_REALIZACJA_DO_SGKF.md`). Ten dokument opisuje, co dalej, i zbiera założenia oraz ograniczenia, które modelowanie musi uwzględnić (sekcja 3.0). Rozstrzygnięte problemy z danymi opisuje `1026_REALIZACJA_DO_SGKF.md`, sekcja 6.

---

## 1. Na czym polega zadanie i dlaczego ocena jest nietypowa

Mamy klasę **P** (NEURO, potwierdzony tętniak) i klasę **U** (KOR, populacja ogólna, nikt nie robił im obrazowania). `label = 0` znaczy „nieoznaczony”, a nie „zdrowy”. Konsekwencje:

1. **Nie zmierzymy liczby pomyłek.** Wysoki wynik pacjenta KOR może być błędem albo niewykrytym chorym, więc „false positive” traci sens.
2. **Accuracy jest bezużyteczna.** Przy 4,45% pozytywnych model mówiący zawsze „0” ma 95,5% trafności.
3. **Oceniamy kontrolowanym testem odzyskiwania** (pomysł prowadzącego). Część znanych chorych ukrywamy w puli U i sprawdzamy, czy model wypchnie ich na górę rankingu.

Cały protokół podporządkowany jest pytaniu: *czy wynik mówi coś o tętniakach, czy o tym, jak powstały dane?*

---

## 2. Wejście: co jest gotowe

| Element | Gdzie | Stan |
|---|---|---|
| dane po decyzjach z 04.10.2026 (38 cech, przed imputacją) | `data/processed/aneurysm_sgkf_input.csv` (część A etapu 3) | **gotowe** |
| przydział pacjent → fold (5 foldów, stratyfikacja etykieta × okno × liczba rekordów) | `3-sgkf-split/sgkf/results/pacjent_fold.csv` | **zamrożone** |
| MICE wewnątrz foldu (ExtraTrees 7/78/10) | `3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py` | gotowe |
| zaimputowane foldy zewnętrzne: 38 cech (główne) i 35 cech (wrażliwość) | `3-sgkf-split/sgkf/results/foldy_imputowane_{38,35}/` (lokalnie, poza gitem) | **policzone**, odtwarzalne jednym poleceniem |

**Zasada nadrzędna:** foldy wyłącznie z `podzial.wczytaj_podzial()`, nigdy liczone od nowa. Każde dopasowanie (scaler, imputer, selekcja cech, model, próg) odbywa się tylko na części treningowej.

---

## 3. Przygotowanie: protokół przed pierwszym modelem

### 3.0 Założenia wejściowe: co już rozstrzygnięto, a co zostaje

**Rozstrzygnięte 04.10.2026** (wdrożone w etapie 3, szczegóły w `1026_REALIZACJA_DO_SGKF.md`, sekcja 6):

| Decyzja | Co to oznacza dla modelowania |
|---|---|
| 63 pacjentów w obu kohortach: pozytywni, rekordy KOR usunięte | nic do zrobienia; flaga `pacjent_mieszany` pozwala ich wykluczyć w analizie wrażliwości |
| wartości niemożliwe (KREA, K, Na) → brak, uzupełniane w foldzie | nic do zrobienia |
| zestaw cech: 38 główne, 35 jako wrażliwość | oba warianty mają gotowe foldy na tym samym podziale |
| rozjazd czasowy kohort: **bez korekty**, opisany jako ograniczenie | **nie dodawać roku ani daty jako cechy**, bo ułatwia rozpoznanie kohorty; analiza we wspólnym oknie dat (271 pozytywnych) tylko jako ewentualna, dodatkowa wrażliwość |
| asymetria braków (NEURO 26%, KOR 12% komórek uzupełnianych): **bez korekty** | zalecana analiza wrażliwości: model na cechach o niskim odsetku braków w obu kohortach |

**Zostaje do potwierdzenia lub decyzji przy planie:**

| Kwestia | Stan | Wpływ |
|---|---|---|
| pochodzenie wyników NEURO | założenie robocze: hospitalizacja (mediana rozpiętości badań 35 dni u 1 215 pacjentów z >1 rekordem) | interpretacja: model rozpoznaje pacjenta **hospitalizowanego** z tętniakiem, nie ryzyko przesiewowe |
| reguła agregacji rekordów do pacjenta | rekomendacja: **mediana**, średnia jako wrażliwość | parametr pipeline'u |

**Uzasadnienie reguły agregacji** (`4-pu-setup/etap0_agregacja.py`, wyniki w `4-pu-setup/results/`, dane z 09.2026 sprzed przygotowania). 41% pacjentów ma więcej niż jeden rekord, a NEURO ma ich średnio dwa razy więcej niż KOR (3,65 wobec 1,83), bo pacjenci leżą w szpitalu. Reguła wrażliwa na liczbę pomiarów tworzy sygnał z samej częstości badania. Korelacja liczby rekordów z wartością cechy, liczona wewnątrz KOR:

| Reguła | Mediana \|ρ\| | Maks. \|ρ\| | Cech z \|ρ\| > 0,2 |
|---|---:|---:|---:|
| **mediana** | 0,079 | 0,257 | **2** |
| ostatni pomiar | 0,083 | 0,296 | 1 |
| średnia | 0,125 | 0,284 | 6 |
| maksimum | 0,190 | 0,379 | **17** |

Pozorna siła sygnału jest przy tym niemal identyczna dla średniej, mediany i maksimum (mediana \|AUC − 0,5\|: 0,0440, 0,0438, 0,0467). Maksimum odpada, bo przemyca liczbę badań do 17 z 35 cech i wciąga skrajne wartości. „Ostatni pomiar” daje wyraźnie silniejszy pozorny sygnał (0,0606), ale też odpada: przy założeniu hospitalizacji oznacza u chorych pomiar po leczeniu, a u KOR zwykły wynik kontrolny. Silniejszy sygnał prawdopodobnie bierze się właśnie z tej różnicy, a nie z choroby.

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
- trzy kolumny: `true_label` (prawda), `observed_label` (to, co widzi model), `is_hidden` (flaga kontrolna),
- scenariusz główny, np. 40% ukrytych, oraz 20% i 60% jako wrażliwość, każdy na 5–10 ziarnach,
- **identyczne maski dla wszystkich modeli**,
- maski są deterministyczne (`patient_id` + udział + ziarno), więc wystarczy zapisać parametry, a nie tabelę liczącą 21 MB.

### 3.3 Terminologia

| Sytuacja | Jak mówimy |
|---|---|
| ukryty chory z wysokim wynikiem | sukces (TP względem `true_label`), nie „false positive” |
| pacjent KOR z wysokim wynikiem | „wysoko sklasyfikowany pacjent U”, „kandydat do diagnostyki”, nigdy „false positive” |
| wynik modelu | **risk score**, nie „prawdopodobieństwo tętniaka”, dopóki nie jest skalibrowany |

### 3.4 Metryki: na pacjentach, nie na rekordach

Pacjent NEURO ma średnio 3,65 rekordu, a KOR 1,83, więc metryki wierszowe premiowałyby chorych.

| Metryka | Rola |
|---|---|
| **RecallHidden@q**: odsetek ukrytych chorych w górnych `q` puli U | **główna** |
| udział ukrytych w top-`q` | uzupełnienie; **nie jest** estymatorem precision |
| Lift@q | czy model jest lepszy od losowania |
| RecallKnown | czy nie gubimy oczywistych przypadków |
| ROC-AUC / PR-AUC P-vs-U i kontrolowane (ukryci vs reszta U) | pomocnicze, opisywane jako P-vs-U, a nie wykrywanie choroby |

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
| **Elkan–Noto** / estymacja class prior | korekta PU; prior szacowany tylko na treningu |

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

Rozdzielczość `RecallHidden@q` wynosi 1/146 ≈ 0,7 pp. Gdyby ograniczyć dane do wspólnego okna dat, zostałoby ~22 ukrytych na fold i rozdzielczość spadłaby do 4,5 pp. To kolejny powód, dla którego rozjazdu czasowego nie korygujemy przez obcięcie danych.

---

## 6. Prototyp z września: czego się nauczyliśmy

We wrześniu powstała robocza implementacja punktów 3.1–3.5: moduły konfiguracji, danych, foldów, ukrywania, metryk i funkcji celu, około 750 linii plus 17 testów. Leżała na branchu `pu-pipeline-1-5-lk` (commit `2cb6d1a`), który usunięto przy porządkach 04.10.2026. Nowa implementacja powstanie według ustalonego planu. Wnioski warte zachowania:

1. **Przebieg kontrolny z losowym scoringiem** (3000 pacjentów, prawdziwe MICE w foldach) dał `RecallHidden@q` 0,079 przy oczekiwanych 0,05. To mieści się w szumie przy 8–14 ukrytych na fold. Metryki były podpięte poprawnie, ale **analizy na podpróbkach są bezużyteczne**, bo przy małej liczbie ukrytych szum dominuje.
2. **Remisy w rankingu** trzeba rozstrzygać deterministycznie (np. po `patient_id`), inaczej wynik zależy od kolejności wierszy.
3. **Fold bez ukrytych ma zwracać brak wartości, nie zero**, bo uśrednianie zer zaniża wynik.
4. **Wszystkie decyzje etapu 0 jako parametry konfiguracji**, a nie logika w kodzie. Zmiana decyzji kosztuje wtedy czas obliczeń, nie przepisywanie.
5. **Gniazdo na kontrolę kohort** w pipeline foldu: funkcja `(train, test) → (train, test, wagi)` wpinana między podział a scaler. Przy decyzji „bez korekty” zostaje puste, ale pozwala później dodać np. ważenie po czasie bez przebudowy.
6. Prototyp liczył **własny podział** stratyfikowany tylko po etykiecie. Nowa implementacja ma czytać zamrożony przydział z `3-sgkf-split/sgkf/results/pacjent_fold.csv` przez `podzial.wczytaj_podzial()`.

---

## 7. Do ustalenia z prowadzącym

1. Pochodzenie wyników NEURO: czy pochodzą sprzed rozpoznania tętniaka, czy z hospitalizacji, w trakcie której go rozpoznano i leczono? Od tego zależy, jak wolno opisać model.
2. Wartość `q`: z jakiej przepustowości diagnostyki obrazowej ją wyprowadzić?
3. Czy 40% jako główny udział ukrywania jest uzasadnione, skoro zmienia proporcję klas widzianą przez model?
4. Stratyfikacja foldów po prawdziwym statusie pacjenta (element konstrukcji benchmarku, model go nie widzi): akceptowalna?
5. Kolejność: imputacja rekordów, potem agregacja do pacjenta (zgodne z benchmarkiem imputacji), czy odwrotnie (taniej, mniej braków)?
6. Jednostka treningu: jeden profil na pacjenta czy rekordy tygodniowe z wagami 1/n?
7. Które metody PU: PU Bagging i Elkan–Noto wystarczą?

---

## 8. Ograniczenia, które trafią do raportu niezależnie od wyników

1. Nie mierzymy rzeczywistego odsetka fałszywych alarmów w KOR. Ukrywanie testuje tylko odzyskiwanie znanych przypadków.
2. Założenie SCAR jest wątpliwe, bo NEURO to chorzy objawowi.
3. ROC-AUC i PR-AUC z KOR jako klasą negatywną to metryki P-vs-U.
4. Kohorty są rozdzielone czasowo i źródłowo (KOR 2020–2021, NEURO 2000–2024); model może rozpoznawać epokę. Decyzja: bez korekty, opisane wprost.
5. Brak daty diagnozy ogranicza interpretację predykcyjną.
6. Przy założeniu hospitalizacji model rozpoznaje profil pacjenta hospitalizowanego, a nie ryzyko przed rozpoznaniem; nie wolno go opisywać jako modelu przesiewowego.
7. Asymetria braków: MICE odtwarza 26% profilu NEURO wobec 12% KOR. Do tego czyszczenie kwantylowe z etapu 1 usunęło 3,2% rekordów NEURO i 0,8% KOR.
