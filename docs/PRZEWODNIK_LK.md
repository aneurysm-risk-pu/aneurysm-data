# Przewodnik po projekcie: powrót po przerwie

*Notatki dla Łukasza, 04.10.2026. Nie jest to dokument dla prowadzącego, tylko ściąga do zrozumienia, co tu się dzieje i dlaczego.*

Jeśli masz 15 minut: przeczytaj sekcje 1, 2 i 7. Jeśli masz godzinę: całość, a potem pytania kontrolne na końcu.

---

## 1. Jedno pytanie, które steruje całym projektem

> **Czy wynik modelu będzie mówił coś o tętniakach, czy tylko o tym, jak powstały nasze dane?**

Prawie każda decyzja w repo jest odpowiedzią na to pytanie. Mamy dwie bazy, które różnią się nie tylko chorobą:

| | KOR (`label = 0`) | NEURO (`label = 1`) |
|---|---|---|
| kto | populacja ogólna | pacjenci oddziału neurologicznego z tętniakiem |
| kiedy | prawie wyłącznie 2020–2021 | 2000–2024, głównie przed 2020 |
| jak często badani | 1,83 rekordu na osobę | 3,65 rekordu na osobę |
| ile braków | 12% komórek | 26% komórek |
| w jakiej sytuacji | rutynowe badania | najpewniej hospitalizacja (mediana rozpiętości badań: 35 dni) |

Model, który ma odróżnić `label = 1` od `label = 0`, może więc nauczyć się rozpoznawać **rok, częstość badań, wzorzec braków albo pobyt w szpitalu** zamiast tętniaka. To zjawisko nazywa się *confounding* (zmienne zakłócające) albo *shortcut learning*. Większość pracy z września to szukanie tych skrótów i ich mierzenie.

---

## 2. Dlaczego „Positive-Unlabeled”, a nie zwykła klasyfikacja

W zwykłej klasyfikacji masz chorych (P) i potwierdzonych zdrowych (N). Tu nikt nie zrobił obrazowania pacjentom KOR, więc wśród nich mogą być osoby z niewykrytym tętniakiem. `label = 0` znaczy **nieoznaczony (U)**, a nie zdrowy.

Konsekwencje, które warto mieć w głowie:

- **„False positive” traci sens.** Pacjent KOR z wysokim wynikiem może być błędem modelu albo niewykrytym chorym, i nie da się tego rozróżnić. Dlatego w dokumentach mówimy „wysoko sklasyfikowany pacjent U”.
- **Accuracy jest bezużyteczna.** 95,5% pacjentów to U, więc model mówiący zawsze „0” ma 95,5% trafności.
- **Ocena przez ukrywanie.** Bierzemy część znanych chorych, udajemy, że są nieoznaczeni (`observed_label = 0`, `true_label = 1`), i sprawdzamy, czy model wypchnie ich na górę rankingu. To pomysł prowadzącego i rdzeń ewaluacji (wstępna implementacja: `4-pu-setup/pu/ukrywanie.py`, `metryki.py` na branchu `pu-pipeline-1-5-lk`).
- **SCAR** (*Selected Completely At Random*) to założenie wielu metod PU, że znani chorzy są losową próbką wszystkich chorych. U nas to nieprawda: NEURO to chorzy objawowi, którzy trafili do szpitala. Tego nie da się naprawić, da się tylko uczciwie opisać.

---

## 3. Mapa repozytorium i kolejność czytania

```
data/raw/              dwa surowe CSV (nie ruszamy)
data/interim/          stany pośrednie czyszczenia
data/imputation-inputs/aneurysm_concatted.csv     ← WEJŚCIE (38 cech, przed imputacją)
data/processed/aneurysm_concatted_cleaned.csv     ← wariant 35 cech
1-data-preparation/    czyszczenie i EDA (Twoje skrypty + notebooki Liwii)
2-imputation/          benchmark imputacji; final/ to wspólny kod, reszta to fazy indywidualne
3-sgkf-split/          PODZIAŁ: podzial.py (v2) + MICE w foldzie
4-pu-setup/            etap 0 (diagnostyka danych) + README: co dalej
01, 04, 05_*.md        podsumowania
```

Kolejność czytania przy powrocie:
1. `05_REALIZACJA_DO_SGKF.md`: co zrobiliśmy i kto co robił.
2. `3-sgkf-split/RAPORT_SGKF_MICE.md`, sekcja 6: najnowsza zmiana.
3. `4-pu-setup/ETAP0_USTALENIA.md`: problemy z danymi.
4. `4-pu-setup/README.md`: co dalej. Wstępne plany modelowania (02/03) i implementacja punktów 1–5 leżą na branchu `pu-pipeline-1-5-lk`, tylko jako odniesienie; nowy plan powstanie po spotkaniu.

---

## 4. Wyciek danych: cztery rodzaje, wszystkie u nas wystąpiły albo groziły

Wyciek (*leakage*) to sytuacja, w której informacja z danych testowych dostaje się do treningu. Model wygląda wtedy lepiej, niż jest. W projekcie zetknęliśmy się z czterema odmianami:

| Rodzaj | Jak wyglądał u nas | Jak to rozwiązano |
|---|---|---|
| **pacjent w obu zbiorach** | pacjent ma do 38 rekordów; losowy podział rozrzuciłby je między train i test, a model „rozpoznawałby osobę” | podział po pacjentach (Group) |
| **preprocessing przed podziałem** | imputacja całego zbioru, potem podział: MICE uczyłby się też na rekordach testowych | scaler i MICE dopasowywane **wewnątrz foldu**, tylko na train |
| **etykieta w selekcji cech** | CRP, MONO i %MONO usunięte na podstawie korelacji z `label` na całym zbiorze | powrót do 38 cech; selekcja, jeśli w ogóle, tylko wewnątrz foldu |
| **strojenie na danych oceny** | przebieg 1 benchmarku imputacji: Optuna i ocena na tej samej masce | osobne maski: seed 43 do strojenia, seed 42 do oceny. W modelowaniu analogicznie: walidacja zagnieżdżona |

Zasada ogólna: **wszystko, co się uczy (a scaler i imputer też się uczą!), musi widzieć tylko train.** Tego samego dotyczy walidacja zagnieżdżona (wstępnie w `pu/foldy.py` na branchu `pu-pipeline-1-5-lk`): foldy wewnętrzne do strojenia są budowane wyłącznie z outer-train, a preprocessing jest dopasowywany od nowa w każdym z nich.

---

## 5. Podział danych: co się zmieniło i dlaczego (v1 → v2)

### 5.1 Jak działa `StratifiedGroupKFold`

- **Group:** dostaje `groups=patient_id` i nigdy nie rozdziela grupy między train i test.
- **Stratified:** próbuje, żeby proporcja `y` w każdym foldzie była zbliżona do globalnej.
- Działa na **rekordach**, więc wyrównuje proporcję rekordów pozytywnych (8,5%), a nie pacjentów (4,45%).
- Przyjmuje **jedną** zmienną stratyfikującą.

### 5.2 Trzy subtelności, które wyszły przy audycie

**(a) `random_state` bez `shuffle=True` nic nie robi.** W sklearn większość splitterów ignoruje `random_state`, gdy `shuffle=False`. W v1 stała `RANDOM_STATE = 42` działała tylko w MICE. Podział wynikał z kolejności wierszy w CSV, a plik jest posortowany: najpierw cały KOR, potem NEURO. Przesortowanie pliku po cichu zmieniłoby foldy.

**(b) Algorytm SGKF z tasowaniem zmienił się między wersjami sklearn.** Ta sama konfiguracja (shuffle, seed 42) daje u Ciebie na Windowsie (sklearn 1.8) rozstęp udziału pozytywnych 0,11 pp, a na Macu (sklearn 1.6, bo jest tu tylko Python 3.9) 0,80 pp. To **inne podziały**. Wniosek ogólny: **przepis na podział nie jest powtarzalny między środowiskami, a zapisany przydział jest.** Stąd `results/pacjent_fold.csv` jako źródło prawdy. `StratifiedKFold` na pacjentach okazał się stabilny między wersjami: zapisany plik z Windowsa odtworzył się na Macu w 100%.

**(c) Wyrównanie klas nie oznacza wyrównania foldów.** Wrześniowa rekomendacja (pacjenci, stratyfikacja po etykiecie) daje idealny balans klas (0,01 pp), ale przy niektórych ziarnach jeden fold dostaje wyraźnie więcej NEURO z epoki KOR albo więcej NEURO z wieloma rekordami: do 5 i 8 pp różnicy. Dlaczego to ważne: te dwie zmienne to znane skróty (sekcja 1). Jeśli fold 2 ma więcej NEURO z lat 2020–2021, to model będzie tam miał **trudniej** (mniej epokowego skrótu), więc wyniki między foldami będą się różnić z powodów niezwiązanych z modelem. Większa wariancja oznacza mniej wiarygodne porównanie modeli.

Siatka stabilności z września tego nie wychwyciła, bo mierzyła udział lat 2020–2021 we **wszystkich** pacjentach foldu, a 96% z nich to KOR, który i tak jest cały w tym oknie. Problem siedział w **podgrupie** NEURO. Lekcja: miary balansu licz tam, gdzie siedzi zmienność.

### 5.3 Stratyfikacja wielokluczowa: trik

`StratifiedKFold` przyjmuje jedną kolumnę `y`, ale to może być **dowolna etykieta kategorii**. Sklejasz kilka zmiennych w jeden klucz:

```
"1_pozaokno_n5+"  ← pozytywny, poza oknem czasowym KOR, 5+ rekordów
"0_okno_n1"       ← nieoznaczony, w oknie, 1 rekord
```

i stratyfikujesz po kluczu. Każda kombinacja jest rozkładana równo między foldy. Ograniczenie: każda warstwa musi mieć co najmniej `n_splits` członków, a w praktyce kilka razy więcej. Stąd `min_warstwa = 20` i scalanie małych warstw do `<etykieta>_inne`. U nas najmniejsza warstwa ma 34 osoby, więc scalanie nie zachodzi. Mieszanych (63) nie wkładamy do klucza, bo powstałyby warstwy po kilka osób.

Dlaczego podział na tabeli pacjentów wciąż jest „Group”: skoro jeden wiersz to jeden pacjent, nie da się go rozdzielić. Warunek grupowania jest spełniony z konstrukcji. Potem `indeksy_foldow()` tłumaczy przydział pacjentów na indeksy rekordów.

### 5.4 Odcisk danych

`podzial_meta.json` zawiera SHA-256 pliku danych. `wczytaj_podzial()` przelicza go i odmawia działania, jeśli dane się zmieniły, bo przydział do starych danych nie musi pasować do nowych. Szczegół, który zjadłby Ci godzinę: git na Windowsie domyślnie zamienia `\n` na `\r\n` przy checkoucie, więc surowe bajty CSV różnią się między komputerami. Dlatego hash liczony jest po ujednoliceniu końców linii.

---

## 6. Imputacja: jak działa to, co wybraliśmy

### 6.1 `IterativeImputer` (u nas „MICE”)

1. Start: braki wypełnione prostą wartością (średnia kolumny).
2. Dla każdej kolumny z brakami po kolei: wytrenuj regresor (u nas ExtraTrees), który przewiduje tę kolumnę z pozostałych, i podmień braki na predykcje.
3. Powtórz całą rundę `max_iter` razy (u nas 7). Każda runda korzysta z lepszych uzupełnień poprzedniej.

**Dlaczego to nie jest „prawdziwe” MICE:** klasyczne *Multiple Imputation by Chained Equations* losuje z rozkładu predykcji (`sample_posterior=True`) i tworzy **kilka** kompletnych zbiorów, żeby oddać niepewność uzupełnienia. My robimy jedną, deterministyczną imputację. To jest w porządku, ale trzeba to nazywać uczciwie, i dokumenty to robią.

**Dlaczego ExtraTrees:** drzewa łapią nieliniowe zależności (np. HGB–HCT–RBC) i nie zakładają rozkładu. BayesianRidge przegrał we wszystkich przebiegach Optuny.

### 6.2 Skalowanie na kompletnych wierszach i `min_value=0, max_value=1`

`MinMaxScaler` jest dopasowywany tylko na wierszach bez braków, tak samo jak w benchmarku. Dlatego parametry imputera z benchmarku pasują. Skutki:
- imputer z `min/max_value = 0/1` przycina **tylko wartości uzupełniane**, więc nowa wartość nie wyjdzie poza zakres kompletnych wierszy, a obserwowane zostają nietknięte,
- poza tym zakresem leży tylko 0,008% obserwowanych wartości, więc w praktyce to bez znaczenia,
- ryzyko koncepcyjne: kompletne wiersze to w 96% KOR, czyli skala jest „KOR-owa”.

### 6.3 Asymetria braków: nowe ustalenie z października

W NEURO MICE uzupełnia 26% komórek, w KOR 12%. Przy NRBC brakuje 47,7% w NEURO i 2,6% w KOR. Imputer uczy się na treningu, który w 91% składa się z KOR. Dwa możliwe skutki w przeciwnych kierunkach:
- uzupełnione wartości NEURO będą „podobne do KOR”, więc różnice się zatrą,
- albo sam fakt, że wartość była uzupełniona, zostawi ślad, który odróżnia kohorty.

Nie wiemy, który efekt przeważa. To dobry kandydat do analizy wrażliwości przy modelowaniu.

### 6.4 Benchmark imputacji: dlaczego tak

- **Maskowanie:** z kompletnych wierszy losowo „wyłączamy” 10% wartości, imputujemy i porównujemy z prawdą. Tylko tak da się zmierzyć błąd, bo przy prawdziwych brakach prawdy nie znamy.
- **RMSE w skali [0,1]:** porównywalne między cechami o różnych jednostkach.
- **KL divergence:** czy imputacja zachowuje kształt rozkładu, a nie tylko trafia średnio.
- **Oddzielne maski 43/42:** gdyby Optuna stroiła na tej samej masce, na której liczymy wynik, dopasowałaby parametry do tych konkretnych komórek. Klasyczny *overfitting do zbioru walidacyjnego*.

---

## 7. Dziennik decyzji: co, dlaczego, jaka alternatywa

| Decyzja | Dlaczego | Alternatywa i dlaczego nie |
|---|---|---|
| podział po pacjentach | jeden pacjent ma wiele rekordów | podział po rekordach: wyciek |
| stratyfikacja etykieta × okno × n_rek | wyrównuje znane skróty między foldami | tylko etykieta: foldy rozjeżdżają się do 8 pp |
| zapisany przydział | powtarzalność między komputerami i wersjami | przepis + seed: zależny od sklearn |
| MICE w foldzie | brak wycieku przez preprocessing | imputacja globalna: wyciek |
| MICE (ExtraTrees) zamiast MissForest | wygrywa na NEURO, szybszy, jeden imputer dla obu | MissForest lepszy tylko na KOR |
| 38 cech | selekcja po `label` to wyciek, a etykieta rozróżnia kohorty, nie chorobę | 35: tylko jako wrażliwość |
| mediana jako agregacja do pacjenta | najmniej koreluje z liczbą badań | maksimum przenosi liczbę badań do 17 cech; „ostatni” znaczy co innego w każdej kohorcie |
| mieszani jako pozytywni z flagą | byli w NEURO, czyli mają tętniaka | wykluczenie: gotowe jako parametr |
| `q` nie jest hiperparametrem | strojenie `q` to dobieranie definicji sukcesu pod wynik | — |

---

## 8. Pułapki: rzeczy, które łatwo zepsuć

1. **Nie używaj `2-imputation/final/results/aneurysm_imputed_*.csv` do modeli.** To imputacja globalna, przydatna tylko jako walidacja metody.
2. **Nie licz podziału od nowa w kolejnych etapach.** Czytaj `3-sgkf-split/results/pacjent_fold.csv` przez `wczytaj_podzial()`. Uwaga: wstępna implementacja PU (`pu/foldy.py` na `pu-pipeline-1-5-lk`) liczy własny podział; w nowym kodzie modelowania tego nie powtarzać.
3. **Metryki licz na pacjentach, nie na rekordach.** Inaczej pacjent z 38 rekordami waży 38 razy więcej.
4. **Nie dodawaj roku ani liczby rekordów jako cechy.** Ułatwiłoby to modelowi rozpoznanie kohorty, a nie choroby.
5. **Nie nazywaj wyniku „prawdopodobieństwem tętniaka”**, tylko *risk score*, dopóki nie jest skalibrowany.
6. **Pełny przebieg MICE to ~20 min na fold**, a z walidacją zagnieżdżoną wielokrotność tego. Do sprawdzania ścieżki: `--podprobka 3000 --szybkie-mice`.

---

## 9. Praca na dwóch komputerach

| | Windows | Mac (ten) |
|---|---|---|
| Python | nowszy; sklearn 1.8.0, pandas 3.0.3 (wg wcześniejszego planu) | 3.9.6 systemowy, więc sklearn maks. 1.6.1 |
| środowisko | twoje | `.venv/` w repo (ignorowany przez git) |

Skutki i zalecenia:
- wyniki zależne od implementacji bibliotek (SGKF z tasowaniem, w mniejszym stopniu ExtraTrees) mogą się różnić, dlatego zapisujemy artefakty, a nie przepisy,
- warto zainstalować na Macu nowszego Pythona (np. przez `uv` albo instalator z python.org) i przypiąć wersje w `requirements.txt` w korzeniu repo,
- `podzial_meta.json` zapisuje wersje bibliotek, więc zawsze widać, gdzie powstał artefakt.

Szybki start na Macu:

```bash
cd ~/repos/aneurysm-data
.venv/bin/python 3-sgkf-split/podzial.py --sprawdz
.venv/bin/python 3-sgkf-split/tests/test_podzial.py
```

---

## 10. Słowniczek

| Termin | Znaczenie |
|---|---|
| **P / U** | pozytywni (NEURO) / nieoznaczeni (KOR) |
| **fold** | jedna z k części danych; w CV każda raz jest testem |
| **outer / inner CV** | zewnętrzne foldy do oceny, wewnętrzne do strojenia; walidacja zagnieżdżona |
| **OOF (out-of-fold)** | predykcja dla pacjenta z modelu, który go nie widział |
| **stratyfikacja** | wymuszenie podobnego rozkładu zmiennej w każdym foldzie |
| **ARI** | Adjusted Rand Index: zgodność dwóch podziałów (1 = identyczne, 0 = jak losowe) |
| **confounder** | zmienna związana i z etykietą, i z cechami (u nas: epoka, liczba badań, braki) |
| **SCAR** | założenie, że znani pozytywni to losowa próbka wszystkich pozytywnych |
| **RecallHidden@q** | odsetek ukrytych chorych w top `q` rankingu puli U; główna metryka PU |
| **complete cases** | wiersze bez braków |
| **MCAR / MAR / MNAR** | braki całkowicie losowe / zależne od obserwowanych / zależne od samej brakującej wartości |

---

## 11. Pytania kontrolne

Odpowiedz sobie w głowie, a potem sprawdź pod spodem.

1. Dlaczego `RANDOM_STATE = 42` w v1 nie gwarantował powtarzalnego podziału?
2. Model ma ROC-AUC 0,95 na KOR vs NEURO. Co to mówi o wykrywaniu tętniaków?
3. Dlaczego nie można zrobić imputacji raz na całym outer-train, a potem dzielić go na foldy wewnętrzne?
4. Czemu stratyfikacja tylko po etykiecie jest niewystarczająca, skoro idealnie wyrównuje klasy?
5. Dlaczego „maksimum” jako reguła agregacji jest groźne właśnie w tym projekcie?
6. Co się stanie, gdy ktoś dopisze 10 rekordów do `aneurysm_concatted.csv` i uruchomi pipeline?

<details>
<summary>Odpowiedzi</summary>

1. Bo `shuffle=False`: `random_state` jest wtedy ignorowany, a podział zależy od kolejności wierszy.
2. Niewiele. To separacja P od U, która może wynikać z epoki, hospitalizacji, liczby badań albo wzorca braków. Nie wiemy też, ilu U jest chorych.
3. Bo MICE dopasowany na całym outer-train widziałby rekordy, które w danym foldzie wewnętrznym są walidacją. To wyciek przez preprocessing.
4. Bo w foldach mogą się różnić czynniki zakłócające (epoka NEURO, liczba rekordów), co zwiększa zmienność wyniku między foldami z powodów niezwiązanych z modelem.
5. Bo NEURO ma 2 razy więcej rekordów, a maksimum z wielu pomiarów jest systematycznie wyższe. Model dostaje sygnał „liczba badań”, a nie „choroba”.
6. `wczytaj_podzial()` rzuci `RuntimeError`, bo odcisk danych się nie zgadza. Trzeba świadomie przebudować podział (`podzial.py`) i odnotować to jako zmianę protokołu.

</details>
