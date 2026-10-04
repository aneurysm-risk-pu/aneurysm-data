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
- **Ocena przez ukrywanie.** Bierzemy część znanych chorych, udajemy, że są nieoznaczeni (`observed_label = 0`, `true_label = 1`), i sprawdzamy, czy model wypchnie ich na górę rankingu. To pomysł prowadzącego i rdzeń ewaluacji (`4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 3.2).
- **SCAR** (*Selected Completely At Random*) to założenie wielu metod PU, że znani chorzy są losową próbką wszystkich chorych. U nas to nieprawda: NEURO to chorzy objawowi, którzy trafili do szpitala. Tego nie da się naprawić, da się tylko uczciwie opisać.

---

## 3. Mapa repozytorium i kolejność czytania

```
data/raw/              dwa surowe CSV (nie ruszamy)
data/interim/          stany pośrednie czyszczenia
data/imputation-inputs/aneurysm_concatted.csv     ← stan po etapie 1 (38 cech, przed imputacją)
data/processed/aneurysm_sgkf_input.csv            ← WEJŚCIE do podziału i modelowania (po decyzjach 04.10)
data/processed/aneurysm_concatted_cleaned.csv     ← historyczny zestaw 35 cech (raport przejściowy, selekcja cofnięta)
1-data-preparation/    czyszczenie i EDA (Twoje skrypty + notebooki Liwii)
2-imputation/          benchmark imputacji; final/ to wspólny kod, reszta to fazy indywidualne
3-sgkf-split/          etap 3, dwie części:
   przygotowanie/      A: decyzje dotyczące danych → aneurysm_sgkf_input.csv (+ spis zmian, podkładka liczbowa)
   sgkf/               B: zamrożony podział + MICE w foldach (38 cech) + ocena imputacji + diagnostyka wyników
   uruchom_sgkf.py     cały etap jednym poleceniem
4-pu-setup/            PLAN_MODELOWANIA.md (jedyny dokument: co dalej) + analiza reguły agregacji
0626_, 1026_*.md       raport przejściowy (06.2026), realizacja etapów 1–3 (10.2026)
PRZEWODNIK_LK.md       ten plik
```

Kolejność czytania przy powrocie:
1. `1026_REALIZACJA_DO_SGKF.md`: co zrobiliśmy i kto co robił.
2. `3-sgkf-split/RAPORT_SGKF_MICE.md`: sekcja „W skrócie”, potem 2 (decyzje dotyczące danych) i 4 (wyniki imputacji).
3. `4-pu-setup/PLAN_MODELOWANIA.md`: co dalej, założenia wejściowe i pytania do prowadzącego.

---

## 4. Wyciek danych: cztery rodzaje, wszystkie u nas wystąpiły albo groziły

Wyciek (*leakage*) to sytuacja, w której informacja z danych testowych dostaje się do treningu. Model wygląda wtedy lepiej, niż jest. W projekcie zetknęliśmy się z czterema odmianami:

| Rodzaj | Jak wyglądał u nas | Jak to rozwiązano |
|---|---|---|
| **pacjent w obu zbiorach** | pacjent ma do 38 rekordów; losowy podział rozrzuciłby je między train i test, a model „rozpoznawałby osobę” | podział po pacjentach (Group) |
| **preprocessing przed podziałem** | imputacja całego zbioru, potem podział: MICE uczyłby się też na rekordach testowych | scaler i MICE dopasowywane **wewnątrz foldu**, tylko na train |
| **etykieta w selekcji cech** | CRP, MONO i %MONO usunięte na podstawie korelacji z `label` na całym zbiorze | powrót do 38 cech; selekcja, jeśli w ogóle, tylko wewnątrz foldu |
| **strojenie na danych oceny** | przebieg 1 benchmarku imputacji: Optuna i ocena na tej samej masce | osobne maski: seed 43 do strojenia, seed 42 do oceny. W modelowaniu analogicznie: walidacja zagnieżdżona |

Zasada ogólna: **wszystko, co się uczy (a scaler i imputer też się uczą!), musi widzieć tylko train.** Tego samego dotyczy walidacja zagnieżdżona (`4-pu-setup/PLAN_MODELOWANIA.md`, sekcja 3.1): foldy wewnętrzne do strojenia są budowane wyłącznie z outer-train, a preprocessing jest dopasowywany od nowa w każdym z nich.

---

## 5. Przygotowanie danych i podział: co się zmieniło i dlaczego (v1 → v2)

### 5.1 Jak działa `StratifiedGroupKFold`

- **Group:** dostaje `groups=patient_id` i nigdy nie rozdziela grupy między train i test.
- **Stratified:** próbuje, żeby proporcja `y` w każdym foldzie była zbliżona do globalnej.
- Działa na **rekordach**, więc wyrównuje proporcję rekordów pozytywnych (8,5%), a nie pacjentów (4,45%).
- Przyjmuje **jedną** zmienną stratyfikującą.

### 5.2 Trzy subtelności, które wyszły przy audycie

**(a) `random_state` bez `shuffle=True` nic nie robi.** W sklearn większość splitterów ignoruje `random_state`, gdy `shuffle=False`. W v1 stała `RANDOM_STATE = 42` działała tylko w MICE. Podział wynikał z kolejności wierszy w CSV, a plik jest posortowany: najpierw cały KOR, potem NEURO. Przesortowanie pliku po cichu zmieniłoby foldy.

**(b) Algorytm SGKF z tasowaniem zmienił się między wersjami sklearn.** Ta sama konfiguracja (shuffle, seed 42) daje u Ciebie na Windowsie (sklearn 1.8) rozstęp udziału pozytywnych 0,11 pp, a na Macu (sklearn 1.6, bo jest tu tylko Python 3.9) 0,80 pp. To **inne podziały**. Wniosek ogólny: **przepis na podział nie jest powtarzalny między środowiskami, a zapisany przydział jest.** Stąd `3-sgkf-split/sgkf/results/pacjent_fold.csv` jako źródło prawdy. `StratifiedKFold` na pacjentach okazał się stabilny między wersjami: zapisany plik z Windowsa odtworzył się na Macu w 100%.

**(c) Wyrównanie klas nie oznacza wyrównania foldów.** Wrześniowa rekomendacja (pacjenci, stratyfikacja po etykiecie) daje idealny balans klas (0,01 pp), ale przy niektórych ziarnach jeden fold dostaje wyraźnie więcej NEURO z epoki KOR albo więcej NEURO z wieloma rekordami: do 5 i 8 pp różnicy. Dlaczego to ważne: te dwie zmienne to znane skróty (sekcja 1). Jeśli fold 2 ma więcej NEURO z lat 2020–2021, to model będzie tam miał **trudniej** (mniej epokowego skrótu), więc wyniki między foldami będą się różnić z powodów niezwiązanych z modelem. Większa wariancja oznacza mniej wiarygodne porównanie modeli.

Siatka stabilności z września tego nie wychwyciła, bo mierzyła udział lat 2020–2021 we **wszystkich** pacjentach foldu, a 96% z nich to KOR, który i tak jest cały w tym oknie. Problem siedział w **podgrupie** NEURO. Lekcja: miary balansu licz tam, gdzie siedzi zmienność.

### 5.3 Stratyfikacja wielokluczowa: trik

`StratifiedKFold` przyjmuje jedną kolumnę `y`, ale to może być **dowolna etykieta kategorii**. Sklejasz kilka zmiennych w jeden klucz:

```
"1_pozaokno_n5+"  ← pozytywny, poza oknem czasowym KOR, 5+ rekordów
"0_okno_n1"       ← nieoznaczony, w oknie, 1 rekord
```

i stratyfikujesz po kluczu. Każda kombinacja jest rozkładana równo między foldy. Ograniczenie: każda warstwa musi mieć co najmniej `n_splits` członków, a w praktyce kilka razy więcej. Stąd `min_warstwa = 20` i scalanie małych warstw do `<etykieta>_inne`. U nas najmniejsza warstwa ma 24 osoby, więc scalanie nie zachodzi. Mieszanych (63) nie wkładamy do klucza, bo powstałyby warstwy po kilka osób.

Dlaczego podział na tabeli pacjentów wciąż jest „Group”: skoro jeden wiersz to jeden pacjent, nie da się go rozdzielić. Warunek grupowania jest spełniony z konstrukcji. Potem `indeksy_foldow()` tłumaczy przydział pacjentów na indeksy rekordów.

### 5.4 Odcisk danych

`podzial_meta.json` zawiera SHA-256 pliku wejściowego (`aneurysm_sgkf_input.csv`). `wczytaj_podzial()` przelicza go i odmawia działania, jeśli dane się zmieniły, bo przydział do starych danych nie musi pasować do nowych. Przygotowanie danych (część A) zapisuje ten plik, więc każda zmiana reguł też zmienia odcisk i wymusza przebudowanie podziału. Szczegół, który zjadłby Ci godzinę: git na Windowsie domyślnie zamienia `\n` na `\r\n` przy checkoucie, więc surowe bajty CSV różnią się między komputerami. Dlatego hash liczony jest po ujednoliceniu końców linii.

### 5.5 Dlaczego przygotowanie danych jest osobnym krokiem

Część A (`3-sgkf-split/przygotowanie/przygotuj_dane.py`) czyta dane źródłowe, stosuje decyzje i **zapisuje plik** `data/processed/aneurysm_sgkf_input.csv`. Część B (`3-sgkf-split/sgkf/`) czyta tylko ten plik. Korzyści:
- każdą decyzję widać w jednym miejscu, a jej skutek w spisie zmian (`usuniete_wartosci.csv`: która komórka, jaka wartość, jaka reguła),
- podział nie wie nic o regułach; wie tylko, że plik się zmienił, i wtedy odmawia działania,
- test `test_plik_w_repozytorium_zgodny_z_kodem` pilnuje, żeby plik w repo był dokładnie wynikiem kodu, czyli nikt go nie poprawił ręcznie.

### 5.6 Reguła dla kreatyniny: dlaczego przez eGFR

To dobry przykład, jak przy brudnych danych szukać **sprzeczności wewnętrznej** zamiast arbitralnego progu.

- eGFR laboratorium **wylicza z kreatyniny** (oraz wieku i płci): im wyższa kreatynina, tym niższe eGFR.
- Dla KREA 5–10 mg/dl mediana eGFR wynosi 8, czyli fizjologicznie poprawnie (ciężka niewydolność nerek).
- Dla KREA 20–50 mediana eGFR wynosi **60**, czyli prawidłowe nerki. To niemożliwe przy prawdziwej kreatyninie 20–50 mg/dl. Najprostsze wyjaśnienie: do kolumny trafiła wartość w µmol/l (norma 50–110), a eGFR policzono z prawidłowej wartości.
- Stąd reguła: **KREA > 50 → brak** (w mg/dl niemożliwe) i **KREA > 10 przy eGFR ≥ 30 → brak** (sprzeczność). KREA 25 przy eGFR 8 zostaje, bo to wiarygodny pacjent dializowany.
- Prosty próg „> 20” wyrzuciłby 83 takich wiarygodnych pacjentów i zostawił sprzeczne wartości 10–20.
- Nie przeliczamy ÷ 88,4, bo nie znamy jednostki konkretnego pomiaru. Zamieniamy na brak, a MICE odtworzy wartość z reszty profilu, w tym z eGFR tego rekordu.

Dwie pułapki, które przy tym wyszły:
- **eGFR jest cenzurowane**: laboratorium raportuje „≥ 60” jako 60 (56% rekordów) i „≥ 90” jako 90 dla CKD-EPI. Próg „eGFR ≥ 30” działa mimo tego, ale eGFR nie odróżnia nerek „dobrych” od „bardzo dobrych”.
- **Dane to średnie tygodniowe.** Niemożliwa średnia (K = 26) oznacza, że w tym tygodniu był niemożliwy pomiar. Możliwa średnia może jednak ukrywać pojedynczy błąd, którego już nie wykryjemy.

Pozostałe wartości niemożliwe: **K > 15 mmol/l i Na < 80 mmol/l**, czyli wartości niezgodne z życiem. Skrajne, ale możliwe zostają: K 9–15 (ciężka hiperkaliemia), Na 80–100, WBC > 200 (białaczka). Liwia czyściła kwantylami, ale granice były statystyczne (dla KOR górna granica K wynosiła 29), więc tych wartości nie złapała. Do tego NEURO czyściła ostrzej (3,2% wobec 0,8% rekordów), co jest kolejną asymetrią do opisania.

### 5.7 63 pacjentów w obu kohortach

Pacjent nie może być jednocześnie P i U, bo podział i etykieta są pacjentowe. Rozpoznanie tętniaka jest faktem, a „nieoznaczony” to tylko brak informacji, więc pacjent zostaje **pozytywny**. Jego rekordy KOR usuwamy, bo bez daty diagnozy nie wiadomo, czy są sprzed choroby. Gdyby zostały z etykietą 1, model uczyłby się, że rutynowy wynik sprzed lat to „profil chorego”. Dodatkowej kolumny w danych nie ma. Listę tych pacjentów zawiera spis `przygotowanie/results/usuniete_rekordy.csv`, gdyby trzeba było zrobić analizę wrażliwości bez nich.

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

### 6.5 Jak zmierzyć jakość imputacji (i dlaczego losowa maska przeszacowuje)

Prawdziwych brakujących wartości nie znamy, więc błędu imputacji nie da się policzyć wprost. Jedyna uczciwa metoda to ta z etapu 2: wziąć wiersze, w których **znamy** wszystkie wartości, część z nich ukryć, uzupełnić i porównać z prawdą. W foldach SGKF zrobiliśmy to out-of-fold: imputer dopasowany na treningu foldu, a oceniane kompletne wiersze części testowej (`sgkf/ocena_imputacji.py`).

Dwie lekcje:
- **RMSE w skali [0,1] zależy od skalera.** NEURO wyszło 0,080, mniej niż 0,098 w benchmarku. Nie znaczy to jednak, że imputacja NEURO się poprawiła: w etapie 2 NEURO miało własny skaler, a tu skaler pochodzi z treningu w 96% złożonego z KOR. Dlatego obok RMSE liczymy **stosunek do uzupełniania średnią**, niezależny od skali: 0,61 dla KOR i 0,65 dla NEURO, czyli praktycznie tak samo.
- **Losowa maska jest za łatwa.** Losowo ukryta komórka ma zwykle obok siebie resztę swojego panelu, np. ukryte HCT przy znanych HGB i RBC, więc łatwo ją odtworzyć. W prawdziwych danych braki idą całymi panelami: brakuje całego koagulogramu albo całego rozmazu. Przy takich realistycznych wzorcach MICE jest prawie równe średniej (0,98 w KOR, 0,89 w NEURO). Wniosek: dla brakujących paneli uzupełnienie to w praktyce „średnia z treningu”, a nie informacja o pacjencie.

### 6.6 Podpis imputacji: jak uzupełnienie może zdradzić kohortę

NRBC (erytroblasty) u dorosłych jest prawie zawsze 0: w 78% zmierzonych rekordów wynosi dokładnie 0. MICE nigdy nie wpisze dokładnego zera, tylko coś w rodzaju 0,009. NRBC brakuje w 48% rekordów NEURO i tylko w 3% KOR. Drzewo decyzyjne łatwo nauczy się reguły „NRBC różne od 0, ale mniejsze niż 0,05 → NEURO”. Taka reguła nie mówi nic o tętniaku, tylko o tym, że NRBC nie zmierzono.

Ogólna lekcja: **imputacja nie usuwa informacji o braku, tylko ją ukrywa.** Sam wzorzec braków odróżnia u nas kohorty z AUC 0,82. Lepiej tę informację kontrolować jawnie (wykluczyć cechę, dodać wskaźnik braku, porównać z modelem na cechach bez braków), niż liczyć, że zniknęła.

---

## 7. Dziennik decyzji: co, dlaczego, jaka alternatywa

| Decyzja | Dlaczego | Alternatywa i dlaczego nie |
|---|---|---|
| podział po pacjentach | jeden pacjent ma wiele rekordów | podział po rekordach: wyciek |
| stratyfikacja etykieta × okno × n_rek | wyrównuje znane skróty między foldami | tylko etykieta: foldy rozjeżdżają się do 8 pp |
| zapisany przydział | powtarzalność między komputerami i wersjami | przepis + seed: zależny od sklearn |
| MICE w foldzie | brak wycieku przez preprocessing | imputacja globalna: wyciek |
| MICE (ExtraTrees) zamiast MissForest | wygrywa na NEURO, szybszy, jeden imputer dla obu | MissForest lepszy tylko na KOR |
| jeden zestaw 38 cech, selekcja z etapu 1 cofnięta | selekcja po `label` na całym zbiorze to wyciek, w PU etykieta to kohorta, a granica wyboru była remisem rozstrzygniętym alfabetycznie | 35 cech: imputacja nieznacznie gorsza, a uzasadnienie niezależne od etykiety istnieje tylko dla MONO/%MONO (redundancja) |
| mediana jako agregacja do pacjenta | najmniej koreluje z liczbą badań | maksimum przenosi liczbę badań do 17 cech; „ostatni” znaczy co innego w każdej kohorcie |
| mieszani: pozytywni, rekordy KOR usunięte | rozpoznanie jest faktem; rekordy KOR mogą być sprzed choroby | zostawić KOR z etykietą 1: uczy „profilu chorego” na rutynowych wynikach; wykluczyć: −63 z 1 823 pozytywnych |
| KREA: > 50 albo > 10 przy eGFR ≥ 30 → brak | sprzeczność z eGFR (liczonym z kreatyniny) | próg „> 20”: wyrzuca wiarygodne niewydolności; przeliczanie ÷ 88,4: zgadywanie jednostki |
| K > 15, Na < 80 → brak; skrajne, ale możliwe zostają | tylko wartości niezgodne z życiem | szersze progi: usuwałyby prawdziwe ciężkie stany |
| rozjazd czasowy: bez korekty | dane są, jakie są; obcięcie do wspólnego okna zostawia 271 pozytywnych | ważenie po czasie: możliwe później (gniazdo w planie) |
| asymetria braków: bez korekty | raportowana w każdym foldzie | model na cechach o niskim odsetku braków jako wrażliwość |
| przygotowanie danych jako osobny krok z plikiem wynikowym | decyzje w jednym miejscu, spis zmian, odcisk pliku chroni podział | czyszczenie „w locie” przy wczytywaniu: decyzje ukryte w kodzie podziału |
| ocena imputacji protokołem z etapu 2, out-of-fold, plus realistyczne wzorce braków | porównywalność z benchmarkiem; realistyczne braki pokazują prawdziwą użyteczność imputacji | porównanie rozkładów uzupełnionych i zmierzonych: nie mierzy błędu, bo prawdy nie znamy |
| `q` nie jest hiperparametrem | strojenie `q` to dobieranie definicji sukcesu pod wynik | — |

---

## 8. Pułapki: rzeczy, które łatwo zepsuć

1. **Nie używaj `2-imputation/final/results/aneurysm_imputed_*.csv` do modeli.** To imputacja globalna, przydatna tylko jako walidacja metody.
2. **Nie licz podziału od nowa w kolejnych etapach.** Czytaj `3-sgkf-split/sgkf/results/pacjent_fold.csv` przez `wczytaj_podzial()`. Prototyp PU z września liczył własny podział; w nowym kodzie modelowania tego nie powtarzać.
3. **Metryki licz na pacjentach, nie na rekordach.** Inaczej pacjent z 38 rekordami waży 38 razy więcej.
4. **Nie dodawaj roku ani liczby rekordów jako cechy.** Ułatwiłoby to modelowi rozpoznanie kohorty, a nie choroby.
5. **Nie nazywaj wyniku „prawdopodobieństwem tętniaka”**, tylko *risk score*, dopóki nie jest skalibrowany.
6. **Pełny przebieg MICE to ~3 min na fold na Macu (M-series)**, około 16 min na wariant cech; z walidacją zagnieżdżoną wielokrotność tego. Do sprawdzania ścieżki: `--podprobka 3000 --szybkie-mice`.
7. **Nie poprawiaj ręcznie `data/processed/aneurysm_sgkf_input.csv`.** Zmień regułę w `przygotuj_dane.py` i uruchom `uruchom_sgkf.py`; test w części A wykryje każdą ręczną zmianę.
8. **Do modelu nie wchodzą kolumny meta**: `patient_id`, `custom_id`, `examination_date`, `label`. `kolumny_cech()` je pomija.

---

## 9. Praca na dwóch komputerach

| | Windows | Mac (ten) |
|---|---|---|
| Python | nowszy; sklearn 1.8.0, pandas 3.0.3 (wg wcześniejszego planu) | 3.9.6 systemowy, więc sklearn maks. 1.6.1 |
| środowisko | twoje | `.venv/` w repo (ignorowany przez git) |

Skutki i zalecenia:
- wyniki zależne od implementacji bibliotek (SGKF z tasowaniem, w mniejszym stopniu ExtraTrees) mogą się różnić, dlatego zapisujemy artefakty, a nie przepisy,
- zaimputowane foldy są poza gitem (~45 MB na wariant), więc na drugim komputerze odtwarza je `uruchom_sgkf.py`. Czy wyszło to samo, sprawdzisz, porównując pole `odcisk_wyniku` w `przebieg_imputacji_*.json`; jeśli się różni, to różnica numeryczna wersji sklearn, którą trzeba ocenić,
- GitHub na Macu: zalogowane `gh` (`gh auth status`), więc `git push` działa bez hasła,
- warto zainstalować na Macu nowszego Pythona (np. przez `uv` albo instalator z python.org) i przypiąć wersje w `requirements.txt` w korzeniu repo,
- `podzial_meta.json` zapisuje wersje bibliotek, więc zawsze widać, gdzie powstał artefakt.

Szybki start na Macu:

```bash
cd ~/repos/aneurysm-data
.venv/bin/python 3-sgkf-split/uruchom_sgkf.py --bez-mice   # część A + podział + testy (~20 s)
.venv/bin/python 3-sgkf-split/sgkf/podzial.py --sprawdz     # czy zapisany podział pasuje do danych
.venv/bin/python 3-sgkf-split/uruchom_sgkf.py               # całość z imputacją (~35 min)
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
7. Dlaczego KREA = 25 przy eGFR = 8 zostaje, a KREA = 25 przy eGFR = 60 znika?
8. Dlaczego zestawy 38 i 35 cech różnią się w praktyce głównie CRP, skoro różnią się trzema kolumnami?
9. Co było nie tak z selekcją cech z etapu 1, skoro dała „rozsądny” wynik?
10. RMSE imputacji NEURO wyszło niższe niż w benchmarku. Czy to znaczy, że imputacja NEURO jest teraz lepsza?
11. Model odróżnia NEURO od KOR samym wzorcem braków z AUC 0,82. Co z tego wynika dla modelowania?

<details>
<summary>Odpowiedzi</summary>

1. Bo `shuffle=False`: `random_state` jest wtedy ignorowany, a podział zależy od kolejności wierszy.
2. Niewiele. To separacja P od U, która może wynikać z epoki, hospitalizacji, liczby badań albo wzorca braków. Nie wiemy też, ilu U jest chorych.
3. Bo MICE dopasowany na całym outer-train widziałby rekordy, które w danym foldzie wewnętrznym są walidacją. To wyciek przez preprocessing.
4. Bo w foldach mogą się różnić czynniki zakłócające (epoka NEURO, liczba rekordów), co zwiększa zmienność wyniku między foldami z powodów niezwiązanych z modelem.
5. Bo NEURO ma 2 razy więcej rekordów, a maksimum z wielu pomiarów jest systematycznie wyższe. Model dostaje sygnał „liczba badań”, a nie „choroba”.
6. Test części A wykryje, że plik `aneurysm_sgkf_input.csv` nie odpowiada już danym źródłowym. Po ponownym przygotowaniu plik się zmieni, a `wczytaj_podzial()` rzuci `RuntimeError` (inny odcisk). Trzeba świadomie przebudować podział i odnotować to jako zmianę protokołu.
7. eGFR jest liczone z kreatyniny. Przy eGFR 8 wysoka kreatynina jest spójna (niewydolność nerek), a przy eGFR 60 (prawidłowe nerki) KREA 25 mg/dl jest sprzeczne, więc to najpewniej błąd jednostki.
8. Bo MONO i %MONO są prawie w całości wyliczalne z innych cech: rozmaz sumuje się do 100%, a MONO ≈ %MONO × WBC / 100. Realnie nowej informacji dokłada tylko CRP.
9. Trzy rzeczy: użyła etykiety na całym zbiorze, czyli wyciek; w PU etykieta to kohorta, więc kryterium wyrzuca cechy, które *nie* odróżniają kohort; a granica wyboru była remisem czterech cech rozstrzygniętym alfabetycznie, więc tą samą metodą mógł wypaść wiek pacjenta. „Rozsądny” wynik dla MONO i %MONO to zbieg okoliczności. Da się go obronić innym argumentem (redundancja), ale nie tym, którego użyto.
10. Nie wprost. RMSE w skali [0,1] zależy od skalera, a tu skaler pochodzi z danych w 96% KOR. Miara niezależna od skali (stosunek do uzupełniania średnią) daje 0,65 dla NEURO i 0,61 dla KOR, czyli imputacja jest dla obu kohort podobnie dobra.
11. Że wysokie AUC P-vs-U nie będzie dowodem wykrywania tętniaka. Informacja „które badania zlecono” sama odróżnia kohorty, a imputacja jej nie usuwa (np. podpis NRBC). Potrzebne są analizy wrażliwości: cechy o niskim odsetku braków, ewentualnie bez NRBC.

</details>

---

## 12. Ściąga na spotkanie z prowadzącym

Materiał dla prowadzącego: `1026_SPOTKANIE_PROWADZACY.md`. Poniżej pytania, które mogą paść, z krótkimi odpowiedziami.

| Pytanie | Odpowiedź w jednym, dwóch zdaniach | Gdzie dowód |
|---|---|---|
| Czemu nie zwykły StratifiedGroupKFold? | Dzielimy tabelę pacjentów `StratifiedKFold`, co jest równoważne grupowaniu, a pozwala stratyfikować po kilku zmiennych naraz i nie zależy od wersji sklearn (SGKF z tasowaniem daje inne foldy w 1.6 i 1.8). | `RAPORT_SGKF_MICE.md` 3.2–3.3 |
| Czy imputacja nie przecieka? | Scaler i MICE są dopasowywane tylko na treningu foldu, a test jest jedynie transformowany. Etykieta nie jest cechą imputera. | 4.1, testy |
| Skąd wiecie, że reguła KREA jest dobra? | eGFR liczy się z kreatyniny. Powyżej KREA 10 mg/dl mediana eGFR wraca do normy, czyli te dwa pomiary sobie przeczą. Wiarygodne skrajne przypadki (eGFR < 30) zostają. | 2.2, `analiza_decyzji.json` D |
| Dlaczego nie usuwacie wartości odstających szerzej? | Usuwamy tylko wartości niemożliwe do przeżycia albo sprzeczne z innym pomiarem; skrajne, ale możliwe stany zostają (K 9–15, WBC > 200). | 2.1 |
| Czemu 38 cech, skoro w czerwcu było 35? | Czerwcowa selekcja użyła etykiety na całym zbiorze, w PU etykieta to kohorta, a granicę rozstrzygnął remis alfabetyczny. Imputacja na 38 jest nieznacznie lepsza. | 2.6, 4.5 |
| Jak dobra jest imputacja? | Przy losowych brakach jak w benchmarku (0,61–0,65 błędu średniej, obie kohorty). Przy brakach całymi panelami prawie równa średniej. | 4.3 |
| Czy to znaczy, że model jest dobry, skoro AUC 0,93? | Nie. To rozdzielność kohort, a sam wzorzec braków daje 0,82, a epoka 0,83. Miarą sukcesu będzie odzyskiwanie ukrytych chorych, a nie AUC P-vs-U. | 5.2, plan 3.4 |
| Co z rozjazdem czasowym? | Zostaje bez korekty, jako ograniczenie, ale epoka jest wyraźnie widoczna w danych, więc proponujemy analizę wrażliwości we wspólnym oknie dat. | 5.2, 1026 6.1 |
| Ilu pacjentów i chorych jest w foldzie? | 8 185 pacjentów, około 365 pozytywnych; przy 40% ukrywania około 146 ukrytych na fold. | 3.4, plan 5 |
| Czy wynik jest powtarzalny? | Tak: przydział zapisany z odciskiem danych, a dwa niezależne przebiegi imputacji dały identyczne odciski wyników. | 3.5, 4.2 |

**Co chcesz wynieść ze spotkania**, czyli odpowiedzi, bez których nie zamrozimy planu modelowania: skąd jest KOR, skąd są wyniki NEURO, akceptacja decyzji o czasie, los NRBC oraz `q` i udział ukrywania.

