# Dokumentacja pipeline'u przygotowania PU (punkty 1–5)

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zakres:** implementacja punktów 1, 2, 4 i 5 z `02_PLAN_PRZED_MODELOWANIEM.md`
**Branch:** `pu-pipeline-1-5-lk`
**Data:** 20.09.2026

Dokument opisuje, co dokładnie robi kod, jakie założenia są w nim zaszyte i gdzie przebiegają granice, których naruszenie oznaczałoby wyciek danych. Jest napisany tak, żeby dało się na jego podstawie przeprowadzić recenzję metodologiczną bez czytania całego kodu — ale każde twierdzenie wskazuje plik i funkcję, w której da się je sprawdzić.

**Czego ten kod jeszcze nie robi:** nie trenuje żadnego modelu. Pierwszy model powstaje w punkcie 6, opisanym w `03_PLAN_MODELOWANIE.md`.

---

## 1. Jak zweryfikować to, co tu opisano

```bash
python 4-pu-setup/tests/test_pu.py                                  # 17 testów, kilkanaście sekund
python 4-pu-setup/uruchom_pipeline.py artefakty                     # tabela foldów i maski, sekundy
python 4-pu-setup/uruchom_pipeline.py fold --podprobka 3000 --szybkie-mice
python 4-pu-setup/uruchom_pipeline.py demo --podprobka 3000 --szybkie-mice
```

Tryb `demo` przepuszcza przez cały pipeline **losowy** scoring. To celowe: sprawdzamy instalację metryk, a nie jakość modelu. Jeśli metryki są poprawne, losowy scoring musi dać `RecallHidden@q ≈ q` i `Lift ≈ 1`.

---

## 2. Stan implementacji

| Punkt planu | Status | Gdzie |
|---|---|---|
| 1. Foldy i imputacja per fold | ✅ zaimplementowane | `pu/foldy.py` |
| 2. Ukrywanie pozytywnych | ✅ zaimplementowane | `pu/ukrywanie.py` |
| 3. Terminologia false positives | ✅ (dokument, nie kod) | `02_PLAN_PRZED_MODELOWANIEM.md` |
| 4. Metryki pacjentowe | ✅ zaimplementowane | `pu/metryki.py` |
| 5. Funkcja celu | ✅ zaimplementowane | `pu/cel.py` |
| 6–9. Trening i wnioski | ⬜ przed nami | `03_PLAN_MODELOWANIE.md` |

Decyzje etapu 0 są **parametrami**, nie kodem. Trzy z nich czekają na spotkanie — patrz sekcja 11.

---

## 3. Przepływ danych

```
dane PRZED imputacją (aneurysm_concatted.csv, 78 197 × 42)
   │
   ├─ kontrola czasu        (0.2)  ── opcjonalny filtr wspólnego okna dat
   ├─ wartości skrajne      (0.5)  ── opcjonalna zamiana na braki
   ├─ etykieta pacjentowa   (0.1)  ── true_label, flaga mieszany
   │
   ▼
PODZIAŁ na foldy zewnętrzne (po pacjentach, stratyfikacja po true_label)
   │
   ├─ dla każdego foldu zewnętrznego:
   │     │
   │     ├─ hak kontroli kohort      ← gniazdo na wariant C z 0.2, dziś puste
   │     ├─ MinMaxScaler.fit         ← WYŁĄCZNIE trening
   │     ├─ MICE.fit                 ← WYŁĄCZNIE trening
   │     ├─ transform train i test
   │     ├─ inverse_transform skalera
   │     └─ agregacja do pacjenta    (0.3, mediana)
   │
   └─ wewnątrz treningu: foldy wewnętrzne, preprocessing od nowa w każdym
```

Kolejność jest nienaruszalna. Scaler i imputer **uczą się**, więc też muszą respektować granicę foldu — to nie są neutralne przekształcenia.

---

## 4. Granice wycieku — najważniejsza część do recenzji

Poniżej wypisane wprost, co widzi trening, a czego nie. To są miejsca, w których projekt najłatwiej byłoby zepsuć.

| Element | Uczy się na | Sprawdzalne w |
|---|---|---|
| `MinMaxScaler` | kompletne wiersze części treningowej foldu | `foldy.py:126` `_dopasuj_scaler` |
| `IterativeImputer` (MICE) | rekordy części treningowej foldu, po skalowaniu | `foldy.py:146` `przygotuj_fold` |
| Foldy wewnętrzne | wyłącznie pacjenci z outer-train | `foldy.py:219` `foldy_wewnetrzne` |
| Preprocessing inner | dopasowywany **od nowa** w każdym foldzie wewnętrznym | `foldy.py:219` |
| Stratyfikacja podziału | `true_label`, ale tylko do konstrukcji foldów | `foldy.py:67` `podzial_pacjentow` |
| Model (punkt 6) | `observed_label` — nigdy `true_label`, nigdy `is_hidden` | — |

**Zakaz zapisany wprost w kodzie:** nie wolno zaimputować całego outer-train raz i dopiero potem dzielić go na foldy wewnętrzne. Statystyki z części walidacyjnej inner weszłyby wtedy do uzupełniania braków treningowych. Test `test_foldy_wewnetrzne_nie_siegaja_poza_trening_zewnetrzny` pilnuje granicy pacjentów, ale kolejności operacji pilnuje wyłącznie struktura kodu — to miejsce warte uwagi recenzenta.

**Stratyfikacja po `true_label` wymaga komentarza.** Foldy powstają przed maskami ukrywania, więc do wyrównania klas używamy prawdziwego statusu pacjenta. Jest to element konstrukcji benchmarku, a nie informacja przekazywana modelowi. Model dostaje wyłącznie `observed_label`. Gdyby ktoś chciał być bardziej rygorystyczny, można stratyfikować po `observed_label` — kosztem gorszego zbilansowania foldów przy 60% ukrywania.

---

## 5. Moduły

### `pu/config.py` — decyzje protokołu

Jedna dataklasa `KonfiguracjaPU` z 24 polami. Każde pole odpowiada decyzji z etapu 0. Metoda `zmieniona(**kw)` robi wariant bez mutowania oryginału, `do_slownika()` zrzuca konfigurację do artefaktu.

Wartości domyślne na 20.09.2026:

| Pole | Wartość | Status decyzji |
|---|---|---|
| `zestaw_cech` | `"38"` | rekomendacja (0.4) |
| `mieszani` | `"pozytywni"` | **do rozstrzygnięcia** (0.1) |
| `kontrola_czasu` | `"brak"` | **do rozstrzygnięcia** (0.2) |
| `wartosci_skrajne` | `"zostaw"` | **do rozstrzygnięcia** (0.5) |
| `jednostka_treningu` | `"pacjent"` | rekomendacja (0.3) |
| `regula_agregacji` | `"mediana"` | rekomendacja (0.3) |
| `poziom_podzialu` | `"pacjent"` | rekomendacja (0.6) |
| `n_splits_outer` / `inner` | 5 / 3 | rekomendacja (0.6) |
| `shuffle` / `seed_podzialu` | `True` / 42 | rekomendacja (0.6) |
| `udzial_ukrycia_glowny` | 0,40 | rekomendacja (punkt 2) |
| `seedy_ukrycia` | 1–5 | rekomendacja (punkt 2) |
| `q` | 0,05 | **wartość tymczasowa** |
| `alfa` | 1,0 | czysty `RecallHidden@q` |

`q` to jedyna wartość, która nie ma uzasadnienia w danych — powinna wynikać z realnej przepustowości diagnostyki obrazowej. Do czasu ustalenia jest to placeholder.

### `pu/dane.py` — warstwa danych

- `wczytaj` — dane **przed** imputacją. Zaimputowany globalnie plik jest tu niedostępny celowo.
- `etykieta_pacjentowa` — jedna etykieta na osobę (`max` po rekordach), flaga `mieszany`, liczba rekordów.
- `zastosuj_kontrole_czasu` — wariant A z 0.2. Filtr przecięcia zakresów dat: 2019-12-29 – 2021-12-26. Zostawia 39 414 pacjentów, ale tylko **271 pozytywnych** zamiast 1 823.
- `zastosuj_zakresy` — wartości spoza granic przeżycia na `NaN`, żeby imputacja odtworzyła je z reszty profilu. Domyślnie wyłączone.
- `agreguj_do_pacjenta` — mediana albo średnia. Maksimum i „ostatni pomiar" są świadomie niedostępne, z uzasadnieniem w `ETAP0_USTALENIA.md`.
- `wagi_rekordowe` — wagi 1/n dla trybu rekordowego, żeby pacjent z 38 pomiarami nie ważył 38 razy więcej.

### `pu/foldy.py` — punkt 1

- `podzial_pacjentow` — `StratifiedKFold` na pacjentach (domyślnie) albo `StratifiedGroupKFold` na rekordach (wariant zgodny z dotychczasowym pipeline'em, zachowany do porównania).
- `przygotuj_fold` — pełny preprocessing jednego foldu, zwraca dataklasę `Fold`.
- `sprawdz_fold` — asercje: brak wspólnych pacjentów, brak `NaN`, brak wartości nieskończonych, brak ujemnych. Wyjście poza zakres min–max treningu daje **ostrzeżenie, nie błąd** — wartość walidacyjna ma prawo wykroczyć poza skalę treningu i nie jest to powód do odrzucenia foldu.
- `foldy_wewnetrzne` — inner CV budowane wyłącznie z pacjentów outer-train.
- `odcisk_konfiguracji` — skrót SHA-256 konfiguracji, żeby cache preprocessingu nie przeżył zmiany decyzji protokołu.
- `hak_kontroli_kohort` — **puste gniazdo** na wariant C z decyzji 0.2 (dopasowanie albo ważenie kohort po czasie i źródle). Sygnatura: `(df_train, df_test, cfg) -> (df_train, df_test, wagi_train)`. Wpięcie wariantu C nie wymaga przebudowy pipeline'u.

### `pu/ukrywanie.py` — punkt 2

`maska_ukrycia(pac, udzial, seed)` losuje **całych pacjentów** spośród pozytywnych i zwraca tabelę `patient_id, true_label, is_hidden, observed_label, udzial_ukrycia, seed_ukrycia`.

`sprawdz_maske` weryfikuje cztery niezmienniki: ukryty musi być prawdziwie pozytywny, ukryty musi mieć `observed_label = 0`, nieukryty musi mieć etykietę niezmienioną, pacjent nie może wystąpić dwa razy.

Maski są generowane globalnie, niezależnie od foldów. Jest to poprawne, bo foldy są rozłączne po pacjentach: ukryty pacjent jest ukryty wszędzie — w treningu figuruje jako nieoznaczony i w teście również. Dokładnie o to chodzi w teście odzyskiwania.

### `pu/metryki.py` — punkt 4

Wszystkie metryki działają na **pacjentach**. Pula rankingowa to pacjenci z `observed_label = 0` w danym foldzie testowym.

| Metryka | Definicja | Czego **nie** mierzy |
|---|---|---|
| `recall_hidden_at_q` | udział ukrytych pozytywnych w górnych `q` puli U | czułości w całej populacji KOR |
| `udzial_ukrytych_w_top` | jaka część wskazanych to kontrolowani ukryci | **nie jest** estymatorem precision — status reszty puli jest nieznany |
| `lift_at_q` | `udzial_ukrytych_w_top / bazowy udział ukrytych` | — |
| `recall_known` | ilu znanych pozytywnych osiąga próg górnego `q` | — |
| `roc_auc_p_vs_u`, `pr_auc_p_vs_u` | na etykietach widocznych | skuteczności wykrywania choroby — to metryki P-vs-U |
| `roc_auc_kontrolowana`, `pr_auc_kontrolowana` | wewnątrz puli U: ukryci vs reszta | — |

Dwie decyzje implementacyjne warte recenzji:

1. **Remisy rozstrzygane po `patient_id` rosnąco** (`_top_k`, `metryki.py:31`). Bez tego wynik zależałby od kolejności wierszy — ten sam błąd, co brak `shuffle` w podziale. Pilnuje tego `test_metryka_nie_zalezy_od_kolejnosci_wierszy`.
2. **Fold bez ukrytych zwraca `NaN`, nie zero.** Brak informacji to nie jest wynik zerowy, a uśrednianie zer zaniżyłoby wynik końcowy.

### `pu/cel.py` — punkt 5

```
S(q) = alfa · RecallHidden@q + (1 − alfa) · RecallKnown
```

Przy `alfa = 1` jest to czysty `RecallHidden@q`. `q` **nie jest** hiperparametrem Optuny — strojenie `q` oznaczałoby dobieranie definicji sukcesu pod wynik. `agreguj_po_foldach` uśrednia po foldach, pomijając te bez ukrytych.

---

## 6. Co jest osiągalne przy obecnych liczbach

Warto mieć te liczby przed spotkaniem, bo określają, co w ogóle da się zmierzyć.

Przy 40 924 pacjentach, 5 foldach, 40% ukrywania i `q = 0,05`:

| Wielkość | Na fold testowy |
|---|---|
| pacjentów | 8 185 |
| pozytywnych (true) | ~365 |
| ukrytych | ~146 |
| znanych pozytywnych | ~219 |
| pula U (ranking) | ~7 966 |
| pacjentów skierowanych (`q = 5%`) | ~399 |

Czyli model ma wskazać 399 osób, wśród których ukrytych chorych jest 146. Maksymalny możliwy `RecallHidden@q` to 1,0 — zadanie jest wykonalne, a nie z góry przegrane. Rozdzielczość metryki to 1/146 ≈ 0,7 punktu procentowego.

**Dla porównania, wariant A z decyzji 0.2** (wspólne okno dat) zostawia 271 pozytywnych pacjentów łącznie, czyli ~22 ukrytych na fold. Rozdzielczość metryki spada wtedy do 4,5 punktu procentowego i wyniki przestają być porównywalne między foldami. To jest konkretny, liczbowy argument przeciwko wariantowi A jako scenariuszowi głównemu.

---

## 7. Artefakty

Tryb `artefakty` zapisuje do `4-pu-setup/results/`:

| Plik | Zawartość | Rozmiar |
|---|---|---|
| `pacjent_fold.csv` | `patient_id → fold, true_label` | 521 KB, 40 924 wiersze |
| `maski_ukrycia.csv` | 15 kombinacji (3 udziały × 5 ziaren) | **21 MB**, 613 860 wierszy |
| `konfiguracja.json` | pełny zrzut protokołu | 846 B |

Rozkład pozytywnych między foldami: 0,0445 / 0,0446 / 0,0446 / 0,0446 / 0,0445. Foldy mają po 8 185 pacjentów (ostatni 8 184).

**Otwarta kwestia:** plik masek waży 21 MB, a maski są w pełni deterministyczne — odtwarzają się z `patient_id`, udziału i ziarna w ułamku sekundy. Rozważyć zapis samych parametrów zamiast pełnej tabeli.

---

## 8. Przebieg kontrolny — wyniki

Podpróbka 3 000 pacjentów, prawdziwy MICE w każdym z 5 foldów, scoring **losowy**:

| Fold | Pula U | Ukrytych | `RecallHidden@q` | Lift |
|---|---:|---:|---:|---:|
| 0 | 584 | 10 | 0,000 | 0,00 |
| 1 | 581 | 8 | 0,250 | 4,84 |
| 2 | 584 | 11 | 0,000 | 0,00 |
| 3 | 587 | 14 | 0,143 | 2,80 |
| 4 | 584 | 11 | 0,000 | 0,00 |

Średnia funkcja celu: **0,079** przy oczekiwanych 0,05. Rozrzut bierze się stąd, że na fold przypada 8–14 ukrytych pacjentów, a `q = 5%` z puli ~584 to 30 miejsc — jeden trafiony pacjent zmienia wynik o 0,1. Na pełnym zbiorze rozdzielczość jest siedmiokrotnie lepsza (sekcja 6).

Wniosek dla recenzji: metryki są podpięte poprawnie i nie wykazują biasu, ale **analiza wrażliwości na podpróbkach jest bezużyteczna** — przy małej liczbie ukrytych szum dominuje nad sygnałem.

---

## 9. Testy

17 testów, wszystkie przechodzą. Poniżej co dokładnie dowodzą.

| Test | Dowodzi |
|---|---|
| `test_model_idealny_odzyskuje_wszystkich_ukrytych` | metryka osiąga 1,0 przy idealnym rankingu |
| `test_model_odwrotny_nie_odzyskuje_nikogo` | metryka osiąga 0,0 przy odwrotnym |
| `test_metryka_nie_zalezy_od_kolejnosci_wierszy` | remisy rozstrzygane deterministycznie |
| `test_ukryci_nie_wchodza_do_puli_znanych` | ukryty pozytywny liczy się jako nieoznaczony |
| `test_brak_ukrytych_daje_nan_zamiast_zera` | brak informacji ≠ wynik zerowy |
| `test_funkcja_celu_*` (2) | `alfa` działa zgodnie z definicją |
| `test_agregacja_po_foldach_pomija_puste` | `NaN` nie zaniża średniej |
| `test_maska_ukrywa_wlasciwy_odsetek_pozytywnych` | udział ukrywania jest dokładny |
| `test_maska_jest_odtwarzalna_i_zalezy_od_ziarna` | powtarzalność przy tym samym ziarnie |
| `test_nigdy_nie_ukrywamy_negatywnego` | ukrywamy wyłącznie prawdziwie pozytywnych |
| `test_komplet_masek_ma_wszystkie_scenariusze` | 3 udziały × 5 ziaren, oznaczenie scenariusza głównego |
| `test_zaden_pacjent_nie_jest_w_dwoch_foldach` | rozłączność foldów |
| `test_fold_nie_ma_brakow_po_imputacji` | imputacja domyka wszystkie braki |
| `test_agregacja_daje_jeden_wiersz_na_pacjenta` | agregacja nie gubi ani nie dubluje pacjentów |
| `test_foldy_wewnetrzne_nie_siegaja_poza_trening_zewnetrzny` | **brak wycieku przez inner CV** |
| `test_poziom_rekordowy_tez_nie_dzieli_pacjenta` | wariant rekordowy też grupuje po pacjencie |

**Czego testy nie sprawdzają:** że kolejność operacji wewnątrz `przygotuj_fold` jest właściwa (scaler przed imputerem, oba na treningu), że imputacja jest sensowna klinicznie oraz że reguła agregacji jest właściwa. Pierwsze wynika ze struktury kodu, dwa pozostałe to kwestie merytoryczne, nie testowalne jednostkowo.

---

## 10. Świadome kompromisy

Rzeczy, które zrobiliśmy w określony sposób, mając świadomość alternatywy.

1. **Imputacja przed agregacją.** Imputujemy rekordy, potem agregujemy do pacjenta. Alternatywa — agregować najpierw, potem imputować — byłaby tańsza obliczeniowo i dałaby mniej braków do uzupełnienia, ale zerwałaby zgodność z benchmarkiem imputacji z raportu przejściowego. **Do rozważenia przy recenzji.**
2. **Scaler na kompletnych wierszach.** Zgodnie z benchmarkiem, ale skala opiera się wtedy na selektywnej podpróbie — kompletne wiersze mogą różnić się klinicznie od rekordów z brakami. Przełącznik `scaler_na_kompletnych=False` dopasowuje kolumnowo na wszystkich wartościach.
3. **MICE deterministyczny.** `sample_posterior=False`, czyli pojedyncza imputacja iteracyjna. Nie propagujemy niepewności imputacji do wyników końcowych.
4. **Brak kontroli epoki.** Hak jest pusty. Dopóki decyzja 0.2 nie zapadnie, model może uczyć się różnic między kohortami zamiast różnic klinicznych.
5. **Wartości skrajne nietknięte.** 86 wartości potasu powyżej 9 mmol/l i podejrzenie mieszanych jednostek KREA wchodzą obecnie do modelu bez zmian.
6. **Brak kalibracji.** Wynik modelu to `risk score`, nie prawdopodobieństwo. Kalibracja wymagałaby wiarygodnego class prior, którego nie mamy.

---

## 11. Otwarte decyzje i ich koszt

| Decyzja | Co przestawia | Koszt zmiany po fakcie |
|---|---|---|
| 0.1 status 63 pacjentów | `mieszani` | niski — dotyczy 0,15% pacjentów |
| 0.2 kontrola czasu, wariant A lub B | `kontrola_czasu` | niski — filtr, ale przeliczenie wszystkiego |
| 0.2 wariant C | `hak_kontroli_kohort` | **średni** — trzeba napisać funkcję, struktura gotowa |
| 0.3 reguła agregacji | `regula_agregacji` | niski |
| 0.4 zestaw cech | `zestaw_cech` | niski |
| 0.5 wartości skrajne | `wartosci_skrajne` | niski |
| 0.6 ziarno i liczba foldów | `seed_podzialu`, `n_splits_*` | niski |
| `q` | `q` | niski, ale **zmienia definicję sukcesu** — musi być zamrożone przed treningiem |

Koszt w każdym przypadku to nie kod, tylko czas obliczeń: pełna imputacja to ~23 minuty na fold, a przy walidacji zagnieżdżonej wielokrotność tego.

---

## 12. Checklista do recenzji

Pytania, na które recenzent powinien odpowiedzieć, żeby uznać punkty 1–5 za zamknięte:

1. Czy granica train/test w `przygotuj_fold` jest szczelna — scaler, imputer, agregacja?
2. Czy stratyfikacja po `true_label` przy budowie foldów jest akceptowalna, czy powinna iść po `observed_label`?
3. Czy imputacja przed agregacją to właściwa kolejność, czy odwrotna byłaby lepsza?
4. Czy pula rankingowa (`observed_label = 0` w foldzie testowym) jest zdefiniowana poprawnie dla `RecallHidden@q`?
5. Czy `udzial_ukrytych_w_top` jest dostatecznie jasno odróżniony od precision w dokumentacji i w nazwach?
6. Czy `q = 0,05` da się uzasadnić przepustowością diagnostyki, czy trzeba inną wartość?
7. Czy 40% jako główny udział ukrywania jest uzasadnione, skoro zmienia proporcję klas widzianą przez model?
8. Czy ostrzeżenie zamiast błędu przy wartościach poza zakresem treningu jest właściwą decyzją?
9. Czy maski powinny być zapisywane w całości (21 MB), czy odtwarzane z ziaren?
10. Czy brak kontroli epoki pozwala w ogóle uruchomić scenariusz główny, czy trzeba najpierw domknąć 0.2?

---

## 13. Pliki

```
4-pu-setup/
├── pu/
│   ├── config.py           116 linii   decyzje protokołu
│   ├── dane.py             133 linii   wczytanie, etykieta, filtry, agregacja
│   ├── foldy.py            249 linii   punkt 1
│   ├── ukrywanie.py         69 linii   punkt 2
│   ├── metryki.py          141 linii   punkt 4
│   └── cel.py               42 linie   punkt 5
├── tests/test_pu.py        240 linii   17 testów
├── uruchom_pipeline.py     183 linie   tryby artefakty / fold / demo
└── results/                            artefakty robocze
```
