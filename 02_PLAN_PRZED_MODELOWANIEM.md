# Plan: przygotowanie do modelowania Positive-Unlabeled

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Branch:** `pu-modeling-setup-lk`
**Zakres:** etap 0 oraz punkty 1–5 — wszystko, co musi być gotowe, zanim wytrenujemy pierwszy model
**Data:** 11.09.2026

Trening modeli zaczyna się dopiero w `03_PLAN_MODELOWANIE.md` (punkty 6–9).

---

## Najważniejsze zastrzeżenie metodologiczne

Wcześniejsza wersja planu mieszała trzy różne rzeczy pod wspólnym szyldem: **klasyfikację P-vs-U**, **kontrolowany eksperyment z ukrytymi pozytywnymi** oraz **ocenę ryzyka klinicznego**. To trzy odrębne cele o różnych kryteriach sukcesu i nie wolno ich mierzyć jedną metryką.

Konsekwencja najważniejsza dla całego projektu:

> Ukrywanie pozytywnych jest **kontrolowanym testem odzyskiwania przypadków**, a nie sposobem na stworzenie grupy kontrolnej. Mówi nam, czy model potrafi odnaleźć chorego, któremu odebraliśmy etykietę. **Nie mówi**, ilu pacjentów KOR jest naprawdę zdrowych ani jaki jest rzeczywisty false-positive rate w KOR — bo tego z tych danych po prostu nie da się zmierzyć.

Cały dalszy protokół jest podporządkowany temu rozróżnieniu.

---

## Gdzie to jest w projekcie

| Etap | Zakres | Status |
|---|---|---|
| 1 | EDA, czyszczenie i konsolidacja kohort KOR + NEURO | zamknięty |
| 2 | Imputacja braków + kontrolowany podział StratifiedGroupKFold | opisany w raporcie przejściowym, **domykany w tym dokumencie** |
| 3 | Modelowanie Positive-Unlabeled | etap 0 i punkty 1–5 to przygotowanie (ten dokument), punkty 6–9 to trening i wnioski (osobny dokument) |
| 4 | Raport końcowy | przed nami |

Pierwszy model przypisujący risk score powstaje w punkcie 6. Wcześniej dopasowywane są już elementy preprocessingu, przede wszystkim scaler i imputer, dlatego również one muszą respektować granice foldów.

---

## Stan wyjściowy — zweryfikowany

Liczby przeliczone bezpośrednio na `data/processed/aneurysm_concatted_cleaned.csv`:

| Co | Wartość |
|---|---|
| Wiersze × kolumny | 78 197 × 39 (35 cech + 4 kolumny meta) |
| `label=0` (KOR, grupa nieoznaczona) | 71 546 wierszy / 39 164 pacjentów |
| `label=1` (NEURO, potwierdzeni) | 6 651 wierszy / 1 823 pacjentów |
| Unikalnych pacjentów łącznie | 40 924 |
| Pacjentów mających jednocześnie oba labele | 63 |
| Rekordów na pacjenta | min 1, mediana 1, średnia 1,91, **maks. 38** |
| Pacjentów z jednym rekordem | 24 027 |

**Asymetria, o której trzeba pamiętać przy każdej metryce:** pacjent NEURO ma średnio **3,65** rekordu, pacjent KOR **1,83**. Liczba wykonanych badań jest więc sama w sobie skorelowana z etykietą — chory bywa badany częściej. To ma dwie konsekwencje: metryki liczone na wierszach automatycznie przeważałyby klasę pozytywną, a sama liczba pomiarów jest potencjalnym wyciekiem sygnału, jeśli kiedykolwiek trafi do cech.

Środowisko: `scikit-learn 1.8.0`, `pandas 3.0.3`, `optuna 4.8.0`, `tabulate 0.10.0`.

---

## Co wyszło przy weryfikacji istniejącego kodu

**1. Brak tasowania w podziale.** W `3-sgkf-split/aneurysm_sgkf_mice_pipeline.py` podział powstaje jako `StratifiedGroupKFold(n_splits=N_SPLITS)`, a sygnatura w sklearn 1.8.0 to `(n_splits=5, shuffle=False, random_state=None)`. Domyślnie **nie ma tasowania**, więc podział zależy wyłącznie od kolejności wierszy w CSV, a stała `RANDOM_STATE = 42` nie wpływa na foldy — działa tylko wewnątrz MICE.

**2. Źródło danych jest poprawne — i tak musi zostać.** Pipeline czyta `data/processed/aneurysm_concatted_cleaned.csv`, czyli zbiór **przed** imputacją, i imputuje wewnątrz foldów. W repo leży też `2-imputation/final/results/aneurysm_imputed_cleaned.csv` — zbiór zaimputowany globalnie, przed podziałem. Tego pliku **nie wolno** użyć do walidacji modeli, bo statystyki ze zbioru walidacyjnego weszły tam do uzupełniania braków treningowych.

**3. Selekcja cech była robiona z użyciem etykiety na całym zbiorze.** CRP, MONO i %MONO usunięto na podstawie korelacji ze zmienną `label` liczonej na skonsolidowanym zbiorze. Formalnie jest to selekcja cech wykorzystująca etykietę i dane, które później trafiają do walidacji. Nie można z góry uznać wpływu za znikomy: dodatkowo etykieta odróżnia kohorty P i U, a nie potwierdzonych chorych i zdrowych. W analizie potwierdzającej należy użyć wszystkich 38 uprzednio ustalonych cech albo wykonywać selekcję wyłącznie wewnątrz foldów; samo opisanie ograniczenia jest słabszym wariantem.

**4. Kohorty są silnie rozdzielone w czasie.** Diagnostyka z `4-pu-setup/etap0_diagnostyka.py` pokazała, że 99,5% rekordów KOR pochodzi z lat 2020–2021, natomiast NEURO obejmuje lata 2000–2024. Tylko 680 z 6 651 rekordów NEURO mieści się w dokładnym oknie dat KOR. Porównanie grup NEURO z różnych okresów wskazuje na możliwy efekt kalendarza, laboratorium lub składu pacjentów, ale nie izoluje „czystego efektu epoki”.

---

## Etap 0 — rozstrzygnięcia, które muszą zapaść przed pisaniem kodu

To nie jest praca programistyczna, tylko decyzje protokołu. Bez nich każdy dalszy wynik będzie niejednoznaczny.

**0.1. Rozstrzygnąć 63 pacjentów występujących jednocześnie w KOR i NEURO.** Skoro grupujemy po `patient_id`, jeden pacjent nie może być naraz pozytywny i nieoznaczony. U 55 osób wszystkie rekordy KOR poprzedzają rekordy NEURO, ale oznacza to tylko zmianę kohorty źródłowej, nie potwierdzoną datę diagnozy. Do wyboru: nadać pacjentowi status pozytywny z osobną flagą pochodzenia rekordu, usunąć przypadki niejednoznaczne albo przeanalizować je osobno. Decyzja musi być świadoma i opisana.

**0.2. Rozstrzygnąć czas i pochodzenie kohort.** Mamy `examination_date` (format `YYYY-MM-DD`, zero braków), ale nie mamy daty rozpoznania tętniaka. Trzeba równolegle:

- ustalić, czy pomiary NEURO pochodzą sprzed diagnozy, z jej okresu czy po leczeniu,
- ograniczyć confounding kalendarzowy i źródłowy przez dopasowanie kohort lub restrykcję do wspólnego okna; dodanie roku jako zwykłej cechy nie rozwiązuje problemu i może ułatwić modelowi rozpoznawanie źródła,
- zaplanować analizę wrażliwości w oknie wspólnym, pamiętając, że pozostaje wtedy tylko 271 pacjentów NEURO.

To jest największe potencjalne źródło przekłamania w całym projekcie i wymaga konsultacji z opiekunem lub właścicielem danych.

**0.3. Zdefiniować jednostkę treningu i agregację predykcji do pacjenta.** Trzeba rozstrzygnąć, czy model dostaje jeden zagregowany profil pacjenta, czy rekordy tygodniowe. W drugim wariancie częściej badani pacjenci nie mogą automatycznie ważyć więcej w funkcji straty; należy zastosować wagi odwrotne do liczby rekordów pacjenta. Niezależnie od jednostki treningu końcowy risk score i wszystkie główne metryki muszą być pacjentowe. Kandydaci dla agregacji wyników to mediana lub maksimum; „ostatni pomiar przed diagnozą” jest możliwy dopiero po uzyskaniu wiarygodnej daty diagnozy.

**0.4. Poprawić selekcję cech pod kątem wycieku.** Usunięcie CRP, MONO i %MONO wykorzystywało etykietę na całym zbiorze. W analizie głównej trzeba albo wrócić do 38 cech bez tej selekcji, albo wykonywać selekcję wyłącznie wewnątrz foldów. Pozostawienie 35 cech można pokazać jako wcześniej ustalony wariant w analizie wrażliwości, ale samo opisanie leakage nie wystarcza dla analizy potwierdzającej. Wariant 38 cech nie wymaga pracy przy danych: plik `data/imputation-inputs/aneurysm_concatted.csv` (78 197 × 42) po usunięciu CRP, MONO i %MONO jest co do wartości identyczny z `data/processed/aneurysm_concatted_cleaned.csv`, więc różni się od niego wyłącznie tymi trzema kolumnami.

**0.5. Analiza wrażliwości dla wartości skrajnych i jednostek.** Sprawdzić WBC, Na, K i KREA — czy ekstremalne wartości to stany kliniczne, błędy wpisu albo mieszane jednostki. W szczególności KREA trzeba połączyć z informacją o jednostce z danych źródłowych; automatyczne dzielenie wartości powyżej arbitralnego progu byłoby niewystarczające. Następnie porównać ranking pacjentów przed i po uzgodnionej korekcie.

**0.6. Domknąć konfigurację podziału.** Najpierw utworzyć jedną etykietę pacjentową po decyzji 0.1, potem włączyć realnie działające `shuffle` i `random_state`. Sprawdzić na siatce kilku ziaren i kilku wartości `n_splits` rozmiary foldów, proporcje etykiety pacjentowej oraz rozkład lat. Foldy zamrozić przed maskami ukrywania; do stratyfikacji można użyć prawdziwego statusu znanych pozytywnych wyłącznie jako elementu konstrukcji benchmarku, ale model nie może go później zobaczyć.

---

## Plan prac

### 1. Foldy i imputacja per fold

Kolejność split → scaler → MICE na train → transform na walidacji jest poprawna i zostaje bez zmian. Te same zewnętrzne foldy muszą być używane przez wszystkie modele. Do zrobienia:

- włączyć `shuffle=True` i `random_state` zgodnie z decyzją z 0.6,
- **zapisać przypisanie pacjentów do foldów** jako osobną tabelę (`patient_id → fold`) — to artefakt, który potem trzeba dołączyć do raportu,
- wystawić pipeline jako importowalną funkcję i cache'ować osobno przekształcenia outer oraz inner foldów; outer-train zaimputowany przed inner CV nie może zastąpić preprocessingu inner foldów,
- rozszerzyć asercje: brak wspólnych pacjentów, brak NaN i wartości nieskończonych oraz kontrola dziedzin klinicznych. Wartości obserwowane mogą wyjść poza zakres min–max treningu, więc nie wolno odrzucać całego foldu tylko dlatego, że po skalowaniu leżą poza [0,1],
- rozważyć dopasowanie `MinMaxScaler` kolumnowo na wszystkich dostępnych wartościach train, zamiast wyłącznie na kompletnych wierszach; obecny wariant zachowuje zgodność z benchmarkiem, ale opiera skalę na selektywnej podpróbie,
- trzymać się danych **przed imputacją**; wybór wariantu 35 lub 38 cech zależy od decyzji 0.4.

**Zastrzeżenie, które zmienia architekturę:** gotowych foldów zewnętrznych **nie wolno** użyć jednocześnie do strojenia w Optunie i do raportowania wyniku — to zawyża jakość. Strojenie wymaga **zagnieżdżonej walidacji**: wewnątrz każdego foldu zewnętrznego osobne inner Group CV, z preprocessingiem dopasowywanym od nowa w każdym foldzie wewnętrznym. Pipeline z tego punktu musi to umożliwiać, nie tylko produkować pięć gotowych podziałów.

### 2. Ukrywanie części pozytywnych — kontrolowany test odzyskiwania

Z grupy NEURO losujemy część pacjentów i odbieramy im widoczną etykietę. To test tego, czy model potrafi ich odzyskać — nie substytut grupy kontrolnej (patrz zastrzeżenie na górze dokumentu).

Warunki, które muszą być spełnione:

- losujemy **całych pacjentów**, nigdy pojedynczych rekordów; wszystkie rekordy pacjenta dostają ten sam status,
- foldy są zamrażane na poziomie pacjentów przed maskami, a maska ukrywania powstaje przed treningiem dla danego scenariusza; ukryci pacjenci w outer test nigdy nie mogą być widocznymi pozytywnymi w outer train,
- w danych trzymamy trzy kolumny: `true_label` (prawda), `observed_label` (to, co widzi model) i `is_hidden` (flaga kontrolna),
- z góry wybieramy jeden udział główny, a pozostałe traktujemy jako analizę wrażliwości — np. 40% jako scenariusz główny oraz 20% i 60% jako dodatkowe,
- każdy udział powtarzamy na **5–10 ziarnach**, żeby wynik nie zależał od jednego losowania,
- **dokładnie te same maski** trafiają do wszystkich modeli, inaczej porównanie jest nieuczciwe.

**Założenie SCAR do opisania jako ograniczenie:** losowe ukrywanie zakłada, że znani pozytywni są losową próbką wszystkich pozytywnych. W naszych danych to wątpliwe — NEURO to kohorta oddziału neurologicznego, więc prawdopodobnie bardziej objawowa i częściej hospitalizowana niż przeciętny nierozpoznany chory. Tego nie da się naprawić dostępnymi danymi, ale trzeba to powiedzieć wprost w raporcie.

### 3. Terminologia — „false positive" wymaga poprawienia

Dotychczasowe nazewnictwo było nieścisłe i trzeba je ujednolicić w kodzie i w raporcie:

| Sytuacja | Poprawne określenie |
|---|---|
| Ukryty chory przewidziany jako dodatni | względem `observed_label` wygląda na FP, ale względem `true_label` jest **TP** — to sukces, nie błąd |
| Pacjent KOR z wysokim wynikiem | **nie** „false positive" — nie znamy jego prawdziwego statusu. Mówimy: „wysoko sklasyfikowany pacjent U" albo „kandydat do dalszej diagnostyki" |

To nie jest kosmetyka. Nazwanie pacjenta KOR false-positive'em sugeruje, że wiemy, iż jest zdrowy — a nie wiemy.

### 4. Metryki — liczone na pacjentach, nie na rekordach

Przy rozkładzie od 1 do 38 rekordów na pacjenta i średniej 3,65 w NEURO wobec 1,83 w KOR, metryki wierszowe dawałyby większą wagę pacjentom częściej badanym, czyli w praktyce chorym. **Wszystkie główne metryki liczymy po agregacji do pacjenta** (reguła z punktu 0.3).

Zestaw metryk:

| Metryka | Rola |
|---|---|
| Recall ukrytych pozytywnych @ top-k% pacjentów U | **główna** — ilu ukrytych chorych odzyskujemy w puli, którą realnie da się skierować na badania |
| Udział kontrolowanych ukrytych pozytywnych w top-k | uzupełnienie powyższej; nie jest estymatorem rzeczywistej precision, bo status pierwotnych KOR pozostaje nieznany |
| Lift@k względem losowego wyboru | pokazuje, czy model jest lepszy od losowania |
| PR-AUC P-vs-U / kontrolowana PR-AUC | ważniejsza od ROC-AUC, ale wymaga wskazania użytych etykiet i puli kandydatów |
| ROC-AUC P-vs-U / kontrolowana ROC-AUC | pomocnicza, z takim samym zastrzeżeniem |
| Recall znanych pozytywnych | kontrola, czy nie tracimy oczywistych przypadków |
| Odsetek pacjentów kierowanych na badanie | koszt operacyjny decyzji |

**Accuracy wypada z głównego porównania** — przy 91% klasy nieoznaczonej jest myląca.

Dodatkowo: ROC-AUC i PR-AUC liczone z KOR jako klasą negatywną trzeba w raporcie opisywać jako metryki **P-vs-U**, a nie jako skuteczność wykrywania choroby. To jest inna wielkość.

### 5. Funkcja celu — bez arbitralnych wag

Wcześniejszy pomysł „metryki ważonej" z ręcznie dobraną karą za false positives trzeba porzucić w tej formie. Nie da się sensownie ustalić kary za FP, skoro prawdziwych FP w KOR nie potrafimy rozpoznać. Wagi mogą wynikać albo z kosztu diagnostyki, albo z dostępnej przepustowości badań — a nie z naszego przeczucia.

Główna funkcja celu dla Optuny w kontrolowanym benchmarku:

```
RecallHidden@q
```

gdzie `q` to odsetek pacjentów U, których realnie dałoby się skierować na diagnostykę (np. 5% albo 10%).

Pula rankingowa musi być zdefiniowana identycznie w każdym eksperymencie: obejmuje pacjentów z `observed_label=0`, a licznik odzyskania korzysta wyłącznie z ukrytych pozytywnych. Wynik nie jest estymatorem czułości w całej populacji KOR.

Wariant rozszerzony, jeśli chcemy pilnować też znanych pozytywnych:

```
S(q) = α · RecallHidden@q + (1 − α) · RecallKnown
```

**Warunek krytyczny:** wartości `q` i `α` ustalamy **przed** zobaczeniem wyników i już ich nie ruszamy. Dobranie ich po eksperymentach byłoby dopasowaniem kryterium do rezultatu.

---

## Zamrożenie przed treningiem

Zanim ruszy cokolwiek z `03_PLAN_MODELOWANIE.md`, zamrażamy i zapisujemy do repo:

1. **foldy** — konkretny podział pacjentów, nie sam przepis na niego,
2. **maski ukrywania** — wszystkie udziały i wszystkie ziarna,
3. **główne metryki wraz z `q` i `α`**,
4. **scenariusz główny** — wariant kohorty, zestaw cech i udział ukrywania; pozostałe warianty są analizami wrażliwości,
5. **wersje środowiska i seedy**.

Od tego momentu te elementy są niezmienne. Każda ich modyfikacja po zobaczeniu wyników musi być odnotowana w raporcie jako zmiana protokołu.

---

## Kolejność i zależności

```
Etap 0 (decyzje protokołu: 0.1 - 0.6)
└─> 1 (foldy + imputacja per fold, z obsługą inner CV)
     └─> 2 (ukryte pozytywne: true_label / observed_label / is_hidden, maski x udziały x ziarna)
          └─> 3 (ujednolicenie terminologii)
               └─> 4 (metryki na poziomie pacjenta)
                    └─> 5 (funkcja celu: RecallHidden@q, ustalone q i alfa)
                         └─> ZAMROŻENIE
                              └─> 03_PLAN_MODELOWANIE.md
```

---

## Co ma być efektem tego etapu

1. Zapisane decyzje z etapu 0 — zwłaszcza rozstrzygnięcie 63 pacjentów i chronologii diagnozy.
2. Udokumentowana konfiguracja podziału (`n_splits`, `shuffle`, `random_state`) wraz z danymi, na podstawie których ją wybraliśmy.
3. Pipeline foldów jako importowalna funkcja, obsługujący walidację zagnieżdżoną, plus zapisane foldy i tabela `patient_id → fold`.
4. Bazowa tabela z `true_label` oraz manifesty masek zawierające `patient_id`, `observed_label`, `is_hidden`, udział ukrywania i seed; identyczne dla wszystkich modeli.
5. Zaimplementowany i przetestowany zestaw metryk pacjentowych wraz z funkcją celu `RecallHidden@q`.
6. Spisane ograniczenia: założenie SCAR, selekcja cech z użyciem etykiety, brak możliwości zmierzenia rzeczywistego FP rate w KOR.
7. Rozstrzygnięcie rozjazdu czasowego kohort oraz z góry zaplanowana analiza wrażliwości dla wspólnego okna dat.
