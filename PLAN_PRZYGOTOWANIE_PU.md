# Plan: przygotowanie do modelowania Positive-Unlabeled

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Branch:** `pu-modeling-setup-lk`
**Zakres:** etap 0 oraz punkty 1–5 — wszystko, co musi być gotowe, zanim wytrenujemy pierwszy model
**Data:** 11.09.2026

Trening modeli zaczyna się dopiero w `PLAN_MODELOWANIE_PU.md` (punkty 6–9).

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

W całym planie pierwszym momentem, w którym cokolwiek się faktycznie uczy, jest punkt 6. Wszystko wcześniej to rozstrzygnięcia protokołu, infrastruktura i definicje metryk.

---

## Stan wyjściowy — zweryfikowany

Liczby przeliczone bezpośrednio na `aneurysm_concatted_cleaned.csv`:

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

**1. Brak tasowania w podziale.** W `aneurysm_sgkf_mice_pipeline.py` (linia 137) podział powstaje jako `StratifiedGroupKFold(n_splits=N_SPLITS)`, a sygnatura w sklearn 1.8.0 to `(n_splits=5, shuffle=False, random_state=None)`. Domyślnie **nie ma tasowania**, więc podział zależy wyłącznie od kolejności wierszy w CSV, a stała `RANDOM_STATE = 42` nie wpływa na foldy — działa tylko wewnątrz MICE.

**2. Źródło danych jest poprawne — i tak musi zostać.** Pipeline czyta `aneurysm_concatted_cleaned.csv`, czyli zbiór **przed** imputacją, i imputuje wewnątrz foldów. W repo leży też `imputation-final/results/aneurysm_imputed_cleaned.csv` — zbiór zaimputowany globalnie, przed podziałem. Tego pliku **nie wolno** użyć do walidacji modeli, bo statystyki ze zbioru walidacyjnego weszły tam do uzupełniania braków treningowych. Warto to zapisać wprost, bo plik jest kuszący i gotowy.

**3. Selekcja cech była robiona z użyciem etykiety na całym zbiorze.** CRP, MONO i %MONO usunięto na podstawie korelacji ze zmienną `label` liczonej na skonsolidowanym zbiorze. Formalnie jest to selekcja cech wykorzystująca etykietę i dane, które później trafiają do walidacji. Praktyczny wpływ jest prawdopodobnie znikomy — odrzuciliśmy cechy o korelacji bliskiej zeru, czyli szum, a nie najsilniejsze predyktory — ale trzeba to jawnie opisać w ograniczeniach raportu, a nie przemilczeć.

---

## Etap 0 — rozstrzygnięcia, które muszą zapaść przed pisaniem kodu

To nie jest praca programistyczna, tylko decyzje protokołu. Bez nich każdy dalszy wynik będzie niejednoznaczny.

**0.1. Rozstrzygnąć 63 pacjentów występujących jednocześnie w KOR i NEURO.** Skoro grupujemy po `patient_id`, jeden pacjent nie może być naraz pozytywny i nieoznaczony. Do wyboru: uznać ich w całości za pozytywnych, usunąć z analizy albo potraktować osobno. Decyzja musi być świadoma i opisana.

**0.2. Ustalić chronologię diagnozy.** Mamy `examination_date` (format `YYYY-MM-DD`, zero braków), ale nie mamy daty rozpoznania tętniaka. Trzeba ustalić, czy pomiary NEURO pochodzą sprzed diagnozy, czy z okresu po niej — a jeśli po, to model może uczyć się skutków leczenia i hospitalizacji zamiast ryzyka. To jest największe potencjalne źródło przekłamania w całym projekcie.

**0.3. Zdefiniować agregację rekordów do pacjenta.** Skoro metryki liczymy na pacjentach (punkt 4), musimy z wielu wyników jednego pacjenta zrobić jeden. Kandydaci: średnia, mediana, maksimum albo ostatni pomiar przed diagnozą. Przy medianie 1 rekordu i maksimum 38 wybór realnie zmienia wynik dla najczęściej badanych pacjentów.

**0.4. Sprawdzić selekcję cech pod kątem wycieku.** Zweryfikować, czy usunięcie CRP, MONO i %MONO dało się zrobić bez użycia etykiety, i zdecydować: albo powtarzamy selekcję wewnątrz foldów, albo zostawiamy jak jest i opisujemy jako ograniczenie.

**0.5. Analiza wrażliwości dla wartości skrajnych.** Sprawdzić WBC, Na, K i KREA — czy ekstremalne wartości to stany kliniczne, czy błędy wpisu, i jak ich obecność wpływa na ranking pacjentów.

**0.6. Domknąć konfigurację podziału.** Włączyć realnie działające `shuffle` i `random_state`, sprawdzić na siatce kilku ziaren i kilku wartości `n_splits`, czy rozmiary foldów i proporcje klas są powtarzalne — również w scenariuszu z częścią pozytywnych już ukrytych, bo to dodatkowo zmniejsza widoczną klasę pozytywną. Efektem jedna udokumentowana decyzja.

---

## Plan prac

### 1. Foldy i imputacja per fold

Kolejność split → scaler → MICE na train → transform na walidacji jest poprawna i zostaje bez zmian. Te same zewnętrzne foldy muszą być używane przez wszystkie modele. Do zrobienia:

- włączyć `shuffle=True` i `random_state` zgodnie z decyzją z 0.6,
- **zapisać przypisanie pacjentów do foldów** jako osobną tabelę (`patient_id → fold`) — to artefakt, który potem trzeba dołączyć do raportu,
- wystawić pipeline jako importowalną funkcję i zapisać gotowe foldy na dysk (imputacja całości trwa ok. 23 minut, nie ma sensu jej powtarzać),
- rozszerzyć asercje: brak wspólnych pacjentów, brak NaN, **brak wartości nieskończonych**, **brak wartości poza dozwolonym zakresem**, brak wartości ujemnych,
- trzymać się `aneurysm_concatted_cleaned.csv` jako jedynego wejścia.

**Zastrzeżenie, które zmienia architekturę:** gotowych foldów zewnętrznych **nie wolno** użyć jednocześnie do strojenia w Optunie i do raportowania wyniku — to zawyża jakość. Strojenie wymaga **zagnieżdżonej walidacji**: wewnątrz każdego foldu zewnętrznego osobne inner Group CV, z preprocessingiem dopasowywanym od nowa w każdym foldzie wewnętrznym. Pipeline z tego punktu musi to umożliwiać, nie tylko produkować pięć gotowych podziałów.

### 2. Ukrywanie części pozytywnych — kontrolowany test odzyskiwania

Z grupy NEURO losujemy część pacjentów i odbieramy im widoczną etykietę. To test tego, czy model potrafi ich odzyskać — nie substytut grupy kontrolnej (patrz zastrzeżenie na górze dokumentu).

Warunki, które muszą być spełnione:

- losujemy **całych pacjentów**, nigdy pojedynczych rekordów; wszystkie rekordy pacjenta dostają ten sam status,
- ukrywanie następuje **przed** rozpoczęciem walidacji krzyżowej,
- w danych trzymamy trzy kolumny: `true_label` (prawda), `observed_label` (to, co widzi model) i `is_hidden` (flaga kontrolna),
- testujemy **kilka udziałów ukrywania** — 20%, 40%, 60%,
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
| Precision@k liczona wyłącznie na kontrolowanych ukrytych | uzupełnienie powyższej |
| Lift@k względem losowego wyboru | pokazuje, czy model jest lepszy od losowania |
| PR-AUC | ważniejsza od ROC-AUC przy tej dysproporcji klas |
| ROC-AUC | pomocnicza |
| Recall znanych pozytywnych | kontrola, czy nie tracimy oczywistych przypadków |
| Odsetek pacjentów kierowanych na badanie | koszt operacyjny decyzji |

**Accuracy wypada z głównego porównania** — przy 91% klasy nieoznaczonej jest myląca.

Dodatkowo: ROC-AUC i PR-AUC liczone z KOR jako klasą negatywną trzeba w raporcie opisywać jako metryki **P-vs-U**, a nie jako skuteczność wykrywania choroby. To jest inna wielkość.

### 5. Funkcja celu — bez arbitralnych wag

Wcześniejszy pomysł „metryki ważonej" z ręcznie dobraną karą za false positives trzeba porzucić w tej formie. Nie da się sensownie ustalić kary za FP, skoro prawdziwych FP w KOR nie potrafimy rozpoznać. Wagi mogą wynikać albo z kosztu diagnostyki, albo z dostępnej przepustowości badań — a nie z naszego przeczucia.

Główna funkcja celu dla Optuny:

```
RecallHidden@q
```

gdzie `q` to odsetek pacjentów U, których realnie dałoby się skierować na diagnostykę (np. 5% albo 10%).

Wariant rozszerzony, jeśli chcemy pilnować też znanych pozytywnych:

```
S(q) = α · RecallHidden@q + (1 − α) · RecallKnown
```

**Warunek krytyczny:** wartości `q` i `α` ustalamy **przed** zobaczeniem wyników i już ich nie ruszamy. Dobranie ich po eksperymentach byłoby dopasowaniem kryterium do rezultatu.

---

## Zamrożenie przed treningiem

Zanim ruszy cokolwiek z `PLAN_MODELOWANIE_PU.md`, zamrażamy i zapisujemy do repo trzy rzeczy:

1. **foldy** — konkretny podział pacjentów, nie sam przepis na niego,
2. **maski ukrywania** — wszystkie udziały i wszystkie ziarna,
3. **główne metryki wraz z `q` i `α`**.

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
                              └─> PLAN_MODELOWANIE_PU.md
```

---

## Co ma być efektem tego etapu

1. Zapisane decyzje z etapu 0 — zwłaszcza rozstrzygnięcie 63 pacjentów i chronologii diagnozy.
2. Udokumentowana konfiguracja podziału (`n_splits`, `shuffle`, `random_state`) wraz z danymi, na podstawie których ją wybraliśmy.
3. Pipeline foldów jako importowalna funkcja, obsługujący walidację zagnieżdżoną, plus zapisane foldy i tabela `patient_id → fold`.
4. Zbiór z kolumnami `true_label`, `observed_label`, `is_hidden` oraz komplet masek ukrywania (udziały × ziarna), identyczny dla wszystkich modeli.
5. Zaimplementowany i przetestowany zestaw metryk pacjentowych wraz z funkcją celu `RecallHidden@q`.
6. Spisane ograniczenia: założenie SCAR, selekcja cech z użyciem etykiety, brak możliwości zmierzenia rzeczywistego FP rate w KOR.
