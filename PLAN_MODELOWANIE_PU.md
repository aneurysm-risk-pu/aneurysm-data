# Plan etapu: modelowanie Positive-Unlabeled

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Branch:** `pu-modeling-setup-lk`
**Data:** 11.09.2026

---

## Gdzie jesteśmy w projekcie

To jest **trzeci etap projektu** i pierwszy po raporcie przejściowym z 12.06.2026.

| Etap | Zakres | Status |
|---|---|---|
| 1 | EDA, czyszczenie i konsolidacja kohort KOR + NEURO | zamknięty |
| 2 | Imputacja braków (benchmark KNN / MICE / MissForest) + kontrolowany podział StratifiedGroupKFold | zamknięty, opisany w raporcie przejściowym |
| **3** | **Modelowanie Positive-Unlabeled — trenowanie, ewaluacja i porównanie modeli** | **← tutaj jesteśmy** |
| 4 | Raport końcowy | przed nami |

Raport przejściowy kończy się rozdziałem „Przyszłe etapy prac", w którym zapowiedziano dokładnie to, co niżej rozpisujemy. Wyniki tego etapu będą trzonem raportu końcowego.

Ten branch obejmuje **punkty 0 i 1** — fundament, na którym stanie reszta etapu. Punkty 2–9 realizujemy dalej, na kolejnych branchach.

---

## Stan wyjściowy — zweryfikowany

Liczby przeliczone bezpośrednio na `aneurysm_concatted_cleaned.csv`, a nie przepisane z raportu:

| Co | Wartość |
|---|---|
| Wiersze × kolumny | 78 197 × 39 (35 cech + 4 kolumny meta) |
| `label=0` (KOR, grupa nieoznaczona) | 71 546 |
| `label=1` (NEURO, potwierdzeni) | 6 651 |
| Unikalnych pacjentów | 40 924 |
| Pacjentów z co najmniej jednym wierszem `label=1` | 1 823 |
| Pacjentów mających jednocześnie oba labele | 63 |

Te 63 przypadki mają znaczenie dla punktu 2: skoro `StratifiedGroupKFold` grupuje po `patient_id`, ukrywanie pozytywnych musi działać **na poziomie pacjenta**, a nie pojedynczego wiersza — inaczej ten sam pacjent byłby jednocześnie ukryty i widoczny.

Środowisko: `scikit-learn 1.8.0`, `pandas 3.0.3`, `optuna 4.8.0`, `tabulate 0.10.0`. `xgboost` i `lightgbm` nie są jeszcze zainstalowane — będą potrzebne dopiero w punkcie 7.

---

## Co wyszło przy weryfikacji istniejącego kodu

W `aneurysm_sgkf_mice_pipeline.py` (linia 137) podział powstaje tak:

```python
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)
```

Sygnatura w sklearn 1.8.0 to `StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)`, czyli **domyślnie nie ma tasowania**. W efekcie:

- podział zależy wyłącznie od kolejności wierszy w pliku CSV,
- stała `RANDOM_STATE = 42` zdefiniowana w skrypcie **nie wpływa na foldy** — działa tylko wewnątrz MICE,
- powtarzalność, którą uznaliśmy za oczywistą, wynika z determinizmu kolejności danych, a nie z kontrolowanego ziarna losowego.

To nie unieważnia niczego, co zrobiliśmy do tej pory (na tych foldach nie był jeszcze trenowany żaden model), ale trzeba to naprawić, zanim foldy staną się podstawą całego etapu 3. Stąd punkt 0.

---

## Plan prac

### 0. Weryfikacja i domknięcie schematu StratifiedGroupKFold

Obecne 5 foldów to wariant bazowy z raportu przejściowego — przyjęty jako standardowy domyślny wybór i nigdy nietestowany pod kątem stabilności. Zanim zamrozimy go na cały etap modelowania, chcemy: włączyć realnie działające `shuffle` i `random_state`; sprawdzić na siatce kilku ziaren i kilku wartości `n_splits`, czy rozmiary foldów i proporcje klas są powtarzalne; a także przeliczyć to samo w scenariuszu, w którym część pozytywnych jest już ukryta (punkt 2), bo to dodatkowo zmniejsza widoczną klasę pozytywną. Efektem ma być jedna udokumentowana decyzja o finalnej konfiguracji (`n_splits`, `shuffle`, `random_state`), obowiązująca w punktach 1–9.

### 1. Imputacja per fold — utrwalenie przyjętego schematu

Sam mechanizm mamy już wdrożony (rozdz. 5.1 raportu) i go nie zmieniamy: podział, potem scaler i MICE dopasowane wyłącznie na foldzie treningowym, część walidacyjna jedynie transformowana. Do zrobienia są trzy rzeczy porządkujące. Po pierwsze, przepięcie stałych na konfigurację ustaloną w punkcie 0. Po drugie, wystawienie pipeline'u jako importowalnej funkcji, żeby Optuna i porównanie modeli korzystały z tych samych foldów bez kopiowania kodu. Po trzecie, zapis gotowych foldów na dysk — imputacja całego zbioru trwa około 23 minut i nie ma sensu powtarzać jej przy każdym eksperymencie. Przy okazji dochodzi trzecia asercja kontrolna: obok braku wspólnych `patient_id` i braku NaN po imputacji sprawdzamy też brak wartości ujemnych.

### 2. Sztuczne ukrywanie części chorych — nasz substytut grupy kontrolnej

Nie mamy pewnej grupy zdrowych, więc nie da się wprost sprawdzić, czy model faktycznie wyłapuje chorych, a nie tylko dopasowuje się do etykiet. Dlatego z grupy NEURO wylosujemy część pacjentów i przeniesiemy ich do puli nieoznaczonej, ukrywając prawdziwą etykietę pod `0`. Losowanie i podmiana działają na poziomie pacjenta, nie wiersza. W danych trzymamy dwie kolumny: `true_label` z prawdziwym statusem, widoczną tylko dla nas do kontroli eksperymentu, oraz `label` — to, co faktycznie widzi model podczas treningu.

### 3. Ponowne spojrzenie na false positives

Model będzie uczony tak, jakby ukryci pacjenci byli zdrowi, więc formalnie ich wykrycie zaliczy się jako false positive względem `label`. To jednak nie błąd, tylko dokładnie to, czego oczekujemy — w ten sposób mierzymy, ile podrzuconych przypadków model odzyskał. Skoro grupa nieoznaczona i tak nie jest czystą grupą kontrolną, część pozostałych false positives też może mieć uzasadnienie kliniczne, a nie być przypadkową pomyłką.

### 4. Metryki oceny

Samo ROC-AUC nie wystarczy, bo nie mówi wprost, ile ukrytych jedynek zostało odzyskanych, a accuracy przy dysproporcji klas (grupa nieoznaczona to ponad 91% zbioru) łatwo zawyżyć, przewidując same zera. Do porównania modeli zestawiamy: ROC-AUC, PR-AUC, recall klasy pozytywnej, odsetek odzyskanych ukrytych `1`, accuracy jako metrykę pomocniczą oraz osobną metrykę ważoną opisaną niżej.

### 5. Metryka ważona — własna funkcja oceny

Chcemy metrykę, która premiuje odzyskanie ukrytych `1` wyraźnie mocniej niż klasyczna accuracy, a jednocześnie karze pominięcie znanych i ukrytych przypadków pozytywnych. Kara za pozostałe false positives ma być umiarkowana, bo w PU learning część z nich może realnie wskazywać pacjentów wymagających dalszej diagnostyki, a nie czysty szum.

### 6. Strojenie hiperparametrów w Optunie

Optuna optymalizuje metrykę ważoną, a nie samą accuracy, i robi to w obrębie foldów przygotowanych w punktach 0 i 1. Zależy nam na modelu, który jak najlepiej odróżnia `0` od `1` i jednocześnie odzyskuje jak najwięcej podrzuconych przypadków.

### 7. Porównanie kilku modeli klasyfikacyjnych

Na tym samym pipeline testujemy regresję logistyczną, Random Forest oraz XGBoost/LightGBM — każdy na identycznych foldach i z tym samym zestawem metryk, żeby porównanie było uczciwe. Tu dochodzą dwie nowe zależności do środowiska.

### 8. Analiza pacjentów oznaczonych jako wysokie ryzyko

Po treningu przyglądamy się pacjentom z grupy `label=0`, którym model przypisał wysokie prawdopodobieństwo ryzyka — zarówno tym podrzuconym sztucznie, jak i pozostałym. W praktyce klinicznej taka grupa odpowiadałaby pacjentom kierowanym na dalszą diagnostykę obrazową, co jest właściwym celem projektu: model ma wspierać selekcję do badań, a nie stawiać diagnozę samodzielnie.

### 9. Efekt końcowy etapu

Pipeline modelowania oparty na gotowych foldach, wyniki kilku modeli klasyfikacyjnych, komplet metryk (klasyczne i ważona), ocena skuteczności odzyskiwania ukrytych pozytywnych, interpretacja false positives jako kandydatów do dalszej diagnostyki oraz wybór najlepszego modelu do raportu końcowego.

---

## Kolejność i zależności

Punkty nie są w pełni swobodne — część z nich musi poprzedzać inne:

```
0 (konfiguracja foldów)
└─> 1 (foldy + imputacja, zapisane na dysk)
     └─> 2 (ukryte pozytywne w etykiecie)
          ├─> 3 (interpretacja — nie osobne zadanie, tylko sposób czytania wyników)
          └─> 4, 5 (metryki + metryka ważona)
               └─> 6 (Optuna optymalizuje metrykę z punktu 5)
                    └─> 7 (porównanie modeli na dostrojonych hiperparametrach)
                         └─> 8 (analiza wysokiego ryzyka na wytrenowanych modelach)
                              └─> 9 (podsumowanie etapu)
```

Punkt 1 obowiązuje przez cały etap jako założenie metodologiczne, a nie jednorazowe zadanie: każdy kolejny eksperyment korzysta z tych samych foldów, żeby wyniki modeli dało się ze sobą porównywać.
