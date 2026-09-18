# Podział danych: StratifiedGroupKFold + MICE — metodologia

**Plik:** `aneurysm_sgkf_mice_pipeline.py`  
**Data:** 2026-06-11  
**Dane wejściowe:** `aneurysm_concatted_cleaned.csv`

---

## 1. Problem i motywacja

Dane zawierają **wielokrotne pomiary tych samych pacjentów** (`patient_id`). Naiwny podział losowy (np. `train_test_split`) mógłby umieścić różne pomiary tego samego pacjenta w zbiorze train i test jednocześnie, co powoduje **data leakage** — model "widzi" pacjenta podczas treningu i ocenia go w teście, co sztucznie zawyża metryki.

Dodatkowo zbiór jest **niezbalansowany** (`label=0` KOR vs `label=1` NEURO), więc losowy podział mógłby zaburzyć proporcje klas w foldach.

---

## 2. Zastosowane podejście

### `StratifiedGroupKFold` (n_splits=5)

Łączy dwa warunki jednocześnie:

| Warunek | Co zapewnia |
|---|---|
| **Group** | Wszystkie pomiary jednego pacjenta trafiają **wyłącznie** do train lub wyłącznie do test |
| **Stratified** | Proporcje klas (`label=0/1`) są zbliżone między train a test w każdym foldzie |

### Imputacja MICE wewnątrz pętli foldów

Imputacja wykonywana **po** podziale, osobno dla każdego foldu:

```
dla każdego foldu:
  1. split → X_train (z brakami), X_test (z brakami)
  2. MinMaxScaler.fit()  ← tylko na complete cases z X_train
  3. IterativeImputer.fit_transform(X_train_scaled)  ← MICE uczy się z train
  4. IterativeImputer.transform(X_test_scaled)       ← test imputowany modelem z train
  5. scaler.inverse_transform()  ← powrót do oryginalnej skali
```

Gdyby imputacja była wykonana **przed** podziałem (na całym zbiorze), statystyki z danych testowych wpłynęłyby na wypełnienie braków w train → leakage.

---

## 3. Parametry MICE

Parametry finalne z `2-imputation/RAPORT_IMPUTACJA.md` (przebieg 4 + walidacja cross-param):

| Parametr | Wartość |
|---|---|
| `estimator` | `ExtraTreesRegressor` |
| `max_iter` | 7 |
| `n_estimators` | 78 |
| `max_depth` | 10 |
| `min_value / max_value` | 0.0 / 1.0 (dane w skali [0,1]) |
| `random_state` | 42 |

**Uzasadnienie wyboru:** ExtraTrees wygrało z BayesianRidge we wszystkich przebiegach Optuna. Parametry wyznaczone dla KOR (`max_iter=7, n_est=78`) dają identyczne RMSE na NEURO (Δ=0.00001) przy 7× krótszym czasie — używamy jednego zestawu dla obu zbiorów.

---

## 4. Weryfikacja poprawności (asercje)

Dla każdego foldu skrypt sprawdza:

- `len(set(groups_train) ∩ set(groups_test)) == 0` — brak wspólnych `patient_id`
- `np.isnan(X_train_imp).sum() == 0` — brak NaN w train po imputacji
- `np.isnan(X_test_imp).sum() == 0` — brak NaN w test po imputacji

---

## 5. Wynik

Funkcja `main()` zwraca listę `splits` — 5 słowników z kluczami:

```
X_train, X_test   — DataFrame, zaimputowane, oryginalna skala
y_train, y_test   — Series z labelami (0/1)
groups_train, groups_test — patient_id
```

Gotowe do bezpośredniego użycia w pętli modelowania.
