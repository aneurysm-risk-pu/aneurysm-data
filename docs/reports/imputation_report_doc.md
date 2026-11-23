# Imputacja brakujących danych w zbiorach medycznych
## Dokumentacja techniczna — pipeline, eksperymenty i wyniki

**Projekt:** Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled  
**Data finalizacji:** 2026-06-04  
**Kod źródłowy:** `imputation-final/`

---

## 1. Problem brakujących danych

### 1.1 Skala braków

Dane kliniczne, z którymi pracujemy, pochodzą z dwóch zbiorów: ogólnokardiologicznego (KOR) i neurologicznego (NEURO). Każdy rekord odpowiada jednemu badaniu laboratoryjnemu pacjenta i zawiera 38 cech — wyniki morfologii krwi, biochemii, koagulologii, wiek i płeć.

Problem polega na tym, że w warunkach klinicznych nie wszystkie badania są zlecane każdemu pacjentowi. Niektóre testy (szczególnie koagulologia: PT, APTT, INR) wykonywane są tylko przy konkretnych wskazaniach medycznych. W efekcie olbrzymia część rekordów jest niekompletna — nie z powodu błędu pomiaru, ale dlatego że badanie po prostu nie zostało wykonane.

| Zbiór | Łączna liczba rekordów | Rekordy kompletne | Odsetek kompletnych |
|---|---|---|---|
| KOR (kardiologiczno-ogólny) | 71 546 | 20 663 | **28,9%** |
| NEURO (neurologiczny) | 6 651 | 780 | **11,7%** |
| **Łącznie** | **78 197** | **21 443** | (13,4% wartości brakujących) |

Bez imputacji modele uczenia maszynowego musiałyby pracować na mniej niż 30% danych (w przypadku KOR) lub niecałych 12% (NEURO), albo tracić informację ze wszystkich niekompletnych wierszy. Dlatego imputacja była warunkiem koniecznym przed przystąpieniem do właściwego modelowania.

### 1.2 Charakter brakujących danych

Braki nie są losowe w sensie statystycznym (ang. Missing Completely At Random). Pacjenci, u których zlecono pełny panel badań, to zazwyczaj inna subpopulacja niż ci z niekompletnymi wynikami — np. starsi, ciężej chorzy lub hospitalizowani dłużej. Ma to konkretne konsekwencje metodologiczne omówione w sekcji 5.2.

---

## 2. Wybrane metody imputacji

Porównaliśmy trzy podejścia, które różnią się zarówno mechanizmem uzupełniania braków, jak i złożonością obliczeniową.

### 2.1 KNN — K najbliższych sąsiadów

Najprostsza z testowanych metod. Dla każdej brakującej wartości algorytm znajduje *k* rekordów o najbardziej podobnych wartościach w pozostałych kolumnach (według odległości euklidesowej) i oblicza ważoną średnią ich wartości na danej kolumnie.

Parametry optymalizowane:
- `n_neighbors` — liczba sąsiadów (zakres: 1–30)
- `weights` — sposób ważenia sąsiadów: `uniform` (każdy liczy jednakowo) lub `distance` (bliżsi ważą więcej)

### 2.2 MICE — iteracyjna imputacja przez regresję

MICE (ang. *Multiple Imputation by Chained Equations*) to podejście iteracyjne: każda kolumna z brakami jest kolejno imputowana przez model regresyjny wytrenowany na pozostałych kolumnach. Proces powtarzany jest przez wiele rund (ang. *iterations*), aż wartości się ustabilizują.

W naszej implementacji użyliśmy `IterativeImputer` ze scikit-learn z dwoma opcjami modelu regresji:
- **BayesianRidge** — regresja bayesowska (prosta, szybka)
- **ExtraTrees** — lasy losowe z losowanymi podziałami (mocniejszy model, ale wolniejszy)

Parametry optymalizowane:
- `estimator_type` — typ regresora: BayesianRidge lub ExtraTrees
- `max_iter` — liczba rund iteracyjnej imputacji (zakres: 5–80)
- `n_estimators` — liczba drzew w ExtraTrees (zakres: 10–200)
- `max_depth` — głębokość drzew (ustalona na stałe: 10, żeby ograniczyć overfitting)

Ważna właściwość: ustawienie `min_value=0.0, max_value=1.0` w `IterativeImputer` gwarantuje, że żadna zaimputowana wartość nie wyjdzie poza zakres [0, 1].

### 2.3 MissForest — iteracyjne lasy losowe

MissForest działa analogicznie do MICE, ale jako regressor zawsze używa lasu losowego (`RandomForestRegressor`). Jest koncepcyjnie prostszy (jeden typ modelu), ale często skuteczniejszy na dużych zbiorach, bo lasy losowe lepiej wyłapują nieliniowe zależności między zmiennymi laboratoryjnymi.

Parametry optymalizowane:
- `n_estimators` — liczba drzew w lesie (zakres: 10–300)
- `max_depth` — głębokość drzew (zakres: 3–15)
- `max_iter` — liczba rund (zakres: 3–15)

---

## 3. Przygotowanie danych i schemat ewaluacji

### 3.1 Preprocessing

Przed imputacją każdy zbiór był przygotowywany w ten sam sposób (plik `prepare_data.py`):

```python
def load_complete(name: str) -> dict:
    df_raw = pd.read_csv(DATASETS[name])

    # Usuń kolumny identyfikacyjne i daty
    drop = [c for c in DROP_COLS if c in df_raw.columns]
    df_raw = df_raw.drop(columns=drop)   # usuwa: patient_id, custom_id, examination_date

    # Zostaw tylko kolumny numeryczne (38 cech)
    df_num = df_raw.select_dtypes(include=[np.number])

    # Filtracja do rekordów bez żadnych braków
    df_complete = df_num.dropna().reset_index(drop=True)

    # Normalizacja MinMax do zakresu [0, 1]
    scaler = MinMaxScaler()
    scaled_arr = scaler.fit_transform(df_complete.values.astype(float))
    df_scaled = pd.DataFrame(scaled_arr, columns=df_complete.columns)

    return {"df_complete": df_complete, "df_scaled": df_scaled, "scaler": scaler, ...}
```

Normalizacja do zakresu [0, 1] jest tutaj kluczowa z dwóch powodów:
1. Wszystkie metryki błędu (RMSE, MAE) są liczone w tej samej skali dla każdej zmiennej — wyniki są porównywalne niezależnie od jednostek (np. PLT w tysiącach vs GLU w mg/dL)
2. Możemy ustawić twarde ograniczenia `min_value=0.0` w imputerze — żadna wartość nie stanie się ujemna

### 3.2 Protokół ewaluacji przez maskowanie

Ponieważ nie znamy "prawdziwych" wartości dla rzeczywistych braków, ewaluację przeprowadzamy na danych, które są kompletne — sztucznie usuwamy część wartości i sprawdzamy, jak dobrze metoda je odtwarza.

```python
def create_mask(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Tworzy boolowską macierz braków: True = wartość ukryta."""
    rng = np.random.default_rng(seed)
    return rng.random(df.shape) < frac   # ~10% wartości

def apply_mask(df_scaled: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Zwraca kopię danych z NaN w zamaskowanych pozycjach."""
    arr = df_scaled.values.astype(float).copy()
    arr[mask] = np.nan
    return pd.DataFrame(arr, columns=df_scaled.columns)
```

Schemat działania:
- Maskujemy **10% wartości** losowo w całej macierzy danych
- Ta sama maska jest używana dla wszystkich trzech metod — wyniki są bezpośrednio porównywalne
- Każda metoda próbuje odtworzyć ukryte wartości, a my mierzymy błąd tylko na tych pozycjach

### 3.3 Miary jakości imputacji

```python
def evaluate_imputation(df_original, imputed_arr, mask, columns) -> dict:
    df_imputed = pd.DataFrame(imputed_arr, columns=columns)
    orig_arr = df_original.values
    return {
        "RMSE":      compute_rmse(orig_arr, imputed_arr, mask),   # pierwiastek z MSE
        "MAE":       compute_mae(orig_arr, imputed_arr, mask),    # średni błąd bezwzględny
        "KL_mean":   compute_kl_mean(df_original, df_imputed),   # wierność rozkładów kolumn
        "negatives": count_negatives(imputed_arr),                # wartości < 0 (powinno być 0)
    }
```

| Miara | Co mierzy |
|---|---|
| **RMSE** | Dokładność punktowa — jak blisko zaimputowana wartość jest prawdziwej. Zakres [0, 1], niżej = lepiej |
| **MAE** | Podobnie jak RMSE, ale mniej karze za duże odchylenia jednostkowe |
| **KL mean** | Czy rozkład statystyczny zmiennych po imputacji jest podobny do oryginału. Wartość 0 = identyczne rozkłady |
| **Negatives** | Sanity check — czy pojawiły się wartości ujemne (fizycznie niemożliwe dla wyników lab.) |

### 3.4 Rozdzielenie optymalizacji od ewaluacji (zapobieganie leakage)

To jeden z kluczowych elementów metodologicznych. Optymalizacja hiperparametrów i finalna ewaluacja muszą używać **różnych masek**, żeby parametry nie były "dopasowane" do konkretnego wzorca braków.

```
dane kompletne (df_scaled)
│
├── maska Optuna  (seed = 43)   ← Optuna nigdy nie widzi maski ewaluacyjnej
│   └── każdy trial optymalizuje na tej masce
│
└── maska ewaluacyjna  (seed = 42)   ← finalne porównanie metod
    └── metoda imputuje dane z tą maską, mierzymy RMSE/MAE/KL
```

W pierwszym podejściu ta separacja była nieobecna — oba procesy używały seed=42. Wyniki z tego przebiegu zostały odrzucone (szczegóły w sekcji 4).

---

## 4. Przebieg eksperymentów

### 4.1 Optymalizacja hiperparametrów przez Optunę

Każda metoda była optymalizowana niezależnie za pomocą biblioteki **Optuna** z samplerem TPE (ang. *Tree-structured Parzen Estimator*). TPE to inteligentny sampler — zamiast losowo próbować konfiguracje, buduje probabilistyczny model przestrzeni hiperparametrów i kieruje poszukiwania w stronę obiecujących regionów.

Dodatkowe usprawnienie dla dużego zbioru KOR (20 663 wierszy): każdy pojedynczy trial Optuna używał losowego podzboru **5 000 wierszy**. Finalna imputacja do ewaluacji wykonywana była na pełnym zbiorze. Dzięki temu optymalizacja KNN zajmowała minuty zamiast godzin.

Ogólny schemat funkcji optymalizacji (na przykładzie KNN):

```python
def optimize_and_impute(df_scaled, mask, n_trials=30, seed=42, val_rows=None):
    # Maska do finalnej imputacji (seed=42 — ewaluacja)
    df_masked_full = apply_mask(df_scaled, mask)

    # Osobna maska do trialu Optuna (seed=43 — zapobiega leakage)
    mask_val_full = create_mask(df_scaled, frac=mask.mean(), seed=seed + 1)

    # Opcjonalnie: podzbiór wierszy dla szybkiej walidacji
    if val_rows and len(df_scaled) > val_rows:
        idx = rng.choice(len(df_scaled), size=val_rows, replace=False)
        df_val = df_scaled.iloc[idx].reset_index(drop=True)
        mask_val = mask_val_full[idx]

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputed = imputer.fit_transform(apply_mask(df_val, mask_val))
        return compute_rmse(df_val.values, imputed, mask_val)  # minimalizujemy RMSE

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Finalna imputacja z najlepszymi parametrami na pełnym zbiorze
    best = study.best_params
    final_imputer = KNNImputer(n_neighbors=best["n_neighbors"], weights=best["weights"])
    imputed_arr = final_imputer.fit_transform(df_masked_full)
    return imputed_arr, best
```

Analogiczna struktura obowiązuje dla MICE i MissForest — różni się tylko definicja `objective` i budowa imputer'a.

---

### 4.2 Przebieg 1 — wersja bazowa (odrzucona z powodu błędu metodologicznego)

Pierwsze uruchomienie ujawniło poważny błąd: Optuna optymalizowała parametry na tej samej masce (seed=42), której używaliśmy do finalnej ewaluacji. Oznacza to, że parametry były "przefitowane" pod konkretny wzorzec braków — każda metoda dostała niejawną podpowiedź o tym, które wartości są ukryte. Wyniki z tego przebiegu zostały odrzucone w całości.

Drugi problem: model ExtraTrees w MICE nie miał ograniczonej głębokości drzew (`max_depth`), co mogło prowadzić do overfittingu w trakcie optymalizacji.

| Problem | Konsekwencja |
|---|---|
| Optuna seed=42 = ewaluacja seed=42 | Leakage hiperparametrów — wyniki zawyżone |
| Brak limitu `max_depth` w ExtraTrees | Potencjalny overfitting modelu regresji |

---

### 4.3 Przebieg 2 — korekta błędów

Zmieniliśmy dwie rzeczy:
- Optuna teraz używa maski z seed=43, ewaluacja dalej seed=42 — pełna separacja
- Dodano `max_depth=10` jako stały limit dla ExtraTrees w MICE
- Rozszerzono zakresy hiperparametrów MissForest (po analizie pierwszych wyników okazało się, że optymalne wartości były blisko granic zakresu)

| Parametr | Zakres v1 | Zakres v2 | Zmiana |
|---|---|---|---|
| MICE `max_depth` | bez limitu | **10 (stały)** | nowe ograniczenie |
| MICE `max_iter` | [5, 40] | [5, 60] | rozszerzono |
| MissForest `n_estimators` | [10, 150] | [10, **300**] | rozszerzono |
| MissForest `max_depth` | [5, 25] | [3, **15**] | zmieniono zakres |
| MissForest `max_iter` | [3, 8] | [3, **15**] | rozszerzono |

---

### 4.4 Przebieg 3 — główne wyniki (seed=42)

Po korekcie uruchomiliśmy pełny benchmark. Poniżej finalna przestrzeń poszukiwań i wyniki dla każdej metody.

**Budżet prób Optuna:** KNN = 30 prób, MICE = 20 prób, MissForest = 15 prób.

Definicja funkcji celu dla MICE (warunkowy hiperparametr `n_estimators`):

```python
def objective(trial):
    estimator_type = trial.suggest_categorical(
        "estimator_type", ["BayesianRidge", "ExtraTrees"]
    )
    max_iter = trial.suggest_int("max_iter", 5, 60)

    # n_estimators jest istotne tylko dla ExtraTrees — parametr warunkowy
    if estimator_type == "ExtraTrees":
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
    else:
        n_estimators = 10  # ignorowane dla BayesianRidge

    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=10, random_state=seed, n_jobs=-1
        ) if estimator_type == "ExtraTrees" else BayesianRidge(),
        max_iter=max_iter,
        min_value=0.0,   # gwarantuje brak wartości ujemnych
        max_value=1.0,
        random_state=seed,
    )
    imputed = imputer.fit_transform(apply_mask(df_val, mask_val))
    return compute_rmse(df_val.values, imputed, mask_val)
```

Definicja funkcji celu dla MissForest:

```python
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    max_depth    = trial.suggest_int("max_depth",    3,  15)
    max_iter     = trial.suggest_int("max_iter",     3,  15)

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        ),
        max_iter=max_iter,
        min_value=0.0, max_value=1.0,
        random_state=seed,
    )
    imputed = imputer.fit_transform(apply_mask(df_val, mask_val))
    return compute_rmse(df_val.values, imputed, mask_val)
```

#### Wyniki — KOR (n = 20 663 rekordów kompletnych)

| Metoda | RMSE ↓ | MAE ↓ | KL mean ↓ | Ujemne | Czas |
|---|---|---|---|---|---|
| **MissForest** ✓ | **0,0783** | **0,0308** | **0,0036** | 0 | 107 min |
| MICE | 0,0865 | 0,0315 | 0,0036 | 0 | 70 min |
| KNN | 0,1022 | 0,0494 | 0,0049 | 0 | 23 min |

Najlepsze znalezione hiperparametry:
- **MissForest:** `n_estimators=118`, `max_depth=15` ⚠️ (osiągnął górną granicę zakresu!), `max_iter=12`
- **MICE:** ExtraTrees, `max_iter=23`, `n_estimators=54`
- **KNN:** `n_neighbors=28`, `weights=distance`

#### Wyniki — NEURO (n = 780 rekordów kompletnych)

| Metoda | RMSE ↓ | MAE ↓ | KL mean ↓ | Ujemne | Czas |
|---|---|---|---|---|---|
| **MICE** ✓ | **0,0978** | **0,0432** | **0,0116** | 0 | 30 min |
| MissForest | 0,1017 | 0,0540 | 0,0129 | 0 | 17 min |
| KNN | 0,1291 | 0,0752 | 0,0156 | 0 | <1 min |

Najlepsze znalezione hiperparametry:
- **MICE:** ExtraTrees, `max_iter=18`, `n_estimators=100` ⚠️ (osiągnął górną granicę zakresu!)
- **MissForest:** `n_estimators=142`, `max_depth=13`, `max_iter=5`
- **KNN:** `n_neighbors=16`, `weights=distance`

> **Zauważone problemy:** MissForest KOR wybrał `max_depth=15` (górna granica zakresu), a MICE NEURO wybrał `n_estimators=100` (górna granica zakresu). Gdy Optuna "uderza w sufit", oznacza to że optymalny parametr może leżeć jeszcze wyżej — nie mamy pewności że znaleźliśmy prawdziwe optimum. To był impuls do przebiegu 4.

---

### 4.5 Weryfikacja stabilności — seed=7

Zanim zdecydowaliśmy się na rozszerzenie zakresu, sprawdziliśmy najpierw czy wyniki z przebiegu 3 są stabilne — czy zależą od metody, czy od konkretnego wzorca braków w masce.

Uruchomiliśmy ewaluację z **tą samą konfiguracją hiperparametrów** co przebieg 3, ale z **nową, niezależną maską ewaluacyjną** (seed=7). Optuna nie była ponownie uruchamiana.

#### KOR — seed=7

| Metoda | RMSE | MAE | KL mean |
|---|---|---|---|
| **MissForest** ✓ | **0,0787** | **0,0309** | **0,0035** |
| MICE | 0,0868 | 0,0315 | 0,0036 |
| KNN | 0,1027 | 0,0496 | 0,0049 |

#### NEURO — seed=7

| Metoda | RMSE | MAE | KL mean |
|---|---|---|---|
| **MICE** ✓ | **0,0948** | **0,0423** | **0,0120** |
| MissForest | 0,0967 | 0,0525 | 0,0124 |
| KNN | 0,1206 | 0,0723 | 0,0139 |

#### Porównanie seed=42 vs seed=7

| Zbiór | Metoda | RMSE seed=42 | RMSE seed=7 | Różnica |
|---|---|---|---|---|
| KOR | MissForest | 0,0783 | 0,0787 | +0,0004 ✅ |
| KOR | MICE | 0,0865 | 0,0868 | +0,0003 ✅ |
| KOR | KNN | 0,1022 | 0,1027 | +0,0005 ✅ |
| NEURO | MICE | 0,0978 | 0,0948 | −0,0030 ✅ |
| NEURO | MissForest | 0,1017 | 0,0967 | −0,0050 ✅ |
| NEURO | KNN | 0,1291 | 0,1206 | −0,0085 ✅ |

**Ranking metod jest identyczny przy obu seedach.** Różnice RMSE są minimalne (poniżej 0,01) — wyniki odzwierciedlają właściwości metody, nie konkretnego wzorca braków. To ważny argument za wiarygodnością wyników.

---

### 4.6 Przebieg 4 — rozszerzona optymalizacja MICE

Z przebiegu 3 wynikało, że `n_estimators=100` dla MICE na zbiorze NEURO to sufit poprzedniego zakresu. Poszerzyliśmy przestrzeń poszukiwań i zwiększyliśmy budżet prób:

| Parametr | Stary zakres | Nowy zakres |
|---|---|---|
| `n_estimators` | [10, 100] | [10, **200**] |
| `max_iter` | [5, 60] | [5, **80**] |
| Liczba prób | 20 | **40** |

Wyniki po rozszerzonej optymalizacji:

| Zbiór | `max_iter` | `n_estimators` | RMSE (walidacja Optuna) | RMSE (finalna ewaluacja) | Czas |
|---|---|---|---|---|---|
| KOR | **7** | 78 | 0,0857 | 0,0861 | 7h05m |
| NEURO | 28 | **110** | 0,0946 | 0,0984 | 3h39m |

Kluczowe obserwacje:
- Żaden z nowych parametrów nie trafił w sufit → zakresy były wystarczające
- KOR: optymalne `max_iter=7` — drastycznie mniej iteracji niż wcześniej (23→7) przy prawie identycznym RMSE. MICE na dużym zbiorze zbiega bardzo szybko
- NEURO: `n_estimators=110` — potwierdza że sufit 100 był ograniczeniem, ale wpływ na wynik był minimalny (+0,0006 RMSE)
- W obu przypadkach BayesianRidge zostało odrzucone na rzecz ExtraTrees

---

## 5. Weryfikacja wybranej metody i wybór parametrów finalnych

### 5.1 Walidacja stabilności MICE na 10 różnych maskach (`validate_mice.py`)

Żeby upewnić się, że wyniki MICE nie są artefaktem jednej konkretnej maski, powtórzyliśmy ewaluację na 10 niezależnych maskach (seed 0–9), używając parametrów z przebiegu 4.

```python
# Fragment validate_mice.py — pętla po seedach
results = []
for seed in range(10):
    mask = create_mask(df_scaled, frac=MASK_FRAC, seed=seed)
    df_masked = apply_mask(df_scaled, mask)

    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=MAX_DEPTH_FIXED,   # stałe 10
            random_state=42, n_jobs=-1
        ),
        max_iter=params["max_iter"],
        min_value=0.0, max_value=1.0,
        random_state=42,
    )
    imputed = imputer.fit_transform(df_masked)
    rmse = compute_rmse(df_scaled.values, imputed, mask)
    results.append({"seed": seed, "RMSE": rmse, ...})
```

Wyniki:

| Zbiór | RMSE średnia | RMSE std | RMSE min | RMSE max |
|---|---|---|---|---|
| KOR | **0,0865** | **0,0007** | 0,0852 | 0,0876 |
| NEURO | **0,0950** | **0,0025** | 0,0917 | 0,1013 |

KOR: odchylenie standardowe 0,0007 to wyjątkowo mała wartość — MICE jest ekstremalnie stabilne na dużym zbiorze. NEURO ma większą wariancję (0,0025), co jest spodziewane przy tylko 780 rekordach. Wyniki z przebiegu 3 (seed=42) mieszczą się w tym rozkładzie — nie były wartościami odstającymi.

### 5.2 Cross-param — czy jeden zestaw parametrów wystarczy dla obu zbiorów?

Sprawdziliśmy, co się stanie jeśli użyjemy parametrów zoptymalizowanych na KOR do imputacji NEURO i odwrotnie:

| Zbiór | Parametry skąd | RMSE | Różnica vs baseline | Czas |
|---|---|---|---|---|
| KOR | **z KOR** (baseline) | **0,0861** | — | 254 s |
| KOR | z NEURO (cross) | 0,0863 | +0,0002 | 1796 s |
| NEURO | **z NEURO** (baseline) | **0,09836** | — | 242 s |
| NEURO | z KOR (cross) | 0,09837 | +0,00001 | **33 s** |

Wniosek jest zaskakująco jednoznaczny: parametry z KOR (`max_iter=7, n_estimators=78`) działają na NEURO równie dobrze jak parametry zoptymalizowane specjalnie dla NEURO — różnica RMSE wynosi dosłownie 0,00001. I co więcej — są **7 razy szybsze** (33 s vs 242 s), bo mniejsze `max_iter` i `n_estimators` to mniej pracy dla algorytmu.

### 5.3 Wybór finalnych parametrów

| Parametr | Wartość |
|---|---|
| `estimator_type` | ExtraTrees |
| `max_iter` | **7** |
| `n_estimators` | **78** |
| `max_depth` | 10 (stały) |

Ten zestaw parametrów jest używany **dla obu zbiorów** — daje identyczną jakość co parametry zoptymalizowane osobno dla NEURO, a jest znacznie szybszy.

---

## 6. Finalna imputacja pełnego zbioru (`impute_final.py`)

Po wybraniu metody i parametrów przeprowadziliśmy imputację całego zbioru danych — nie tylko complete cases używanych w benchmarku, ale wszystkich 78 197 rekordów łącznie z tymi, które mają prawdziwe braki.

```python
# impute_final.py — kluczowe fragmenty

# 1. Wczytaj pełny zbiór (KOR + NEURO połączone)
df = pd.read_csv("aneurysm_concatted.csv")        # 78 197 wierszy

# 2. Scaler dopasowany tylko na complete cases — spójność z benchmarkiem
df_complete = df[feat_cols].dropna()               # 21 443 wierszy
scaler = MinMaxScaler()
scaler.fit(df_complete.values.astype(float))

# 3. Ręczne skalowanie z zachowaniem NaN
scale_range = scaler.data_max_ - scaler.data_min_
df_scaled_full = (df[feat_cols] - scaler.data_min_) / scale_range
# (NaN pozostają NaN po tej operacji)

# 4. Imputacja z finalnymi parametrami
FINAL_PARAMS = dict(max_iter=7, n_estimators=78, max_depth=10)

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(
        n_estimators=FINAL_PARAMS["n_estimators"],
        max_depth=FINAL_PARAMS["max_depth"],
        random_state=42, n_jobs=-1,
    ),
    max_iter=FINAL_PARAMS["max_iter"],
    min_value=0.0, max_value=1.0,
    random_state=42,
)
imputed_scaled = imputer.fit_transform(df_scaled_full)

# 5. Inwersja normalizacji — powrót do oryginalnych jednostek
imputed_original = scaler.inverse_transform(imputed_scaled)

# 6. Zapis z zachowaniem kolumn meta (patient_id, label itp.)
df_result = pd.concat([df_meta, pd.DataFrame(imputed_original, columns=feat_cols)], axis=1)
df_result.to_csv("results/aneurysm_imputed_final.csv", index=False)
```

Statystyki procesu:

| | Wartość |
|---|---|
| Wejście | 78 197 rekordów, 13,4% brakujących wartości |
| Czas imputacji | 26,1 min |
| NaN po imputacji | **0** |
| Wartości ujemne po imputacji | **0** |
| Wyjście | `results/aneurysm_imputed_final.csv` |

---

## 7. Walidacja statystyczna zaimputowanego zbioru (`validate_imputed.py`)

Po imputacji sprawdziliśmy, czy dane mają sens statystyczny — czy rozkłady zmiennych nie zostały zaburzone.

### 7.1 Zmiany w średnich wartościach zmiennych

Porównaliśmy średnie wartości wybranych zmiennych przed i po imputacji:

| Zmienna | Średnia przed | Średnia po | Różnica | Wyjaśnienie |
|---|---|---|---|---|
| PLT (płytki krwi) | 250,5 | 255,2 | +4,7 | Pacjenci bez PLT tworzą inną subpopulację |
| Wiek pacjenta | 63,1 | 58,5 | −4,6 | Młodsi pacjenci rzadziej mają kompletne wyniki |
| GLU (glukoza) | 126,8 | 123,9 | −2,9 | Glukoza rzadziej mierzona u stabilnych pacjentów |

Te różnice są spodziewane i wynikają z wspomnianego na początku selection bias — rekordy z kompletnymi wynikami to inna subpopulacja niż reszta. Imputacja "uzupełnia" brakujące wartości dla tej drugiej grupy, co zmienia statystyki agregowane całego zbioru.

### 7.2 Test Kołmogorowa-Smirnowa

Test KS sprawdza, czy dwa zbiory próbek mogą pochodzić z tego samego rozkładu. Porównaliśmy complete cases (referencja) z pełnym zbiorem po imputacji.

| Grupa | Kolumny z p > 0,05 | Kolumny z p ≤ 0,05 |
|---|---|---|
| Cały zbiór | 0 z 38 | 38 z 38 |
| KOR | 0 z 38 | 38 z 38 |
| NEURO | 0 z 38 | 38 z 38 |

Wszystkie kolumny mają p ≤ 0,05 — wyniki wyglądają alarmująco, ale jest ważne wyjaśnienie metodologiczne: przy n = 78 000 test KS jest tak czuły, że wykrywa nawet mikroskopijne różnice wynikające z selection bias, a nie z błędu imputacji. Kolumny koagulologiczne (PT/APTT/INR) mają szczególnie duże wartości statystyki KS (0,33–0,45), bo są robione selektywnie — to nie jest błąd imputacji, to naturalna cecha danych. Właściwą miarą jakości imputacji pozostaje RMSE z benchmarku.

### 7.3 Zachowanie struktury korelacji

Sprawdziliśmy, czy imputacja nie zaburzyła zależności między zmiennymi (np. naturalna korelacja między HGB a HCT).

| Grupa | Norma Frobeniusa Δkorelacji | Maks. zmiana korelacji | Ocena |
|---|---|---|---|
| Cały zbiór | 1,21 | 0,17 | dobry |
| KOR | 1,23 | 0,18 | dobry |
| NEURO | 2,56 | 0,28 | akceptowalny |

Wyższa wartość dla NEURO wynika z małej liczebności zbioru referencyjnego (780 wierszy) — przy małej próbie macierz korelacji jest mniej stabilna. Maksymalna zmiana korelacji o 0,28 to dopuszczalny poziom dla zbioru tej wielkości.

---

## 8. Wnioski

### Dlaczego MICE a nie MissForest?

MissForest wygrał na zbiorze KOR (RMSE = 0,0783 vs 0,0865 dla MICE), ale MICE wygrał na NEURO (0,0978 vs 0,1017). Skoro wyniki są zależne od zbioru, który jest "lepszy"?

Zdecydowaliśmy się na MICE z kilku powodów:

1. **Brak wyraźnej słabości** — MICE działa dobrze na obu zbiorach, MissForest jest lepszy tylko na dużym KOR
2. **Stabilność** — odchylenie standardowe RMSE w teście 10-seedowym: KOR σ=0,0007, NEURO σ=0,0025. Wyjątkowo przewidywalny
3. **Jeden zestaw parametrów** — te same parametry działają identycznie na KOR i NEURO (Δ RMSE = 0,00001)
4. **Interpretowalność różnicy** — MissForest zyskuje na dużych zbiorach (lasy losowe lubią dużo danych), MICE (regresja iteracyjna) jest bardziej odporna przy małej próbie. To sensowne zachowanie, nie artefakt

### Dlaczego KNN odrzucono?

KNN jest wyraźnie gorszy od obu metod drzewowych we wszystkich metrykach i obu zbiorach. Na NEURO RMSE wynosi 0,1291 vs 0,0978 dla MICE — to 32% więcej błędu. Uśrednianie sąsiadów w wysokowymiarowej przestrzeni (38 cech) nie wyłapuje złożonych zależności między zmiennymi laboratoryjnymi.

### Podsumowanie wyników benchmarku

| Zbiór | Najlepsza metoda | RMSE | Metoda finalna | RMSE finalna |
|---|---|---|---|---|
| KOR | MissForest | 0,0783 | MICE | 0,0865 |
| NEURO | MICE | 0,0978 | MICE | 0,0978 |

Koszt wyboru MICE zamiast MissForest na KOR to różnica RMSE 0,0082 — przy zakresie zmiennych [0, 1] oznacza to przeciętnie ~0,8 punktu procentowego większy błąd imputacji. To akceptowalna cena za metodę, która działa konsekwentnie dobrze na obu zbiorach i nie wymaga osobnej optymalizacji.

**Zaimputowany zbiór `aneurysm_imputed_final.csv` (78 197 kompletnych rekordów) jest gotowy do pipeline'u modelowania PU-learning.**
