# Raport: Imputacja brakujących danych — pipeline i wyniki
**Projekt:** Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled  
**Data finalizacji:** 2026-06-04  
**Folder:** `imputation-final/`

---

## 1. Dane wejściowe

### 1.1 Zbiory danych

| Zbiór | Rekordów (raw) | Kompletnych | % kompletnych | Kolumn cech |
|---|---|---|---|---|
| KOR (`kor_shortend.csv`) | 71 546 | 20 663 | 28,9% | 38 |
| NEURO (`neuro_shortend.csv`) | 6 651 | 780 | 11,7% | 38 |

Cechy obejmują: morfologię krwi (WBC, RBC, HGB, HCT, PLT i inne), biochemię (GLU, KREA, MOCZNIK, CHL i inne), koagulologię (PT, APTT, INR, WAPTT), wiek i płeć pacjenta.

### 1.2 Przygotowanie danych (`prepare_data.py`)

1. Usunięcie kolumn identyfikacyjnych: `patient_id`, `custom_id`, `examination_date`
2. Selekcja kolumn numerycznych (38 cech)
3. Filtracja do **complete cases** — wiersze bez żadnych braków (podstawa dla scalera i ewaluacji)
4. Skalowanie **MinMaxScaler → [0, 1]** dopasowane na complete cases

> Skalowanie do [0, 1] gwarantuje, że wszystkie kolumny mają identyczny zakres podczas optymalizacji metryk oraz że imputacja z ograniczeniem `min_value=0.0` eliminuje wartości ujemne.

---

## 2. Metodologia ewaluacji

### 2.1 Protokół maskowania

Ewaluacja opiera się na symulacji brakujących danych przez **losowe maskowanie**:

```
df_complete (complete cases, scaled [0,1])
│
├── maska Optuna  seed=43  ← hiperparametryzacja (Optuna nigdy nie widzi maski ewaluacyjnej)
│
└── maska eval   seed=42  ← finalna ewaluacja porównawcza
```

- 10% wartości zamaskowanych globalnie (losowo po całej macierzy)
- Identyczna maska dla wszystkich trzech metod → bezpośrednia porównywalność
- Rozdzielenie masek Optuna i ewaluacji: zapobiega leakage hiperparametrów

### 2.2 Metryki

| Metryka | Opis |
|---|---|
| **RMSE** | Pierwiastek ze średniego błędu kwadratowego na zamaskowanych pozycjach, skala [0,1] |
| **MAE** | Średni błąd bezwzględny na zamaskowanych pozycjach, skala [0,1] |
| **KL mean** | Średnia dywergencja Kullbacka-Leiblera po wszystkich 38 kolumnach (wierność rozkładu) |
| **Negatives** | Liczba wartości < 0 po imputacji (powinno być 0) |

### 2.3 Optymalizacja hiperparametrów (Optuna)

Każda metoda była optymalizowana niezależnie przez **Optunę z samplerem TPE** (`random_state=42`).

Dla zbioru KOR (20 663 wierszy): każdy trial Optuna korzystał z losowego podzboru **5 000 wierszy** w celu przyspieszenia optymalizacji — finalna imputacja ewaluacyjna wykonywana na pełnym zbiorze.

| Metoda | Liczba trials |
|---|---|
| KNN | 30 |
| MICE (v3) | 20 |
| MissForest | 15 |
| MICE (v4, rozszerzona) | 40 |

---

## 3. Historia przebiegów i korekty metodologiczne

### Przebieg 1 — odrzucony: leakage hiperparametrów ❌

Optuna używała tej samej maski co finalna ewaluacja (seed=42 dla obu). Parametry były przefitowane pod konkretny wzorzec braków. Wyniki odrzucone.

Dodatkowy problem: MICE bez ograniczenia `max_depth` → potencjalny overfitting ExtraTrees.

### Przebieg 2 — korekta leakage i zakresów ✅

| Problem | Korekta |
|---|---|
| Leakage maski | Optuna seed=43, ewaluacja seed=42 |
| `max_depth` bez limitu (ExtraTrees) | Ograniczono do 10 |
| Zakresy MissForest zbyt wąskie | `n_estimators` [10, 150] → [10, 300] |

### Przebieg 3 — wyniki główne (seed=42) ✅

Ten sam protokół co v2. Wyniki zebrane w sekcji 4.1.

Przestrzeń hiperparametrów:

| Metoda | Parametr | Zakres |
|---|---|---|
| KNN | `n_neighbors` | [1, 30] |
| KNN | `weights` | uniform / distance |
| MICE | `estimator_type` | BayesianRidge / ExtraTrees |
| MICE | `max_iter` | [5, 60] |
| MICE | `n_estimators` | [10, 100] (ExtraTrees) |
| MICE | `max_depth` | 10 (stały) |
| MissForest | `n_estimators` | [10, 300] |
| MissForest | `max_depth` | [3, 15] |
| MissForest | `max_iter` | [3, 15] |

### Przebieg 4 — rozszerzona Optuna MICE ✅ (2026-06-04)

NEURO w przebiegu 3 osiągnął sufit `n_estimators=100`. Rozszerzono zakres i zwiększono budżet trials:

| Parametr | Zakres v3 | Zakres v4 |
|---|---|---|
| `n_estimators` | [10, 100] | [10, 200] |
| `max_iter` | [5, 60] | [5, 80] |
| n_trials | 20 | 40 |

**Wyniki Optuna v4:**

| Zbiór | `estimator_type` | `max_iter` | `n_estimators` | val RMSE | final RMSE | Czas |
|---|---|---|---|---|---|---|
| KOR | ExtraTrees | 7 | 78 | 0.0857 | 0.0861 | 7h05m |
| NEURO | ExtraTrees | 28 | 110 | 0.0946 | 0.0984 | 3h39m |

- BayesianRidge odrzucone przez Optunę we wszystkich przebiegach — ExtraTrees wygrywa na obu zbiorach
- KOR: `max_iter=7` — MICE zbiega bardzo szybko na dużym zbiorze
- Żaden parametr nie trafił w sufit → zakresy v4 wystarczające

---

## 4. Wyniki benchmarku

### 4.1 Seed=42 — główny przebieg

#### KOR (n = 20 663)

| Metoda | RMSE | MAE | KL mean | Ujemne | Czas |
|---|---|---|---|---|---|
| **MissForest** ✓ | **0.0783** | **0.0308** | **0.0036** | 0 | 107 min |
| MICE | 0.0865 | 0.0315 | 0.0036 | 0 | 70 min |
| KNN | 0.1022 | 0.0494 | 0.0049 | 0 | 23 min |

Optymalne hiperparametry:
- MissForest: `n_estimators=118`, `max_depth=15` (sufit), `max_iter=12`
- MICE: ExtraTrees, `max_iter=23`, `n_estimators=54`
- KNN: `n_neighbors=28`, `weights=distance`

#### NEURO (n = 780)

| Metoda | RMSE | MAE | KL mean | Ujemne | Czas |
|---|---|---|---|---|---|
| **MICE** ✓ | **0.0978** | **0.0432** | **0.0116** | 0 | 30 min |
| MissForest | 0.1017 | 0.0540 | 0.0129 | 0 | 17 min |
| KNN | 0.1291 | 0.0752 | 0.0156 | 0 | <1 min |

Optymalne hiperparametry:
- MICE: ExtraTrees, `max_iter=18`, `n_estimators=100` (sufit v3)
- MissForest: `n_estimators=142`, `max_depth=13`, `max_iter=5`
- KNN: `n_neighbors=16`, `weights=distance`

---

### 4.2 Seed=7 — weryfikacja stabilności

Te same hiperparametry co seed=42, niezależna maska ewaluacyjna. Optuna nie była ponownie uruchamiana.

#### KOR

| Metoda | RMSE | MAE | KL mean |
|---|---|---|---|
| **MissForest** ✓ | **0.0787** | **0.0309** | **0.0035** |
| MICE | 0.0868 | 0.0315 | 0.0036 |
| KNN | 0.1027 | 0.0496 | 0.0049 |

#### NEURO

| Metoda | RMSE | MAE | KL mean |
|---|---|---|---|
| **MICE** ✓ | **0.0948** | **0.0423** | **0.0120** |
| MissForest | 0.0967 | 0.0525 | 0.0124 |
| KNN | 0.1206 | 0.0723 | 0.0139 |

---

### 4.3 Stabilność wyników (seed=42 vs seed=7)

| Zbiór | Metoda | RMSE seed=42 | RMSE seed=7 | Δ RMSE |
|---|---|---|---|---|
| KOR | MissForest | 0.0783 | 0.0787 | +0.0004 ✅ |
| KOR | MICE | 0.0865 | 0.0868 | +0.0003 ✅ |
| KOR | KNN | 0.1022 | 0.1027 | +0.0005 ✅ |
| NEURO | MICE | 0.0978 | 0.0948 | −0.0030 ✅ |
| NEURO | MissForest | 0.1017 | 0.0967 | −0.0050 ✅ |
| NEURO | KNN | 0.1291 | 0.1206 | −0.0085 ✅ |

**Ranking metod identyczny w obu seedach.** Wyniki stabilne.

---

### 4.4 Porównanie MICE v3 vs v4

| Zbiór | Metryka | v3 (20 trials, n_est ≤ 100) | v4 (40 trials, n_est ≤ 200) | Δ |
|---|---|---|---|---|
| KOR | RMSE | 0.0865 | **0.0861** | −0.0004 |
| KOR | `max_iter` | 23 | **7** | −16 |
| KOR | `n_estimators` | 54 | 78 | +24 |
| NEURO | RMSE | 0.0978 | 0.0984 | +0.0006 |
| NEURO | `max_iter` | 18 | 28 | +10 |
| NEURO | `n_estimators` | 100 (sufit) | **110** | +10 |

Różnice RMSE mieszczą się w szumie pomiarowym (±0.001). Parametry v4 są bardziej wiarygodne metodologicznie (brak sufitów, szerszy zakres).

---

### 4.5 Walidacja MICE — multi-seed i cross-param (`validate_mice.py`)

#### Eksperyment A: stabilność na 10 seedach (params v4)

| Zbiór | RMSE mean | RMSE std | RMSE min | RMSE max | MAE mean | KL mean |
|---|---|---|---|---|---|---|
| KOR | **0.0865** | **0.0007** | 0.0852 | 0.0876 | 0.0315 | 0.00374 |
| NEURO | **0.0950** | **0.0025** | 0.0917 | 0.1013 | 0.0422 | 0.01331 |

- KOR: σ = 0.0007 — wyjątkowo stabilny (duży zbiór, MICE szybko zbiega)
- NEURO: σ = 0.0025 — spodziewana wyższa wariancja przy małej próbie (780 wierszy)
- Wynik seed=42 mieści się w rozkładzie → nie był wartością odstającą

#### Eksperyment B: cross-param (czy jeden zestaw params wystarczy?)

| Zbiór | Params from | RMSE | Δ vs baseline | Czas |
|---|---|---|---|---|
| KOR | **KOR** (baseline) | **0.0861** | — | 254 s |
| KOR | NEURO (cross) | 0.0863 | +0.0002 | 1796 s |
| NEURO | **NEURO** (baseline) | **0.09836** | — | 242 s |
| NEURO | KOR (cross) | 0.09837 | +0.00001 | **33 s** |

**Wniosek:** params KOR (`max_iter=7, n_est=78`) działają identycznie na NEURO (Δ = 0.00001) i są **7× szybsze** (33 s vs 242 s).

#### Wybór finalnych parametrów MICE

| | KOR | NEURO |
|---|---|---|
| `max_iter` | **7** | **7** |
| `n_estimators` | **78** | **78** |
| RMSE | 0.0861 | 0.0984 |
| Czas NEURO | — | 33 s |

Jeden uniwersalny zestaw parametrów — identyczna jakość, znacząco krótszy czas dla NEURO.

---

## 5. Finalna imputacja pełnego zbioru

### 5.1 Imputacja (`impute_final.py`)

Imputacja połączonego zbioru KOR + NEURO metodą MICE z parametrami z sekcji 4.5.

| Parametr | Wartość |
|---|---|
| Wejście | `aneurysm_concatted.csv` |
| Łączna liczba rekordów | 78 197 |
| Brakujące wartości (wejście) | 399 546 / 2 971 486 (13,4%) |
| Kompletne wiersze (referencja scalera) | 21 443 |
| Metoda | MICE (ExtraTrees, `max_iter=7`, `n_estimators=78`) |
| Czas imputacji | 26,1 min |
| NaN po imputacji | **0** |
| Wartości < 0 | **0** |
| Wyjście | `results/aneurysm_imputed_final.csv` |

### 5.2 Walidacja statystyczna (`validate_imputed.py`)

#### Statystyki opisowe — zmiany średniej po imputacji

| Kolumna | Mean przed | Mean po | Δ | Wyjaśnienie |
|---|---|---|---|---|
| `PLT` | 250,5 | 255,2 | +4,7 | Pacjenci bez PLT to inna subpopulacja |
| `patient_age` | 63,1 | 58,5 | −4,6 | Młodsi pacjenci częściej mają niepełne wyniki |
| `GLU` | 126,8 | 123,9 | −2,9 | Glukoza rzadziej mierzona u stabilnych |

Zmiany wynikają z **selection bias** — complete cases to selektywna próba, nie reprezentatywna dla całej populacji. Jest to spodziewane i metodologicznie poprawne.

#### Test Kołmogorowa-Smirnowa (complete cases vs zbiór zaimputowany)

| Grupa | p > 0.05 | p ≤ 0.05 |
|---|---|---|
| ALL | 0/38 | 38/38 |
| KOR | 0/38 | 38/38 |
| NEURO | 0/38 | 38/38 |

**Uwaga:** KS test porównuje complete cases (~21k, selektywna próba) z pełnym zbiorem (~78k). Przy tej liczebności KS jest ekstremalnie czuły — wykrywa nawet mikroskopijne różnice wynikające z selection bias. Znaczące p-value dla koagulologii (PT/APTT/INR/WAPTT, KS ≈ 0.33–0.45) wynika z faktu, że te badania są zlecane selektywnie. Właściwą miarą jakości imputacji pozostaje **RMSE = 0.086 / 0.095** z benchmarku maskowania.

#### Zachowanie struktury korelacji

| Grupa | Frobenius norm Δcorr | Max |Δcorr| | Ocena |
|---|---|---|---|
| ALL | 1.21 | 0.17 | dobry |
| KOR | 1.23 | 0.18 | dobry |
| NEURO | 2.56 | 0.28 | akceptowalny |

Wyższa norma dla NEURO wynika z małej liczebności zbioru referencyjnego (780 wierszy).

---

## 6. Zidentyfikowane problemy metodologiczne

| Problem | Opis | Status |
|---|---|---|
| Leakage hiperparametrów | Optuna optymalizowała na tej samej masce co ewaluacja | ✅ Naprawione: Optuna seed=43, ewaluacja seed=42 |
| `n_estimators=150` — sufit MissForest | Optuna trafiła w górną granicę zakresu | ✅ Rozszerzono zakres do 300 |
| `max_depth` bez limitu ExtraTrees | Brak ograniczenia głębokości w MICE | ✅ Ograniczono do 10 |
| `n_estimators=100` — sufit MICE v3 NEURO | Optuna trafiła w sufit na NEURO | ✅ Rozszerzono do 200, wynik: n_est=110 |
| `max_depth=15` — sufit MissForest KOR | Optuna wybrała górną granicę [3,15] | ⚠️ Otwarte (wpływ nieznany) |

---

## 7. Wnioski

**KNN odrzucone** — wyraźnie gorszy od metod drzewowych we wszystkich metrykach i obu zbiorach. RMSE wyższe o ~30% względem najlepszej metody.

**Zależność od rozmiaru zbioru:**
- KOR (n > 20k): MissForest wygrywa (RMSE = 0.0783) — lasy losowe zyskują na dużej próbie
- NEURO (n = 780): MICE wygrywa (RMSE = 0.0978) — iteracyjna regresja lepiej generalizuje przy małej próbie

**MICE jako wybór finalny:**
- Dobra jakość na obu zbiorach (jedyna metoda bez wyraźnej słabości)
- Wyjątkowo wysoka stabilność: σ = 0.0007 (KOR), σ = 0.0025 (NEURO) w teście 10-seedowym
- Jeden uniwersalny zestaw parametrów działa identycznie na obu zbiorach

**Brak wartości ujemnych** wszędzie — ograniczenie `min_value=0.0` w IterativeImputer działa poprawnie.

**KL divergence:** niskie przy KOR (0.003–0.005), wyższe przy NEURO (0.010–0.016) — wynika z małej liczebności NEURO, nie z błędu metody.

**Wynik finalny:** zbiór `aneurysm_imputed_final.csv` (78 197 rekordów, 0 brakujących wartości) gotowy do pipeline'u PU-learning.
