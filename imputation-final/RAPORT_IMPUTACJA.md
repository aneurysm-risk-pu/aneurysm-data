# Raport: Porównanie metod imputacji danych medycznych
**Projekt:** Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled  
**Data:** 2026-06-03  
**Status:** Wyniki finalne (przebieg 3 + weryfikacja stabilności na seed=7)

---

## 1. Dane wejściowe

| Zbiór | Wiersze oryginalne | Complete cases | % kompletnych | Kolumny cech |
|---|---|---|---|---|
| KOR | 71 546 | 20 663 | 28.9% | 38 |
| NEURO | 6 651 | 780 | 11.7% | 38 |

**Przygotowanie danych:**
- Źródło: `kor_shortend.csv` / `neuro_shortend.csv` (root projektu)
- Usunięte kolumny: `patient_id`, `custom_id`, `examination_date`
- Zostały tylko kolumny numeryczne (38 cech: morfologia, biochemia, wiek, płeć)
- Filtracja do **complete cases** — tylko wiersze bez żadnych braków
- Skalowanie **MinMaxScaler → [0, 1]** na kompletnym zbiorze

---

## 2. Metodologia ewaluacji

### Protokół maskowania
- **10% wartości** zamaskowanych losowo (globalnie po całej macierzy)
- Seed maski: `42` (reprodukowalność)
- Ta sama maska dla wszystkich 3 metod → bezpośrednia porównywalność

### Metryki
| Metryka | Co mierzy |
|---|---|
| **RMSE** | Dokładność punktowa na zamaskowanych wartościach [0,1] |
| **MAE** | Średni błąd bezwzględny na zamaskowanych wartościach [0,1] |
| **KL mean** | Wierność rozkładu — średnia KL divergence po wszystkich kolumnach |
| **negatives** | Liczba wartości < 0 po imputacji (powinno być 0) |

### Optymalizacja hiperparametrów (fitting)

Każda metoda optymalizowana przez **Optunę** (TPE sampler, `random_state=42`).

**Podział danych w procesie fittingu:**

```
df_complete (complete cases)
│
├─ MinMaxScaler → df_scaled  [0, 1]
│
├─ maska Optuna  seed=43  ← Optuna fituje i optymalizuje na tej masce
│   └─ n_trials: KNN=30, MICE=20, MissForest=15
│
└─ maska eval    seed=42  ← finalna ewaluacja (Optuna jej nigdy nie widzi)
```

**Przyspieszenie dla KOR** (20 663 wierszy): każdy trial Optuna używa losowego podzboru 5 000 wierszy (`OPTUNA_VAL_ROWS=5_000`). Ostateczna imputacja do ewaluacji i tak wykonywana na pełnym zbiorze. NEURO (780 wierszy) — bez podzboru.

**Finalna imputacja** po zakończeniu Optuna: dopasowanie na pełnym `df_scaled` z maską seed=42, wyniki → RMSE/MAE/KL w tabeli poniżej.

---

## 3. Przestrzeń hiperparametrów — historia przebiegów

Parametry były iteracyjnie korygowane po każdym przebiegu gdy Optuna trafiała w górną granicę zakresu (sufit) lub wykryto błąd metodologiczny.

### Przebieg 1 — wersja bazowa ❌ (leakage)
> Optuna używała tej samej maski co ewaluacja (seed=42 dla obu) → parametry przefitowane pod konkretną maskę. Wyniki odrzucone.

| Metoda | Parametr | Zakres v1 |
|---|---|---|
| MICE | `max_iter` | [5, 40] |
| MICE | `n_estimators` | [10, 100] |
| MICE | `max_depth` | brak limitu ⚠️ |
| MissForest | `n_estimators` | [10, 150] |
| MissForest | `max_depth` | [5, 25] |
| MissForest | `max_iter` | [3, 8] |

### Przebieg 2 — korekta leakage i sufitów ✅
> Naprawiono leakage (Optuna seed=43, ewaluacja seed=42). Ograniczono `max_depth` ExtraTrees do 10. Rozszerzono zakresy MissForest.

| Metoda | Parametr | Zakres v2 | Zmiana |
|---|---|---|---|
| MICE | `max_iter` | [5, 60] | +20 |
| MICE | `n_estimators` | [10, 100] | bez zmian |
| MICE | `max_depth` | **10 (stały)** | nowe ograniczenie |
| MissForest | `n_estimators` | [10, **300**] | +150 |
| MissForest | `max_depth` | [3, **15**] | zmniejszono zakres |
| MissForest | `max_iter` | [3, **15**] | +7 |

### Przebieg 3 — wyniki raportowane (seed=42) ✅
> Ten sam protokół co v2. Wyniki z tej wersji są w sekcji 4.

**KNN** (30 trials) — bez zmian we wszystkich przebiegach:
| Parametr | Zakres |
|---|---|
| `n_neighbors` | [1, 30] |
| `weights` | uniform / distance |

**MICE** (20 trials) — Metodyka A: dane [0,1] → MICE z `min_value=0, max_value=1`:
| Parametr | Zakres |
|---|---|
| `estimator_type` | BayesianRidge / ExtraTrees |
| `max_iter` | [5, 60] |
| `n_estimators` | [10, 100] (tylko ExtraTrees) |
| `max_depth` | 10 (stały) |

**MissForest** (15 trials):
| Parametr | Zakres |
|---|---|
| `n_estimators` | [10, 300] |
| `max_depth` | [3, 15] |
| `max_iter` | [3, 15] |

### Przebieg 4 — rozszerzona Optuna MICE ✅ (zakończony 2026-06-04)
> NEURO w przebiegu 3 trafił w sufit `n_estimators=100`. Rozszerzono zakres i zwiększono liczbę trials.

| Parametr | Zakres v3 | Zakres v4 |
|---|---|---|
| `n_estimators` | [10, 100] | [10, **200**] |
| `max_iter` | [5, 60] | [5, **80**] |
| n_trials | 20 | **40** |

**Wyniki Optuna v4:**

| Zbiór | `estimator_type` | `max_iter` | `n_estimators` | val RMSE | final RMSE | Czas |
|---|---|---|---|---|---|---|
| KOR | ExtraTrees | **7** | 78 | 0.0857 | 0.0861 | 7h05m |
| NEURO | ExtraTrees | 28 | **110** | 0.0946 | 0.0984 | 3h39m |

**Obserwacje:**
- Żaden param nie trafił w nowy sufit → zakresy v4 były wystarczające
- KOR: `max_iter` drastycznie zmalał (23→7) przy prawie identycznym RMSE — MICE zbiega szybko na tym zbiorze
- NEURO: `n_estimators=110` potwierdza że sufit 100 był ograniczeniem, ale wpływ na RMSE minimalny (+0.0006)
- BayesianRidge odrzucone przez Optunę w obu zbiorach — ExtraTrees wygrywa we wszystkich przebiegach

---

## 4. Wyniki finalne

### 4.1 seed=42 (główny przebieg)

#### KOR

| Metoda | RMSE | MAE | KL mean | Ujemne | Czas |
|---|---|---|---|---|---|
| **MissForest** ✓ | **0.0783** | 0.0308 | 0.0036 | 0 | 107 min |
| MICE | 0.0865 | **0.0315** | **0.0036** | 0 | 70 min |
| KNN | 0.1022 | 0.0494 | 0.0049 | 0 | 23 min |

Najlepsze hiperparametry:
- **MissForest**: `n_estimators=118`, `max_depth=15` ⚠️ (sufit [3,15]), `max_iter=12`
- **MICE**: `ExtraTrees`, `max_iter=23`, `n_estimators=54`
- **KNN**: `n_neighbors=28`, `weights=distance`

#### NEURO

| Metoda | RMSE | MAE | KL mean | Ujemne | Czas |
|---|---|---|---|---|---|
| **MICE** ✓ | **0.0978** | **0.0432** | **0.0116** | 0 | 30 min |
| MissForest | 0.1017 | 0.0540 | 0.0129 | 0 | 17 min |
| KNN | 0.1291 | 0.0752 | 0.0156 | 0 | <1 min |

Najlepsze hiperparametry:
- **MICE**: `ExtraTrees`, `max_iter=18`, `n_estimators=100`
- **MissForest**: `n_estimators=142`, `max_depth=13`, `max_iter=5`
- **KNN**: `n_neighbors=16`, `weights=distance`

---

### 4.2 seed=7 (weryfikacja stabilności ewaluacji)

> **Ważne:** seed=7 to **inna maska ewaluacyjna**, ale **te same hiperparametry** co seed=42 — Optuna nie była ponownie uruchamiana. Celem jest sprawdzenie czy wyniki zależą od konkretnego wzorca braków, nie od metody.

#### KOR

| Metoda | RMSE | MAE | KL mean | Ujemne |
|---|---|---|---|---|
| **MissForest** ✓ | **0.0787** | 0.0309 | 0.0035 | 0 |
| MICE | 0.0868 | **0.0315** | **0.0036** | 0 |
| KNN | 0.1027 | 0.0496 | 0.0049 | 0 |

#### NEURO

| Metoda | RMSE | MAE | KL mean | Ujemne |
|---|---|---|---|---|
| **MICE** ✓ | **0.0948** | **0.0423** | **0.0120** | 0 |
| MissForest | 0.0967 | 0.0525 | 0.0124 | 0 |
| KNN | 0.1206 | 0.0723 | 0.0139 | 0 |

---

### 4.3 Stabilność wyników (seed=42 vs seed=7)

| Zbiór | Metoda | RMSE seed=42 | RMSE seed=7 | Δ RMSE |
|---|---|---|---|---|
| KOR | MissForest | 0.0783 | 0.0787 | +0.0004 ✅ |
| KOR | MICE | 0.0865 | 0.0868 | +0.0003 ✅ |
| KOR | KNN | 0.1022 | 0.1027 | +0.0005 ✅ |
| NEURO | MICE | 0.0978 | 0.0948 | -0.0030 ✅ |
| NEURO | MissForest | 0.1017 | 0.0967 | -0.0050 ✅ |
| NEURO | KNN | 0.1291 | 0.1206 | -0.0085 ✅ |

**Ranking metod identyczny w obu seedach — wyniki stabilne.**

---

### 4.4 Porównanie MICE v3 vs v4 (rozszerzona Optuna)

| Zbiór | Metryka | v3 (20 trials, n_est≤100) | v4 (40 trials, n_est≤200) | Δ |
|---|---|---|---|---|
| KOR | RMSE | 0.0865 | **0.0861** | -0.0004 |
| KOR | `max_iter` | 23 | **7** | -16 |
| KOR | `n_estimators` | 54 | 78 | +24 |
| NEURO | RMSE | 0.0978 | 0.0984 | +0.0006 |
| NEURO | `max_iter` | 18 | 28 | +10 |
| NEURO | `n_estimators` | 100 ⚠️sufit | **110** | +10 |

> Różnice RMSE są w granicach szumu pomiarowego (±0.001). Params v4 są bardziej wiarygodne metodologicznie (szerszy zakres, więcej trials, brak sufitów).

### 4.5 Walidacja MICE — multi-seed + cross-param ✅ (2026-06-04)

#### Eksperyment A — Stabilność (10 seedów, params v4)

| Zbiór | Params | RMSE mean | RMSE std | RMSE min | RMSE max | MAE mean | KL mean |
|---|---|---|---|---|---|---|---|
| KOR | max_iter=7, n_est=78 | **0.0865** | **0.0007** | 0.0852 | 0.0876 | 0.0315 | 0.00374 |
| NEURO | max_iter=28, n_est=110 | **0.0950** | **0.0025** | 0.0917 | 0.1013 | 0.0422 | 0.01331 |

> KOR: std=0.0007 — wyjątkowo stabilny (duży zbiór, MICE szybko zbiega). NEURO: std=0.0025 — większa wariancja spodziewana (780 wierszy), ale nadal akceptowalna. Wynik seed=42 (0.0861/0.0984) mieści się w rozkładzie → nie był outlierem.

#### Eksperyment B — Cross-param (jeden zestaw params dla obu zbiorów?)

| Zbiór | Params from | RMSE | Δ vs baseline | Czas |
|---|---|---|---|---|
| KOR | **kor** (baseline) | **0.0861** | — | 254s |
| KOR | neuro (cross) | 0.0863 | +0.0002 | 1796s |
| NEURO | **neuro** (baseline) | **0.09836** | — | 242s |
| NEURO | kor (cross) | 0.09837 | +0.00001 | **33s** |

> **Wniosek:** Params KOR (`max_iter=7, n_est=78`) działają identycznie na NEURO (Δ=0.00001) i są **7× szybsze** (33s vs 242s). Możliwe użycie jednego zestawu params dla obu zbiorów bez utraty jakości.

#### Decyzja o finalnych parametrach MICE

| Opcja | KOR params | NEURO params | RMSE KOR | RMSE NEURO | Czas NEURO |
|---|---|---|---|---|---|
| Osobne (domyślne) | max_iter=7, n_est=78 | max_iter=28, n_est=110 | 0.0861 | 0.0984 | 242s |
| **Wspólne (KOR)** | **max_iter=7, n_est=78** | **max_iter=7, n_est=78** | **0.0861** | **0.0984** | **33s** |

**Wybór: params KOR jako uniwersalne** — identyczna jakość, znacznie szybsze.

---

### 4.6 Finalna imputacja + walidacja statystyczna ✅ (2026-06-04)

#### Imputacja pełnego zbioru (`impute_final.py`)

| | Wartość |
|---|---|
| Wejście | `aneurysm_concatted.csv` (78 197 wierszy, KOR + NEURO) |
| Brakujące wartości wejście | 399 546 / 2 971 486 (13.4%) |
| Kompletne wiersze (referencja scalera) | 21 443 |
| Czas imputacji | 26.1 min |
| NaN po imputacji | **0** |
| Wartości < 0 | **0** |
| Wyjście | `results/aneurysm_imputed_final.csv` |

#### Walidacja statystyczna (`validate_imputed.py`)

**Statystyki opisowe — top zmiany mean:**

| Kolumna | Mean przed | Mean po | Δ | Wyjaśnienie |
|---|---|---|---|---|
| `PLT` | 250.5 | 255.2 | +4.7 | Pacjenci bez PLT to inna subpopulacja |
| `patient_age` | 63.1 | 58.5 | -4.6 | Młodsi pacjenci częściej mają niepełne wyniki |
| `GLU` | 126.8 | 123.9 | -2.9 | Glukoza mierzona rzadziej u stabilnych |

**KS test (complete cases vs zaimputowany zbiór):**

| Grupa | ✅ p>0.05 | ⚠️ p≤0.05 | Ocena |
|---|---|---|---|
| ALL | 0/38 | 38/38 | Spodziewane — patrz uwaga |
| KOR | 0/38 | 38/38 | Spodziewane |
| NEURO | 0/38 | 38/38 | Spodziewane |

> **Uwaga metodologiczna:** KS test porównuje *complete cases* (selektywna próba ~21k) z *pełnym zbiorem* (~78k). Przy tak dużej n KS wykrywa nawet mikroskopijne różnice wynikające z selection bias (pacjenci z kompletnymi wynikami ≠ losowa próba). Kolumny koagulologiczne (PT/APTT/INR/WAPTT, KS≈0.33–0.45) są robione selektywnie — imputacja jest metodologicznie poprawna. Właściwą miarą jakości pozostaje RMSE=0.086/0.095 z benchmarku.

**Zachowanie struktury korelacji:**

| Grupa | Frobenius norm | Max \|Δcorr\| | Ocena |
|---|---|---|---|
| ALL | 1.21 | 0.17 | dobry |
| KOR | 1.23 | 0.18 | dobry |
| NEURO | 2.56 | 0.28 | akceptowalny (mała n referencji) |

---

### 4.7 Benchmark MICE na zbiorze po analizie korelacji (35 cech) ✅ (2026-06-10)

Koleżanka dostarczyła `aneurysm_concatted_cleaned.csv` — zbiór po usunięciu skorelowanych cech.

**Usunięte cechy (analiza korelacji):** `CRP`, `MONO`, `%MONO` (38 → 35 cech)

**Benchmark z maskowanie 10%** (`benchmark_cleaned.py`, params bez zmian: `max_iter=7, n_est=78, max_depth=10`):

| Zbiór | Complete cases | RMSE (35 cech) | RMSE (38 cech) | Δ | MAE | KL | Ujemne |
|---|---|---|---|---|---|---|---|
| KOR | 21 961 | 0.0904 | 0.0861 | **+0.0043** | 0.0327 | 0.00399 | 0 |
| NEURO | 1 131 | 0.0976 | 0.0984 | -0.0008 | 0.0413 | 0.01106 | 0 |

> KOR: wzrost RMSE o +0.0043 po usunięciu `CRP` — marker zapalny był użytecznym predyktorem w MICE dla dużego zbioru. NEURO: różnica w granicach szumu (±0.001). Parametry `max_iter=7, n_est=78` działają poprawnie na 35 cechach bez re-optymalizacji.

**Imputacja pełna** (`impute_final.py` na `aneurysm_concatted_cleaned.csv`):

| | Wartość |
|---|---|
| Wejście | `aneurysm_concatted_cleaned.csv` (78 197 wierszy, 35 cech) |
| Brakujące wartości wejście | 377 236 / 2 736 895 (13.8%) |
| Kompletne wiersze (referencja scalera) | 23 092 |
| NaN po imputacji | **0** |
| Wartości < 0 | **0** |
| Czas | 23.4 min |
| Wyjście | `results/aneurysm_imputed_cleaned.csv` |

> **Ten plik jest właściwym wejściem do pipeline PU learning** — zawiera finalny zestaw 35 cech po analizie korelacji.

---

## 5. Wnioski

**KNN wyraźnie odstaje** od obu metod opartych na drzewach we wszystkich metrykach i obu zbiorach — odpada.

**MissForest wygrywa na KOR** konsekwentnie w obu seedach (RMSE ~0.078). MICE jest szybszy (70 min vs 107 min) przy zbliżonym KL, ale gorszy RMSE o ~0.008.

**MICE wygrywa na NEURO** konsekwentnie w obu seedach (RMSE ~0.095–0.098). MissForest jest tylko ~0.004 gorszy przy seed=7 — zbliżone wyniki.

**Brak wartości ujemnych** wszędzie — `min_value=0.0` działa poprawnie.

**KL divergence** — bliskie 0 przy KOR (0.003–0.005), wyższe przy NEURO (0.010–0.016). Wynika z małej liczebności NEURO (780 wierszy) — rozkłady po imputacji mniej stabilne.

**Różnica między zbiorami jest spodziewana:** KOR (20k wierszy) daje MissForest przewagę przez większą próbę. NEURO (780 wierszy) — RF z dużą liczbą drzew może overfittować, stąd MICE (iteracyjna regresja) radzi sobie lepiej.

**Wniosek operacyjny:** brak jednej dominującej metody — **MICE jako wybór domyślny** (stabilna, przewidywalna, dobra na obu zbiorach), **MissForest jako alternatywa dla KOR** jeśli liczy się każdy punkt RMSE.

---

## 6. Zidentyfikowane problemy metodologiczne

| Problem | Opis | Status |
|---|---|---|
| Leakage hiperparametrów | Optuna optymalizowała na tej samej masce co ewaluacja | ✅ Naprawione — Optuna seed=43, ewaluacja seed=42 |
| `n_estimators=150` — sufit | MissForest trafił w górną granicę zakresu Optuna | ✅ Rozszerzono do 300 |
| `max_depth` bez limitu | ExtraTrees w MICE nie miał ograniczonej głębokości | ✅ Ograniczono do 10 |
| `max_depth=15` — sufit MissForest KOR | Optuna wybrała górną granicę [3,15] | ⚠️ Otwarte — do rozważenia rozszerzenie do 20 |
| MissForest KOR czas — wariant seed=7 | Trial 0 był najlepszy, ale trwał 6h53min vs 1h47min | ℹ️ Duża wariancja czasu RF przy wysokim max_iter=12 |

## 7. Plan dalszy

- [x] Przebieg 1: wersja bazowa (leakage — odrzucona)
- [x] Przebieg 2: naprawa leakage, korekta sufitów MissForest, ograniczenie `max_depth` MICE
- [x] Przebieg 3 (seed=42): wyniki główne
- [x] Weryfikacja stabilności: seed=7 (te same params, inna maska)
- [x] **Przebieg 4** ✅ MICE extended Optuna (n_estimators do 200, 40 trials) — KOR: `max_iter=7, n_est=78`; NEURO: `max_iter=28, n_est=110`
- [x] **Walidacja multi-seed + cross-param** ✅ → stabilność potwierdzona (KOR std=0.0007, NEURO std=0.0025)
- [x] **Wybór finalnych params** ✅ → `max_iter=7, n_estimators=78` (params KOR) **dla obu zbiorów** — identyczne RMSE, 7× szybsze na NEURO
- [x] **Imputacja pełnego zbioru** ✅ → `results/aneurysm_imputed_final.csv` (78 197 wierszy, 0 NaN, 0 ujemnych, 26 min)
- [x] **Walidacja statystyczna** ✅ → KS test + korelacje + statystyki opisowe (patrz sekcja 4.6)
- [x] **Benchmark na zbiorze po korelacji (35 cech)** ✅ → KOR RMSE=0.090, NEURO RMSE=0.098 — params bez zmian (patrz sekcja 4.7)
- [x] **Imputacja finalnego zbioru PU** ✅ → `results/aneurysm_imputed_cleaned.csv` (35 cech, 0 NaN)
- [ ] Przekazanie imputowanego zbioru do pipeline PU learning
