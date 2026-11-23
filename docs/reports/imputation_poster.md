# Imputacja brakujących danych w zbiorach medycznych
## Projekt: Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled

---

## Problem

Dane kliniczne z badań laboratoryjnych zawierają strukturalne braki — nie z błędu pomiaru, lecz dlatego że część badań jest zlecana selektywnie. W zbiorze KOR (kardiologiczno-ogólnym) 71% rekordów miało co najmniej jedną brakującą wartość, w zbiorze NEURO (neurologicznym) aż 88%.

| Zbiór | Rekordów | Kompletnych | Braki |
|---|---|---|---|
| KOR | 71 546 | 20 663 (28,9%) | 71,1% |
| NEURO | 6 651 | 780 (11,7%) | 88,3% |
| **Łącznie** | **78 197** | **21 443** | **13,4% wartości** |

Bez imputacji modele PU-learning musiałyby operować na ułamku danych lub tracić informację z niekompletnych rekordów.

---

## Metody

Porównano trzy metody imputacji, każda zoptymalizowana osobno przez **Optunę (TPE)**:

| Metoda | Opis |
|---|---|
| **KNN** | K najbliższych sąsiadów w przestrzeni cech |
| **MICE** | Iteracyjna imputacja przez regresję (IterativeImputer + ExtraTrees) |
| **MissForest** | Iteracyjne lasy losowe (MissForest z Random Forest) |

Wszystkie metody pracowały na danych znormalizowanych do zakresu [0, 1] (MinMaxScaler), co gwarantuje brak wartości ujemnych po imputacji.

---

## Metodologia ewaluacji

- **Maskowanie:** 10% wartości losowo zakrytych (identyczna maska dla wszystkich metod)
- **Anty-leakage:** osobna maska dla optymalizacji Optuna (seed+1) i finalnej ewaluacji (seed=42)
- **Metryki:** RMSE i MAE na zamaskowanych pozycjach + KL divergence rozkładów kolumn
- **Weryfikacja stabilności:** powtórzenie ewaluacji z niezależną maską (seed=7), 10-krotna walidacja multi-seed dla wybranej metody

---

## Wyniki benchmarku

### KOR (n = 20 663)

| Metoda | RMSE ↓ | MAE ↓ | KL mean ↓ |
|---|---|---|---|
| **MissForest** ✓ | **0.0783** | **0.0308** | **0.0036** |
| MICE | 0.0865 | 0.0315 | 0.0036 |
| KNN | 0.1022 | 0.0494 | 0.0049 |

### NEURO (n = 780)

| Metoda | RMSE ↓ | MAE ↓ | KL mean ↓ |
|---|---|---|---|
| **MICE** ✓ | **0.0978** | **0.0432** | **0.0116** |
| MissForest | 0.1017 | 0.0540 | 0.0129 |
| KNN | 0.1291 | 0.0752 | 0.0156 |

> Wyniki stabilne — ranking metod identyczny przy niezależnej masce ewaluacyjnej (seed=7, Δ RMSE < 0.005).

---

## Wnioski

- **KNN** wyraźnie gorszy od metod drzewowych we wszystkich metrykach i obu zbiorach
- **MissForest** dominuje na dużym zbiorze (KOR, n > 20k) — lasy losowe zyskują na dużej próbie
- **MICE** wygrywa na małym zbiorze (NEURO, n = 780) — iteracyjna regresja jest bardziej odporna na overfitting przy ograniczonej liczbie obserwacji
- **MICE wybrano jako metodę finalną** — jedna metoda działa dobrze na obu zbiorach, wyniki są przewidywalne i stabilne (σ = 0.0007 dla KOR, σ = 0.0025 dla NEURO w teście 10-seedowym)

---

## Finalna imputacja

| Parametr | Wartość |
|---|---|
| Metoda | MICE (ExtraTrees, `max_iter=7`, `n_estimators=78`) |
| Wejście | 78 197 rekordów, 13,4% brakujących wartości |
| Wyjście | 78 197 kompletnych rekordów |
| NaN po imputacji | **0** |
| Wartości spoza zakresu | **0** |
| Czas imputacji | 26,1 min |
| Zachowanie korelacji | Frobenius norm Δcorr = 1.21 (KOR), 2.56 (NEURO) |

Zaimputowany zbiór (`aneurysm_imputed_final.csv`) stanowi wejście do pipeline'u PU-learning.
