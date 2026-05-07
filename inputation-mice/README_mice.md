# Imputacja MICE

## Co to jest MICE?

MICE (Multiple Imputation by Chained Equations) wypełnia brakujące wartości w danych.
Dla każdej kolumny z brakami trenuje model regresji na pozostałych kolumnach i przewiduje brakujące wartości.
Proces powtarza się iteracyjnie (tu: 30 razy) aż wyniki się ustabilizują.

Zaleta nad średnią/medianą: zachowuje korelacje między parametrami laboratoryjnymi.

## Dane wejściowe

- `datasets/cleaned/neuro_cleaning.csv` — zbiór POSITIVE (pacjenci z tętniakiem)
- `datasets/cleaned/kor_cleaning.csv` — zbiór UNLABELED (populacja ogólna)

Imputowane są kolumny wartości (HGB, WBC, KREA, Na, K, CRP, ALT, AST...) — **bez** flag `-norm`, `diagnosis_id` i kolumn meta.

## Jak uruchomić

```bash
python inputation-mice/mice_imputation.py
```

Wymagane: `scikit-learn`, `pandas`, `scipy`, `numpy`

## Podział train/test

Podział 80/20 **na poziomie `patient_id`** (nie rekordów) — żeby rekordy tego samego pacjenta nie trafiły jednocześnie do train i test.

`fit()` imputera tylko na train → `transform()` na train i test osobno (brak data leakage).

## Wyniki

Zapisywane do `inputation-mice/results/`:

| Plik | Opis |
|------|------|
| `neuro_mice_train.csv` | NEURO po imputacji — zbiór treningowy |
| `neuro_mice_test.csv` | NEURO po imputacji — zbiór testowy |
| `neuro_mice_validation.csv` | Metryki walidacji dla NEURO |
| `kor_mice_train.csv` | KOR po imputacji — zbiór treningowy |
| `kor_mice_test.csv` | KOR po imputacji — zbiór testowy |
| `kor_mice_validation.csv` | Metryki walidacji dla KOR |

## Walidacja

Dla każdej cechy porównano rozkład **wartości oryginalnych** (tam gdzie dane były) z **wartościami wstawionymi w miejsca braków** (tam gdzie ich nie było).

### KL divergence — główna metryka

KL divergence (Kullback-Leibler) mierzy **jak bardzo różnią się dwa rozkłady**. Im mniejsza wartość, tym lepiej.

```
KL(P || Q) = Σ P(x) * log( P(x) / Q(x) )
```

Gdzie P = rozkład oryginalnych wartości, Q = rozkład imputowanych wartości.

| KL | Interpretacja |
|----|---------------|
| < 1 | Bardzo dobra imputacja — rozkłady praktycznie identyczne |
| 1 – 5 | Akceptowalna |
| > 10 | Imputacja zniekształca rozkład |

**Ta metryka służy do porównania MICE z KNN** — zestawiamy tabele `*_mice_validation.csv` i `*_knn_validation.csv` i patrzymy które KL jest niższe dla każdej cechy.

### Test KS — metryka pomocnicza (używać ostrożnie)

Test KS (Kolmogorov-Smirnov) odpowiada TAK/NIE: *czy dwa zestawy danych pochodzą z tego samego rozkładu?*

- p > 0.05 → rozkłady nieróżnią się istotnie → dobra imputacja
- p ≤ 0.05 → rozkłady się różnią

**Ważne:** przy tysiącach rekordów (6k–73k) test KS wykrywa nawet minimalne różnice jako "istotne statystycznie". U nas p≈0.0 dla wszystkich cech — to **nie jest błąd imputacji**, to właściwość testu przy dużym N. Dlatego główną metryką jest KL divergence, a nie KS.

---

## Wyniki — podsumowanie jakości

**Dobre (KL < 1):** KREA, Na, K, ALT, AST, BUN, CRP, GLU, INR, PT, APTT, NEUT, MONO, LYMPH, EO, BAZO i ich frakcje procentowe — imputacja zachowuje rozkład oryginału.

**Słabe (KL > 10):** HGB, RBC, MCV, HCT, MCH, MPV — parametry morfologii krwi.

Morfologia wypada słabo nie z powodu błędu algorytmu, ale z powodu **struktury danych**: brakuje jej zawsze w całości (HGB, RBC, MCV, HCT, MCH jednocześnie), bo to jeden pomiar. MICE nie ma wtedy żadnych "sąsiadów" do nauki.

---

## Możliwe ulepszenia

1. **Usunąć cechy z bardzo dużym odsetkiem braków** — cechy takie jak ALT (61%), AST (66%), BUN (54%), GLU (56%), APTT (55%) mają ponad połowę braków. MICE wypełnia je "w ciemno" — na podstawie innych cech, które same mogą być imputowane. Można przyjąć próg np. >50% braków → cecha usuwana ze zbioru przed imputacją. To zmniejsza szum i przyspiesza zbieżność NEURO.

2. **Dodać `patient_age` i `patient_sex` jako predyktory** — są zawsze obecne i mogą pomóc dla morfologii (HGB/RBC różnią się między kobietami i mężczyznami). Wymaga małej zmiany w skrypcie.

3. **Imputować osobno kobiety i mężczyzn** — morfologia ma różne normy dla obu płci, wspólna imputacja "miesza" dwa rozkłady.

4. **Zaakceptować i odnotować w raporcie** — dla parametrów biochemicznych (KREA, Na, K, CRP...) imputacja jest dobra. Dla morfologii jakość jest ograniczona ze względu na brak częściowych pomiarów — to uczciwe naukowo i najczęstsze podejście w badaniach medycznych.

---

## Uwagi techniczne

- **KOR** — zbiegło po 11 iteracjach ✓
- **NEURO** — nie zbiegło (oscyluje przy `max_iter=30`). Przyczyna: wiele cech ma >50% braków (ALT, AST, BUN, GLU, INR...). Przy takiej ilości braków MICE ciężko ustabilizować — udokumentowane zachowanie `IterativeImputer`, nie błąd kodu.
