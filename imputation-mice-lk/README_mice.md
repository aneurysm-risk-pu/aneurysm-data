# Imputacja danych brakujących — metoda MICE

**Autor:** kjubig | **Gałąź:** `imputation-mice-lk`  
**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (PU)

---

## 1. Kontekst projektu

Dane laboratoryjne pacjentów — zbiór NEURO (Positive) i KOR (Unlabeled) — zawierają naturalnie
brakujące wartości wynikające z różnych zakresów badań zlecanych różnym pacjentom. Przed etapem
modelowania PU konieczne jest uzupełnienie tych braków. Wybór metody imputacji wpływa bezpośrednio
na jakość reprezentacji danych, a przez to na zdolność modelu do nauki granic decyzyjnych.

W projekcie zaimplementowano dwa podejścia imputacyjne (porównanie równoległe):
- **KNN (k-Nearest Neighbors)** — imputacja przez uśrednianie k najbliższych sąsiadów (implementacja kolegi)
- **MICE (Multiple Imputation by Chained Equations)** — imputacja iteracyjna przez regresję (niniejszy moduł)

---

## 2. Teoria braków danych

### 2.1 Mechanizmy powstawania braków (Rubin, 1976)

W literaturze statystycznej wyróżnia się trzy mechanizmy:

| Mechanizm | Definicja | Przykład |
|-----------|-----------|---------|
| **MCAR** *(Missing Completely At Random)* | Brak niezależny od danych obserwowanych i nieobserwowanych | Losowa awaria analizatora |
| **MAR** *(Missing At Random)* | Brak zależy od danych obserwowanych, ale nie od brakującej wartości | Brak HBA1C u pacjentów bez cukrzycy w historii |
| **MNAR** *(Missing Not At Random)* | Brak zależy od samej brakującej wartości | Brak pomiaru przy bardzo wysokim/niskim wyniku |

Dla danych laboratoryjnych dominuje mechanizm **MAR** — obecność lub brak danego badania
jest zdeterminowana profilem klinicznym pacjenta (obserwowanymi cechami), a nie wartością samego wyniku.
MICE jest metodą zaprojektowaną właśnie dla danych MAR.

### 2.2 Problemy z prostymi metodami imputacji

- **Imputacja średnią/medianą**: eliminuje wariancję w imputowanych kolumnach; niszczy korelacje
  między parametrami (np. HGB–HCT–RBC są silnie skorelowane — imputacja jednego powinna wykorzystywać pozostałe)
- **Imputacja stałą (zero/flaga)**: wprowadza artefakty; wartość zero ma znaczenie kliniczne (brak vs. wynik = 0)
- **KNN**: zachowuje lokalne korelacje, ale skala odległości euklidesowej jest wrażliwa na jednostki
  i zakłada liniową strukturę sąsiedztwa

---

## 3. Algorytm MICE

### 3.1 Idea

MICE (znany też jako *Fully Conditional Specification*, FCS; van Buuren & Groothuis-Oudshoorn, 2011)
wypełnia braki iteracyjnie, modelując każdą kolumnę z brakami jako zmienną zależną w osobnym modelu
regresji, gdzie pozostałe kolumny są predyktorami.

### 3.2 Przebieg algorytmu

```
Inicjalizacja: wypełnij braki wartościami średnimi (start)

Dla każdej iteracji t = 1, ..., max_iter:
  Dla każdej kolumny j z brakami:
    1. Użyj aktualnie imputowanych wartości pozostałych kolumn jako X
    2. Wytrenuj model f_j(X) na wierszach kompletnych dla kolumny j
    3. Przewidź brakujące wartości w kolumnie j: ŷ_j = f_j(X_missing)
    4. Zastąp braki w kolumnie j wartościami ŷ_j

Wynik: macierz bez braków po max_iter przejściach
```

### 3.3 Zbieżność

Algorytm działa na zasadzie Gibbs samplingu — każda kolumna jest warunkowana na wszystkich
pozostałych. W praktyce kilka–kilkanaście iteracji wystarcza do zbieżności (co potwierdzają
nasze eksperymenty: różnica RMSE między max_iter=5 a max_iter=50 wynosi < 0.001).

### 3.4 Estymator wewnętrzny a jakość imputacji

Wybór modelu regresji f_j ma kluczowe znaczenie:

| Estymator | Charakter | Zalety | Wady |
|-----------|-----------|--------|------|
| **BayesianRidge** | Liniowy | Bardzo szybki, stabilny numerycznie | Zakłada liniowe zależności |
| **ExtraTreesRegressor** | Nieliniowy (drzewa) | Modeluje nieliniowości, odporny na outliery | Wolniejszy, wymaga doboru n_estimators |

Parametry laboratoryjne wykazują **nieliniowe** zależności kliniczne, np.:
- eGFR jest nieliniową funkcją kreatyniny, wieku i płci (wzór CKD-EPI)
- morfologia krwi: relacja MCV–MCHC–RDW jest nieliniowa przy stanach chorobowych
- koaguologia: APTT–INR przy terapii antykoagulacyjnej

Dlatego ExtraTreesRegressor dominuje nad BayesianRidge we wszystkich testowanych konfiguracjach.

---

## 4. Metodyka implementacji

### 4.1 Porównanie podejść (Metodyki A/B/C)

W trakcie projektu zidentyfikowano i porównano trzy możliwe podejścia (`mice_compare_methodologies.py`):

| Metodyka | Normalizacja | MICE na | Metryki | Status |
|----------|-------------|---------|---------|--------|
| **A** | MinMax PRZED MICE | danych [0,1] | [0,1] | ✅ Przyjęta — spójna end-to-end |
| B | brak | danych surowych | surowych | ❌ Nieporównywalna z KNN |
| C (hybrid) | MinMax PO MICE | danych surowych | [0,1] | ❌ Niespójna — MICE uczy się bez skali |

**Metodyka A** jest jedyną metodologicznie spójną: MICE uczy się relacji między zmiennymi
w tej samej przestrzeni, w której oceniamy jego jakość.

### 4.2 Potok przetwarzania (Metodyka A)

```
Dane surowe (z NaN)
        │
        ▼
[1] MinMaxScaler.fit()  ←── tylko kompletne wiersze (bez NaN)
        │
        ▼
[2] .transform(pełny zbiór)  →  dane w [0,1], NaN pozostają NaN
        │
        ▼
[3] IterativeImputer.fit_transform()  →  dane w [0,1] bez NaN
        │
        ▼
[4] MinMaxScaler.inverse_transform()  →  oryginalne jednostki bez NaN
        │
        ▼
Plik CSV z imputowanymi danymi
```

**Dlaczego MinMaxScaler zamiast StandardScaler?**  
Parametry laboratoryjne mają dolne ograniczenie (nie ujemne). MinMax zachowuje tę właściwość
i jest zgodny z parametrem `min_value=0.0, max_value=1.0` IterativeImputera — gwarantuje,
że żadna imputowana wartość nie wyjdzie poza zaobserwowany zakres danych.

### 4.3 Dlaczego nie train/test split przy imputacji?

Imputacja jest **krokiem preprocessingowym**, nie modelem predykcyjnym. Jej celem jest
uzupełnienie macierzy danych wejściowych przed właściwym modelowaniem PU. Podział na
zbiory treningowy/testowy należy do etapu modelowania — dzielenie danych *przed* imputacją
spowodowałoby wyciek informacji (scaler fitowany na train musiałby transformować test,
co jest identyczne z klasycznym preprocessingiem w pipeline).

---

## 5. Walidacja

### 5.1 Masked validation (walidacja przez maskowanie)

Brak zewnętrznego "ground truth" dla imputacji (nie znamy prawdziwych brakujących wartości).
Standardowym rozwiązaniem jest **symulacja braków na kompletnych wierszach**:

```
1. Wybierz wiersze bez żadnych NaN
2. Znormalizuj MinMax (te same parametry co do docelowej imputacji)
3. Zamaskuj losowo ~10% wartości (całomacierzowo: rng.random(shape) < 0.10)
4. Uruchom IterativeImputer.fit_transform() na zamaskowanej macierzy
5. Oblicz metryki: RMSE, MAE, KL między wartościami oryginalnymi a imputowanymi
```

Próg 10% odpowiada typowemu poziomowi braków w danych; podejście jest identyczne z walidacją
użytą przez kolegę w implementacji KNN (umożliwia bezpośrednie porównanie).

### 5.2 Metryki

**RMSE (Root Mean Squared Error)** na skali [0,1]:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$

Penalizuje duże błędy kwadratowo — ważne przy parametrach klinicznych, gdzie duże odchylenie
od rzeczywistości może zmienić interpretację wyniku.

**MAE (Mean Absolute Error)** na skali [0,1]:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

Intuicyjnie interpretowalny: średni błąd imputacji jako ułamek pełnego zakresu danego parametru.

**KL Divergence** (Kullback–Leibler, histogram-based):

$$D_{KL}(P \| Q) = \sum_{k} P(k) \log \frac{P(k)}{Q(k)}$$

Mierzy, czy rozkład imputowanych wartości jest zbieżny z rozkładem oryginalnych.
RMSE/MAE mierzą błąd punkt-po-punkcie; KL mierzy, czy imputacja zachowuje **kształt rozkładu**
(np. czy bimodalny rozkład hemoglobiny jest zachowany). Ważne dla późniejszego modelowania PU —
model uczy się z rozkładu cech.

---

## 6. Optymalizacja hiperparametrów

### 6.1 Przestrzeń przeszukiwania

Analogicznie do kolegi (KNN: K ∈ {1,3,5,7,...,21} × wagi ∈ {uniform, distance}):

| Parametr | Wartości | Liczba kombinacji |
|----------|---------|-------------------|
| `estimator` | BayesianRidge, ExtraTrees(n=10), ExtraTrees(n=50), ExtraTrees(n=100) | 4 |
| `max_iter` | 5, 10, 20, 30, 50 | 5 |
| **Łącznie** | | **20 konfiguracji** |

Dla ExtraTrees `n_jobs=1` (jednowątkowy) — zapobiega MemoryError przy dużych zbiorach (KOR: 72k wierszy).

### 6.2 Uzasadnienie wyboru `n_estimators`

Więcej drzew → mniejsza wariancja estymatora → stabilniejsze imputacje. W praktyce:
- n=10: szybkie, wystarczające dla silnych korelacji
- n=50: dobry kompromis jakość/czas
- n=100: diminishing returns powyżej 50, wzrost czasu obliczeń liniowy

Wyniki potwierdzają: ExtraTrees(n=100) nieznacznie dominuje, ale n=50 jest bliskie.

---

## 7. Wyniki

### 7.1 NEURO (zbiór Positive, 7 370 pacjentów, 30 cech)

Próbka walidacyjna: 2 228 kompletnych wierszy (wszystkie).

| Rank | Estymator | max_iter | RMSE | MAE | KL |
|------|-----------|----------|------|-----|----|
| 🥇 1 | ExtraTrees(n=100) | 10 | **0.0487** | **0.0203** | 1.1102 |
| 2 | ExtraTrees(n=100) | 5 | 0.0490 | 0.0204 | 1.1967 |
| 3 | ExtraTrees(n=50) | 5 | 0.0491 | 0.0207 | 1.0931 |
| ... | ExtraTrees(n=10) | 5–50 | 0.051–0.052 | 0.022 | ~0.95 |
| ... | BayesianRidge | 5–50 | 0.056–0.056 | 0.024 | ~1.10 |

KNN referencja (K=7, distance): RMSE=**0.1169**, MAE=**0.0579**

### 7.2 KOR (zbiór Unlabeled, 72 653 pacjentów, 35 cech)

Próbka walidacyjna: 5 000 losowych kompletnych wierszy (z 22 642 dostępnych).

| Rank | Estymator | max_iter | RMSE | MAE | KL |
|------|-----------|----------|------|-----|----|
| 🥇 1 | ExtraTrees(n=50) | 30 | **0.0341** | **0.0121** | 0.6903 |
| ~1 | ExtraTrees(n=100) | 5–50 | ~0.0341 | ~0.0120 | ~0.70 |
| ... | ExtraTrees(n=10) | 5–50 | 0.035–0.036 | 0.013 | ~0.50 |
| ... | BayesianRidge | 5–50 | 0.040–0.041 | 0.016 | ~0.77 |

KNN referencja (K=21, distance): RMSE=**0.0993**, MAE=**0.0436**

### 7.3 Porównanie MICE vs KNN

| Zbiór | Metoda | RMSE | MAE | Poprawa RMSE |
|-------|--------|------|-----|--------------|
| NEURO | MICE — ExtraTrees(n=100), max_iter=10 | 0.0487 | 0.0203 | |
| NEURO | KNN — K=7, distance | 0.1169 | 0.0579 | **MICE lepszy o ~58%** |
| KOR | MICE — ExtraTrees(n=50), max_iter=30 | 0.0341 | 0.0121 | |
| KOR | KNN — K=21, distance | 0.0993 | 0.0436 | **MICE lepszy o ~66%** |

### 7.4 Interpretacja

**Dlaczego MICE >> KNN?**  
KNN imputuje przez uśrednianie k sąsiadów w przestrzeni euklidesowej — zakłada tym samym,
że zależności między cechami są lokalne i liniowe. Parametry laboratoryjne mają złożone,
nieliniowe relacje kliniczne. MICE z ExtraTrees modeluje je bezpośrednio jako problem regresji
drzewiastej, co pozwala wychwycić nieliniowości i interakcje niemożliwe do uchwycenia przez KNN.

**Dlaczego ExtraTrees > BayesianRidge?**  
BayesianRidge to regresja liniowa — nawet z regularyzacją bayesowską nie modeluje nieliniowych
zależności między parametrami morfologii, biochemii i koagulologii.

**Obserwacja dla max_iter:**  
Zbieżność jest szybka — małe wartości max_iter (5–10) wystarczają. Dla ExtraTrees wyższe max_iter
nawet nieznacznie pogarsza wyniki (przeuczenie na etapie imputation chain). Dla BayesianRidge
różnica jest pomijalna (< 0.001 RMSE).

---

## 8. Struktura plików

```
inputation-mice/
├── mice_hyperparameter_search.py       # Główny skrypt: optymalizacja + finalna imputacja
├── mice_compare_methodologies.py       # Uzasadnienie wyboru Metodyki A (A vs B vs C)
├── README_mice.md                      # Niniejszy dokument
└── results/
    ├── neuro_mice_hyperparameter_results.csv   # Ranking 20 konfiguracji — NEURO
    ├── kor_mice_hyperparameter_results.csv     # Ranking 20 konfiguracji — KOR
    ├── mice_hyperparameter_all_results.csv     # Połączony ranking obu zbiorów
    ├── neuro_mice_best_imputed.csv             # NEURO z imputacją (najlepsza konfiguracja)
    ├── kor_mice_best_imputed.csv               # KOR z imputacją (najlepsza konfiguracja)
    ├── wykres_mice_neuro.png                   # RMSE/MAE/KL vs max_iter — NEURO
    └── wykres_mice_kor.png                     # RMSE/MAE/KL vs max_iter — KOR
```

---

## 9. Uruchomienie

```bash
# Pełna optymalizacja hiperparametrów + zapis finalnie imputowanych zbiorów
python inputation-mice/mice_hyperparameter_search.py

# Porównanie metodyk A/B/C (dokumentacyjne)
python inputation-mice/mice_compare_methodologies.py
```

**Wymagania:** `scikit-learn >= 1.2`, `pandas`, `numpy`, `matplotlib`, `seaborn`  
**Czas działania:** ~15–30 min (ExtraTrees n=100, KOR 72k wierszy, n_jobs=1)

---

## 10. Literatura

- Rubin, D.B. (1976). *Inference and Missing Data*. Biometrika, 63(3), 581–592.
- van Buuren, S. & Groothuis-Oudshoorn, K. (2011). *mice: Multivariate Imputation by Chained Equations in R*. Journal of Statistical Software, 45(3).
- Geurts, P., Ernst, D., Wehenkel, L. (2006). *Extremely randomized trees*. Machine Learning, 63(1), 3–42.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press. [online: stefvanbuuren.name/fimd]

## Mozliwe rozszerzenia

- **Wiecej drzew ExtraTrees** — przetestowac `n_estimators=50/100` na NEURO (male dane, szybkie);
  jesli wyniki sie nie poprawiaja, n=10 jest uzasadniony
- **Metryki dystrybucyjne** — dodac KL divergence per kolumna do pliku walidacyjnego;
  RMSE/MAE mierzy dokladnosc punktowa, KL mierzy wiernosc rozkladu (istotne dla downstream PU)
- **Filtrowanie kolumn** — kolega KNN odrzuca kolumny z >60% brakow (prog 40% dostepnosci);
  moze poprawic zbieznosc MICE na NEURO (nie zbiega w 30 iteracjach przez kolumny jak ALT 61%, AST 66%)

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
