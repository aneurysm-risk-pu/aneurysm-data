# Podział danych po pacjentach + MICE wewnątrz foldu — metodologia

**Pliki:** `podzial.py` (podział), `aneurysm_sgkf_mice_pipeline.py` (imputacja w foldach), `tests/test_podzial.py`
**Wersja 1:** 11.06.2026 — StratifiedGroupKFold na rekordach, do raportu przejściowego
**Wersja 2:** 04.10.2026 — zamrożony podział po pacjentach ze stratyfikacją wielokluczową (sekcja 6)
**Dane wejściowe:** `data/imputation-inputs/aneurysm_concatted.csv` (38 cech, rekomendacja 0.4); wariant 35 cech nadal dostępny

---

## 1. Problem i motywacja

Dane zawierają **wielokrotne pomiary tych samych pacjentów** (`patient_id`): od 1 do 38 rekordów na osobę, średnio 1,91. Naiwny podział losowy (np. `train_test_split`) umieściłby różne pomiary tego samego pacjenta w zbiorze train i test jednocześnie. To **data leakage**: model „widzi” pacjenta podczas treningu i ocenia go w teście, co sztucznie zawyża metryki.

Dodatkowo zbiór jest **niezbalansowany**: 4,45% pacjentów ma `label=1` (NEURO), więc losowy podział mógłby zaburzyć proporcje klas w foldach.

---

## 2. Zastosowane podejście

### Podział: każdy pacjent w dokładnie jednym foldzie, proporcje wyrównane

| Warunek | Co zapewnia | Jak (v2) |
|---|---|---|
| **Group** | Wszystkie pomiary jednego pacjenta trafiają **wyłącznie** do train albo wyłącznie do test | podział robiony na tabeli pacjentów (jeden wiersz = jedna osoba) |
| **Stratified** | Zbliżone proporcje klas w każdym foldzie | `StratifiedKFold` po kluczu *etykieta × okno czasowe × liczba rekordów* |

Wersja 1 realizowała oba warunki przez `StratifiedGroupKFold` na rekordach. Wersja 2 realizuje je przez `StratifiedKFold` na pacjentach. Gwarancja jest ta sama: zero wspólnych pacjentów między train i test. Wersja 2 pozwala jednak stratyfikować po kilku zmiennych naraz i nie zależy od wersji sklearn. Powody zmiany opisuje sekcja 6.

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

Gdyby imputacja była wykonana **przed** podziałem (na całym zbiorze), statystyki z danych testowych wpłynęłyby na wypełnienie braków w train. To byłby leakage. Z tego powodu plik `2-imputation/final/results/aneurysm_imputed_cleaned.csv` (imputacja globalna) służy tylko do walidacji samej metody imputacji i nie może być wejściem do modeli.

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

**Uzasadnienie wyboru:** ExtraTrees wygrało z BayesianRidge we wszystkich przebiegach Optuny. Parametry wyznaczone dla KOR (`max_iter=7, n_est=78`) dają na NEURO RMSE gorsze tylko o 0,00001, przy 7× krótszym czasie, dlatego używamy jednego zestawu dla obu zbiorów.

Technicznie jest to pojedyncza, deterministyczna imputacja iteracyjna (`sample_posterior=False`), a nie klasyczne *multiple imputation*. Niepewność uzupełnienia nie jest propagowana do wyników.

---

## 4. Weryfikacja poprawności (asercje)

Dla każdego foldu skrypt sprawdza:

- `len(set(groups_train) ∩ set(groups_test)) == 0`: brak wspólnych `patient_id`
- `np.isnan(X_train_imp).sum() == 0` i to samo dla `X_test_imp`: brak NaN po imputacji
- *(v2)* brak wartości ujemnych po imputacji
- *(v2)* po pętli każdy pacjent wystąpił w części testowej **dokładnie raz**, co jest warunkiem poprawnych predykcji out-of-fold

Sam podział (`podzial.py`) sprawdza dodatkowo:

- każdy pacjent ma dokładnie jeden fold, a każdy fold zawiera pozytywnych,
- odcisk SHA-256 pliku danych zgadza się z zapisanym przy budowie podziału; jeśli dane się zmieniły, wczytanie podziału kończy się błędem.

`tests/test_podzial.py`: 10 testów (dane syntetyczne + zapisany artefakt), wszystkie przechodzą.

---

## 5. Wynik

`podzial.py` zapisuje do `results/`:

| Plik | Zawartość |
|---|---|
| `pacjent_fold.csv` | `patient_id, fold, label, mieszany, warstwa`: **źródło prawdy o podziale** |
| `podzial_diagnostyka.csv` | statystyki każdego foldu (tabela w sekcji 6.3) |
| `podzial_meta.json` | konfiguracja, odcisk danych, wersje bibliotek, data utworzenia |
| `podzial_porownanie_strategii.csv` | porównanie strategii z sekcji 6.2 |
| `etap0_sgkf_stabilnosc.csv` | siatka stabilności z 18.09.2026 (`etap0_sgkf_stabilnosc.py`, sklearn 1.8): wcześniejszy etap audytu |

`aneurysm_sgkf_mice_pipeline.py` → `main()` zwraca listę `splits`, czyli 5 słowników z kluczami:

```
fold                      — numer foldu, jak w pacjent_fold.csv (od 0)
X_train, X_test           — DataFrame, zaimputowane, oryginalna skala
y_train, y_test           — Series z labelami (0/1)
groups_train, groups_test — patient_id
```

---

## 6. Audyt wersji 1 i zmiany w wersji 2

### 6.1 Co było nie tak w wersji 1

| # | Problem | Skutek | Dowód |
|---|---|---|---|
| 1 | `StratifiedGroupKFold(n_splits=5)` bez `shuffle` | podział zależy od kolejności wierszy w CSV; `RANDOM_STATE = 42` nie miał na niego żadnego wpływu (działał tylko w MICE) | sygnatura sklearn: `shuffle=False` domyślnie; plik jest posortowany po etykiecie |
| 2 | przydział pacjentów nie był zapisywany | każdy etap liczy podział od nowa; przesortowanie pliku albo inna wersja sklearn po cichu zmienia foldy | — |
| 3 | `shuffle=True` w SGKF zależy od wersji sklearn | ta sama konfiguracja (seed 42) daje rozstęp udziału pozytywnych **0,11 pp w sklearn 1.8** i **0,80 pp w sklearn 1.6**, czyli faktycznie inne podziały na dwóch komputerach zespołu | `results/etap0_sgkf_stabilnosc.csv` vs przebieg z 04.10.2026 |
| 4 | stratyfikacja tylko po etykiecie rekordu | foldy nie są wyrównane pod względem czynników, o których wiemy, że niosą fałszywy sygnał: epoki badania i liczby rekordów pacjenta | sekcja 6.2 |
| 5 | 63 pacjentów z obiema etykietami bez jawnej reguły | SGKF trzyma ich w jednym foldzie (brak wycieku), ale etykieta pacjenta była nieokreślona | — |
| 6 | notebook `aneurysm_data_StratifiedGroupKFold.ipynb`: `X = df.drop(columns=['label'])` | `patient_id`, `custom_id` i `examination_date` zostają w `X` | kod notebooka; nie wpływa na sam podział, ale `X` nie nadaje się wprost do modelu |

Problem 4 jest istotny, bo podział, który nie wyrównuje czynników zakłócających, zwiększa zmienność wyniku między foldami. Część różnic między foldami wynikałaby wtedy z tego, ile do danego foldu trafiło NEURO z epoki KOR, a nie z jakości modelu.

### 6.2 Porównanie strategii

Te same dane (40 924 pacjentów, 1 823 pozytywnych, mieszani jako pozytywni), 5 foldów. Rozstęp = różnica między najbardziej a najmniej obciążonym foldem. Dla strategii tasowanych podany jest **najgorszy** wynik z 5 ziaren (42, 7, 13, 101, 2024).

| Strategia | Udział poz. (pacjenci) | Udział poz. (rekordy) | Rozmiar foldu (rekordy) | NEURO z okna KOR | NEURO z 5+ rekordami |
|---|---:|---:|---:|---:|---:|
| SGKF rekordowy, `shuffle=False` (v1) | 0,09 pp | 0,01 pp | 0,01% | 2,03 pp | 0,78 pp |
| SGKF rekordowy, `shuffle=True`¹ | 0,80 pp | 2,52 pp | 3,38% | 3,42 pp | 4,95 pp |
| pacjenci, stratyfikacja: etykieta (rekomendacja z etapu 0.6) | 0,01 pp | 2,04 pp | 3,93% | **5,18 pp** | **7,95 pp** |
| **pacjenci, etykieta × okno × n_rek (v2)** | **0,01 pp** | 1,14 pp | 1,66% | **0,79 pp** | **0,49 pp** |

¹ sklearn 1.6.1; w sklearn 1.8 wynik jest inny (problem 3).

**Wnioski:**

1. Rekomendacja z etapu 0.6 (pacjenci, stratyfikacja tylko po etykiecie) idealnie wyrównuje klasy, ale przy niektórych ziarnach **rozjeżdża foldy na czynnikach zakłócających**: do 5 pp w udziale NEURO z okna KOR i do 8 pp w udziale NEURO z 5+ rekordami. Siatka stabilności z etapu 0.6 tego nie mierzyła, bo patrzyła na udział lat 2020–2021 we wszystkich rekordach foldu, zdominowany przez KOR.
2. Stratyfikacja wielokluczowa naprawia to przy każdym z 5 ziaren, a balans klas zostaje bez zmian (0,01 pp).
3. Wersja 1 wypada dobrze na rekordach, bo SGKF optymalizuje właśnie proporcję rekordów. Przy modelu pacjentowym (reguła agregacji 0.3) ważniejsza jest proporcja pacjentów. Rozjazd 1,1 pp na rekordach w v2 wynika z tego, że pacjenci mają różną liczbę rekordów, i nie przenosi się na metryki pacjentowe.
4. Wersja 1 była zrównoważona przypadkiem, bo zależała od kolejności pliku. Etap 0.6 sprawdził, że tym razem to nie zaszkodziło, ale nie da się tego powtórzyć z innym ziarnem.

### 6.3 Zamrożony podział (v2, seed 42)

| Fold | Pacjenci | Rekordy | Pozytywni | Udział poz. | NEURO z okna KOR | NEURO z 5+ rek. | Mediana roku NEURO | Mieszani |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 185 | 15 614 | 364 | 4,45% | 9,9% | 21,2% | 2019 | 8 |
| 1 | 8 185 | 15 672 | 365 | 4,46% | 10,1% | 21,6% | 2018 | 15 |
| 2 | 8 185 | 15 699 | 365 | 4,46% | 10,4% | 21,4% | 2019 | 15 |
| 3 | 8 185 | 15 550 | 365 | 4,46% | 10,7% | 21,4% | 2019 | 10 |
| 4 | 8 184 | 15 662 | 364 | 4,45% | 10,2% | 21,4% | 2018 | 15 |

Warstwy stratyfikacji (12): etykieta {0, 1} × okno {w oknie dat KOR, poza} × liczba rekordów {1, 2, 3–4, 5+}. Pacjent jest „w oknie”, jeśli co najmniej połowa jego rekordów leży we wspólnym oknie dat kohort (2019-12-29 – 2021-12-26). Warstwa mniejsza niż 20 pacjentów byłaby scalona do „<etykieta>_inne”; przy obecnych danych najmniejsza liczy 34 osoby, więc scalenie nie zachodzi.

Mieszanych (63) nie dodano do klucza, bo dałoby to warstwy po kilka osób. Rozkład 8–15 na fold jest akceptowalny przy 0,15% udziale w danych.

### 6.4 Imputacja w foldzie a asymetria kohort

Przy audycie wyszła rzecz, której v1 nie raportował: **braki są rozłożone bardzo nierówno między kohortami**.

| | KOR | NEURO |
|---|---:|---:|
| komórki cech uzupełniane przez MICE | 12,3% | **26,0%** |
| kompletne wiersze (na nich uczy się scaler) | 20 663 | 780 |
| przykład: braki NRBC / %NRBC | 2,6% | 47,7% |
| przykład: braki NEUT / %NEUT | 2,9% | 42,3% |

Udział jest stabilny między foldami (KOR 12,1–12,5%, NEURO 25,2–27,8%). Pipeline v2 drukuje go dla każdego foldu.

**Co to znaczy:** około jednej czwartej profilu NEURO odtwarza imputer, który uczy się głównie na KOR (91% rekordów treningu). Wartości imputowane mogą być przez to „podobne do KOR”, co osłabia różnice, albo nieść ślad wzorca braków, co tworzy sztuczną różnicę. Wzorzec braków sam w sobie odróżnia kohorty. To nie jest błąd podziału, ale **ryzyko do opisania w raporcie** i kandydat do analizy wrażliwości w modelowaniu, np. model na samych cechach o niskim odsetku braków albo flagi braków.

Skalowanie na kompletnych wierszach jest w praktyce bezpieczne: poza zakresem kompletnych wierszy leży 209 obserwowanych wartości (0,008%).

### 6.5 Czego v2 nie rozstrzyga

- **Decyzja 0.1** (63 mieszanych): domyślnie `mieszani="pozytywni"`, a wariant `"bez_mieszanych"` jest jednym parametrem w `KonfiguracjaPodzialu`.
- **Decyzja 0.2** (kontrola czasu): stratyfikacja po oknie wyrównuje foldy, ale **nie usuwa** rozjazdu czasowego kohort. Model nadal może rozpoznawać epokę. To zadanie protokołu modelowania, nie podziału.
- **Spójność z przyszłym modelowaniem**: wstępna implementacja PU na branchu `pu-pipeline-1-5-lk` (`4-pu-setup/pu/foldy.py`) liczy własny podział (stratyfikacja tylko po etykiecie). Każdy przyszły kod modelowania powinien czytać przydział z `3-sgkf-split/results/pacjent_fold.csv` przez `podzial.wczytaj_podzial()`, a nie liczyć go od nowa.

---

## 7. Jak uruchomić

```bash
python 3-sgkf-split/podzial.py                    # buduje i zapisuje podział (sekundy)
python 3-sgkf-split/podzial.py --porownanie       # + tabela z sekcji 6.2
python 3-sgkf-split/podzial.py --sprawdz          # weryfikuje zapisany podział względem danych
python 3-sgkf-split/tests/test_podzial.py         # 10 testów
python 3-sgkf-split/aneurysm_sgkf_mice_pipeline.py --podprobka 3000 --szybkie-mice   # szybki test ścieżki
python 3-sgkf-split/aneurysm_sgkf_mice_pipeline.py                                   # pełny przebieg, ~5 × 20 min
```

Odcisk danych jest liczony po ujednoliceniu końców linii, więc ten sam podział wczytuje się na Windowsie (CRLF po checkoucie) i na macOS/Linux.
