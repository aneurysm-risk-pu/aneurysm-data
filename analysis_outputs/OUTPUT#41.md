# Raport z preprocessingu i walidacji danych (kohorty KOR i NEURO)

## 1. Dane wejściowe i weryfikacja struktury
* **Źródła danych:** `kor_merged_aggregated_1W_mean.csv` (kohorta kardiologiczna) oraz `neuro_merged_aggregated_1W_mean.csv` (kohorta neurologiczna).
* **Spójność schematu:** Zweryfikowano nazwy i liczbę kolumn w obu zbiorach — oba pliki posiadały identyczną strukturę liczącą **128 kolumn** (8 kolumn metadanych pacjenta/badania oraz 40 parametrów laboratoryjnych rozbitych na 3 kolumny każdy: wartość główna, `-unit`, `-norm`)[cite: 17].

## 2. Czyszczenie i walidacja (Preprocessing w notatniku)
* **Deduplikacja wewnątrz kohort:** Sprawdzono unikalność klucza `custom_id` (`patient_id-YYYY-Www`). W zbiorze `KOR` nie wykryto duplikatów (0), natomiast w `NEURO` zidentyfikowano 2 zduplikowane wiersze (`194143-2017-W52`), z których zachowano pierwsze wystąpienie (`keep='first'`).
* **Analiza nakładania się zbiorów (KOR vs NEURO):** Wykryto **559 wspólnych rekordów** (`custom_id`) występujących w obu plikach jednocześnie, z czego w 557 wierszach odnotowano różnice w wartościach (np. w `patient_age`, `examination_type` czy wynikach laboratoryjnych).
* **Walidacja i imputacja jednostek (`-unit`):** Przeskanowano wszystkie kolumny jednostek pod kątem niespójności względem mody — nie wykryto błędnych/mieszanych jednostek. Brakujące wartości (`NaN`) w kolumnach `-unit` uzupełniono najczęstszą wartością dla danego parametru (wyjątkiem są `INR-unit` oraz `WAPTT-unit`, które jako wielkości bezwymiarowe w całości zawierały `NaN`).
* **Ekstrakcja i rozpakowanie zagnieżdżonych słowników:**
  * Kolumnę `descriptive_result` przekształcono ze stringów słownikowych `{'values': [...]}` na czysty tekst rozdzielony spacją.
  * Przeprowadzono automatyczne rozpakowanie wszystkich pozostałych kolumn zawierających strukturę `{'values': [...]}` (w tym `examination_type` oraz kolumn pomiarowych, takich jak `BUN`) do postaci wartości oddzielonych przecinkiem.
* **Redukcja wymiarowości i zapis:** Usunięto wszystkie pomocnicze kolumny jednostek (`-unit`) oraz flag norm (`-norm`), redukując zbiór do kolumn metadanych i 40 czystych cech pomiarowych. Oczyszczone zbiory zapisano jako `kor_cleaned1.csv` oraz `neuro_cleaned1.csv`[cite: 17].

## 3. Weryfikacja analityczna i kontrola wycieku danych (`issue41_3.py`)
* **Audyt pokrycia cech między kohortami:** Skrypt wczytuje `kor_cleaned1.csv` i `neuro_cleaned1.csv`, wyodrębnia rok badania z `examination_date` (lub awaryjnie z `custom_id`) i zlicza niepuste pomiary (`notna()`) dla każdej z 40 cech w podziale na kohortę i rok[cite: 16].
* **Klasyfikacja zmiennych (detekcja *data leakage*):** Każda pojedyncza cecha jest klasyfikowana do jednej z trzech grup:
  * `BOTH` — cechy występujące w obu kohortach,
  * `ONLY_KOR` — cechy obecne wyłącznie w kohorcie KOR,
  * `ONLY_NEURO` — cechy obecne wyłącznie w kohorcie NEURO (potencjalne źródło wycieku danych dla modeli klasyfikacyjnych)[cite: 16].
* **Identyfikacja synonimów i mapowanie nazw:** Przeprowadzono automatyczne wykrywanie synonimów w `examination_type` (na podstawie podobieństwa tekstowego oraz wektorów obecności wyników powyżej 80%) oraz zmapowano skróty kolumn (np. `KREA`, `HGB`, `CRP`) na ich pełne polskie nazwy kliniczne, generując końcowe tabele przestawne, wykresy oraz raport pokrycia cech[cite: 16, 17].