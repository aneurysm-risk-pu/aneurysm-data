# Podsumowanie analizy danych — NEURO (Sprint 1)

---

## 1. Wykonane czyszczenie strukturalne

| Krok | Skrypt | Co zrobiono | Wynik |
|------|--------|-------------|-------|
| 1 | `clean_datasets_values.py` | Usunięto kolumny `examination_type` i `descriptive_result` (tylko tekst); pozostałe komórki `{'values': [...]}` zastąpiono średnią | 7655 → 7655 wierszy, 128 → 126 kolumn |
| 2 | `clean_anomalies.py` | Usunięto starszy duplikat `custom_id = '194143-2017-W52'` (zachowano rekord z 2017-12-31) | 7655 → 7654 wierszy |

**Aktualny stan:** `datasets/cleaned/neuro_cleaning.csv` — 7654 wiersze × 126 kolumn

---

## 2. Anomalie medyczne — wyniki `check_med_data.py`

### WBC — Leukocyty (białe krwinki)
- **Badanie:** Morfologia krwi
- **Norma:** 4–10 G/l
- **Próg anomalii:** > 100 G/l
- **Znalezione w NEURO:** 3 rekordy

| patient_id | data | WBC [G/l] |
|------------|------|-----------|
| 430573 | 2021-04-04 | 178.97 |
| 1746102 | 2015-10-18 | 154.86 |
| 1866960 | 2017-09-10 | 119.59 |


---

### PLT — Płytki krwi (trombocyty)
- **Badanie:** Morfologia krwi
- **Norma:** 150–400 G/l
- **Próg anomalii:** PLT = 0
- **Znalezione w NEURO:** 0 rekordów (problem dotyczy KOR)

---

### Na — Sód
- **Badanie:** Elektrolity
- **Norma:** 136–145 mmol/l
- **Próg anomalii:** < 100 mmol/l (biologicznie niemożliwe)
- **Znalezione w NEURO:** 7 rekordów

| patient_id | data | Na [mmol/l] |
|------------|------|-------------|
| 195553 | 2020-12-13 | 92.8 |
| 410779 | 2019-02-24 | 74.0 |
| 906367 | 2012-11-04 | 22.0 |
| 1025691 | 2015-06-07 | 74.0 |
| 1036163 | 2010-11-28 | 56.0 |
| 1803659 | 2016-07-10 | 70.5 |
| 2500679 | 2024-08-25 | 95.0 |


---

### K — Potas
- **Badanie:** Elektrolity
- **Norma:** 3.5–5.0 mmol/l
- **Próg anomalii:** > 9 mmol/l (śmiertelna hiperkaliemia)
- **Znalezione w NEURO:** 13 rekordów

| patient_id | data | K [mmol/l] |
|------------|------|------------|
| 142817 | 2019-06-16 | 11.2 |
| 279733 | 2013-05-05 | 12.85 |
| 298782 | 2012-11-18 | **23.9** |
| 410779 | 2019-01-27 | 13.75 |
| 410779 | 2019-02-24 | 13.4 |
| 906367 | 2012-11-04 | **16.0** |
| 1025691 | 2015-06-07 | **39.0** |
| 1036163 | 2010-11-28 | **34.0** |
| 1600890 | 2016-10-02 | **22.8** |
| 1803659 | 2016-07-10 | 9.05 |
| 1835278 | 2023-09-24 | **18.875** |
| 2125188 | 2020-06-28 | 9.9 |
| 2394218 | 2023-08-13 | 10.38 |


> Pacjent `410779` i `906367` mają jednocześnie anomalie Na i K w tym samym tygodniu - prawdopodobnie błędne pobranie lub pomyłka w wpisie.

---

### KREA — Kreatynina
- **Badanie:** Biochemia — funkcja nerek
- **Norma:** 0.6–1.2 mg/dl
- **Próg anomalii:** > 20 mg/dl (pierwotna hipoteza: błąd jednostek µmol/l)
- **Wniosek po analizie:** Wartości > 20 mg/dl **są klinicznie możliwe** u pacjentów z ciężką niewydolnością nerek (stadium dializy). Przelicznik ÷ 88.4 daje wartości < 1 mg/dl co jest zbyt niskie. **Nie przeliczamy, nie usuwamy.**

---

## 3. Do zrobienia dalej w NEURO - co do tych danych

- 
- 
- 

## 4. Na przyszłość - kwestie KOR

- Błędny wiek (222 lata) 
- PLT = 0 
- Ujednolicenie płci (`"0.0"`/`"1.0"` vs `"0"`/`"1"`)
