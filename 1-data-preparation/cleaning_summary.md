# Podsumowanie czyszczenia danych NEURO

### Znalzione błędy i anomalie (sprint 1)

| Problem | Szczegóły |
|---------|-----------|
| Kolumny ze słownikami | Kolumny w formacie `{'values': [...]}` — nie nadają się do analizy |
| Duplikat `custom_id` | 1 duplikat: `'194143-2017-W52'` |

---

### Pierwsze wykonane czyszczenie (`{'values': [...]}`)

Skrypt: `clean_datasets_values.py`

| Kolumna / typ | Działanie | Uzasadnienie |
|---------------|-----------|--------------|
| `examination_type` | **Usunięta** | Zawierała wyłącznie wartości `{'values': [...]}` z nazwami badań — brak wartości numerycznej |
| `descriptive_result` | **Usunięta** | Zawierała wyłącznie wartości `{'values': [...]}` z opisami tekstowymi — brak wartości numerycznej |
| Pozostałe kolumny numeryczne z `{'values': [...]}` | **Uśrednione** | Plik źródłowy to agregat tygodniowy (`_1W_mean`) — komórki z listą to błąd pipeline agregacji; średnia dokańcza zamierzoną operację |

> **Do sprawdzenia:** przypadki o dużej wariancji w tygodniu (np. CRP: `[10.5, 80.1, 113.2]`) — uśrednianie może maskować gwałtowne zmiany kliniczne.


### Drugie wykonane czyszczenie (duplikaty `custom_id`)

Skrypt: `clean_anomalies.py`

| custom_id | Działanie | Uzasadnienie |
|-----------|-----------|--------------|
| `194143-2017-W52` | Usunięto starszy rekord (2017-01-01), zachowano nowszy (2017-12-31) | Ten sam `patient_id`, ten sam tydzień — nowszy rekord ma inny wiek (72 vs 71) - najprawdopobniej wynika to ze spsobu liczenia|

