# Podsumowanie czyszczenia danych

### Znalzione błędy i anomalie (sprint 1)

| Problem | Szczegóły |
|---------|-----------|
| Kolumny ze słownikami | Kolumny w formacie `{'values': [...]}` — nie nadają się do analizy |
| Błędny wiek | 1 rekord z wiekiem = 222 lata |
| Duplikat `custom_id` | 1 duplikat: `'194143-2017-W52'` |
| Powtarzający się pacjenci | 358 `patient_id` pojawia się więcej niż raz |

---

### Pierwsze wykonane czyszczenie (`{'values': [...]}`)

Skrypt: `clean_datasets_values.py`

| Kolumna / typ | Działanie | Uzasadnienie |
|---------------|-----------|--------------|
| `examination_type` | **Usunięta** | Zawierała wyłącznie wartości `{'values': [...]}` z nazwami badań — brak wartości numerycznej |
| `descriptive_result` | **Usunięta** | Zawierała wyłącznie wartości `{'values': [...]}` z opisami tekstowymi — brak wartości numerycznej |
| Pozostałe kolumny numeryczne z `{'values': [...]}` | **Uśrednione** | Plik źródłowy to agregat tygodniowy (`_1W_mean`) — komórki z listą to błąd pipeline agregacji; średnia dokańcza zamierzoną operację |

> **Do sprawdzenia:** przypadki o dużej wariancji w tygodniu (np. CRP: `[10.5, 80.1, 113.2]`) — uśrednianie może maskować gwałtowne zmiany kliniczne.


