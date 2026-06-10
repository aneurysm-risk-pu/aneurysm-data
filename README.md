# Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled

**Zespołowy Projekt Badawczy 2026**

## Dane

| Plik | Zbiór | Pacjenci | Rekordy | Śr. rekordów/pacjent |
|------|-------|---------|---------|----------------------|
| `datasets/neuro_merged_aggregated_1W_mean.csv` | **Positive** (pacjenci z tętniakiem) | 2 016 | 7 655 | 3.8 |
| `datasets/kor_merged_aggregated_1W_mean.csv` | **Unlabeled** (pacjenci ogólnie) | 39 893 | 73 679 | 1.8 |

Rekord = jeden tydzień zagregowanych (średnia) wyników laboratoryjnych danego pacjenta; klucz: `{patient_id}-{rok}-W{tydzień}`.

## Cechy

Morfologia (HGB, RBC, WBC, PLT, MCV, MCH, MCHC, HCT, RDW), biochemia (KREA, BUN, Na, K, GLU, ALT, AST), koagulacja (PT, INR, APTT), CRP, eGFR (MDRD, CKD-EPI).

Dla każdego parametru: wartość + flaga normy (`-1` / `0` / `1`).

## TODO 

- Zapoznać się z danymi - spotkanie 09.04
- Zwerfikować dane pod względem poprawności (niepoprawny wiek itd.) - spotkanie 09.04
- Obsłużyć braki danych - spotkanie 09.04




-  **część pacjentów z NEURO pojawia się również w KOR** - co zrobić? - @kjubig