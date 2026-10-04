# Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled

**Zespołowy Projekt Badawczy 2026**

## Dane

| Plik | Zbiór | Pacjenci | Rekordy | Śr. rekordów/pacjent |
|------|-------|---------|---------|----------------------|
| `data/raw/neuro_merged_aggregated_1W_mean.csv` | **Positive** (pacjenci z tętniakiem) | 2 016 | 7 655 | 3.8 |
| `data/raw/kor_merged_aggregated_1W_mean.csv` | **Unlabeled** (pacjenci ogólnie) | 39 893 | 73 679 | 1.8 |

Rekord = jeden tydzień zagregowanych (średnia) wyników laboratoryjnych danego pacjenta; klucz: `{patient_id}-{rok}-W{tydzień}`.

Liczby dotyczą plików surowych. Po czyszczeniu: 78 197 rekordów, 40 924 pacjentów (1 823 pozytywnych) — patrz `05_REALIZACJA_DO_SGKF.md`.

## Cechy

Morfologia (HGB, RBC, WBC, PLT, MCV, MCH, MCHC, HCT, RDW), biochemia (KREA, BUN, Na, K, GLU, ALT, AST), koagulacja (PT, INR, APTT), CRP, eGFR (MDRD, CKD-EPI).

Dla każdego parametru: wartość + flaga normy (`-1` / `0` / `1`).


## Struktura i dokumenty

| Etap | Katalog / plik |
|---|---|
| Przygotowanie danych | `1-data-preparation/` |
| Imputacja | `2-imputation/` (`RAPORT_IMPUTACJA.md`) |
| Podział na foldy (zamrożony) | `3-sgkf-split/` (`RAPORT_SGKF_MICE.md`, `results/pacjent_fold.csv`) |
| Etap 0 (diagnostyka danych), co dalej | `4-pu-setup/` |
| **Realizacja etapów 1–3, podział pracy** | `05_REALIZACJA_DO_SGKF.md` |
| Synteza: problem, problemy danych, decyzje | `04_PODSUMOWANIE_STANU_PROJEKTU.md` |
| Wstępne plany modelowania (odniesienie) | branch `pu-pipeline-1-5-lk` |
