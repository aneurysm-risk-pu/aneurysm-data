# Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled

**Zespołowy Projekt Badawczy 2026**

## Dane

| Plik | Zbiór | Pacjenci | Rekordy | Śr. rekordów/pacjent |
|------|-------|---------|---------|----------------------|
| `data/raw/neuro_merged_aggregated_1W_mean.csv` | **Positive** (pacjenci z tętniakiem) | 2 016 | 7 655 | 3.8 |
| `data/raw/kor_merged_aggregated_1W_mean.csv` | **Unlabeled** (pacjenci ogólnie) | 39 893 | 73 679 | 1.8 |

Rekord = jeden tydzień zagregowanych (średnia) wyników laboratoryjnych danego pacjenta; klucz: `{patient_id}-{rok}-W{tydzień}`.

Liczby dotyczą plików surowych. Po czyszczeniu: 78 197 rekordów, 40 924 pacjentów (1 823 pozytywnych) — patrz `1026_REALIZACJA_DO_SGKF.md`.

## Cechy

Morfologia (HGB, RBC, WBC, PLT, MCV, MCH, MCHC, HCT, RDW), biochemia (KREA, BUN, Na, K, GLU, ALT, AST), koagulacja (PT, INR, APTT), CRP, eGFR (MDRD, CKD-EPI).

Dla każdego parametru: wartość + flaga normy (`-1` / `0` / `1`).


## Struktura i dokumenty

| Etap | Katalog / plik |
|---|---|
| Przygotowanie danych | `1-data-preparation/` |
| Imputacja | `2-imputation/` (`RAPORT_IMPUTACJA.md`) |
| Podział na foldy (zamrożony) | `3-sgkf-split/` (`RAPORT_SGKF_MICE.md`, `results/pacjent_fold.csv`) |
| Ustalenia z danych i plan modelowania | `4-pu-setup/` |
| Materiał do raportu przejściowego (12.06) | `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` |
| **Realizacja etapów 1–3, podział pracy** | `1026_REALIZACJA_DO_SGKF.md` |
| Problemy wykryte w danych, pytania do prowadzącego | `4-pu-setup/USTALENIA_DANYCH.md` |
| Propozycja planu modelowania | `4-pu-setup/PLAN_MODELOWANIA.md` |
