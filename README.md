# Ocena ryzyka tętniaka mózgu — modelowanie Positive-Unlabeled

**Zespołowy Projekt Badawczy 2026**

## Dane

| Plik | Zbiór | Pacjenci | Rekordy | Śr. rekordów/pacjent |
|------|-------|---------|---------|----------------------|
| `data/raw/neuro_merged_aggregated_1W_mean.csv` | **Positive** (pacjenci z tętniakiem) | 2 016 | 7 655 | 3.8 |
| `data/raw/kor_merged_aggregated_1W_mean.csv` | **Unlabeled** (pacjenci ogólnie) | 39 893 | 73 679 | 1.8 |

Rekord = jeden tydzień zagregowanych (średnia) wyników laboratoryjnych danego pacjenta; klucz: `{patient_id}-{rok}-W{tydzień}`.

Liczby dotyczą plików surowych. Po czyszczeniu (etap 1) i przygotowaniu danych do podziału (etap 3, decyzje z 04.10.2026): **78 052 rekordy, 40 924 pacjentów, 1 823 pozytywnych**. Szczegóły: `1026_REALIZACJA_DO_SGKF.md`.

**Wejście do podziału i modelowania:** `data/processed/aneurysm_sgkf_input.csv` (38 cech, przed imputacją). Imputacja odbywa się wewnątrz foldów.

## Cechy (38)

| Grupa | Cechy |
|---|---|
| morfologia | HGB, RBC, HCT, MCV, MCH, MCHC, RDW, PLT, MPV, WPT |
| rozmaz | WBC, NEUT, %NEUT, LYMPH, %LYMPH, MONO, %MONO, EO, %EO, BAZO, %BAZO, IG, %IG, NRBC, %NRBC |
| biochemia | KREA, eGFR-MDRD, eGFRCKD, Na, K, GLU, CRP |
| koagulologia | PT, INR, APTT, WAPTT |
| demografia | patient_age, patient_sex |

Wariant 35 cech (analiza wrażliwości) to te same dane bez CRP, MONO i %MONO. Kolumny meta (nie są cechami): `patient_id`, `custom_id`, `examination_date`, `label`, `pacjent_mieszany`.

## Struktura i dokumenty

| Etap | Katalog / plik |
|---|---|
| 1. Przygotowanie danych | `1-data-preparation/` |
| 2. Imputacja (benchmark metod) | `2-imputation/` (`RAPORT_IMPUTACJA.md`) |
| 3. Przygotowanie danych do SGKF (A) i podział z imputacją w foldach (B) | `3-sgkf-split/` (`RAPORT_SGKF_MICE.md`; całość: `python 3-sgkf-split/uruchom_sgkf.py`) |
| 4. Plan modelowania PU | `4-pu-setup/PLAN_MODELOWANIA.md` |
| Materiał do raportu przejściowego (06.2026, historyczny) | `0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md` |
| **Realizacja etapów 1–3, podział pracy, decyzje** (10.2026) | `1026_REALIZACJA_DO_SGKF.md` |
| Archiwum sprintów i raportów cząstkowych | `docs/` |

## Środowisko

```bash
python3 -m venv .venv && .venv/bin/pip install scikit-learn pandas numpy scipy pyarrow
.venv/bin/python 3-sgkf-split/uruchom_sgkf.py --bez-mice   # szybkie sprawdzenie etapu 3 (~20 s)
```
