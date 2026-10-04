# 4-pu-setup: etap 0 i co dalej

Katalog zbiera weryfikację danych wykonaną po raporcie przejściowym (etap 0) i jest miejscem na przyszłe przygotowanie modelowania Positive-Unlabeled. **Plan modelowania zostanie ustalony na nowo po spotkaniu z prowadzącym.**

## Co tu jest

| Plik | Zawartość |
|---|---|
| `ETAP0_USTALENIA.md` | wyniki diagnostyki danych: rozjazd czasowy kohort, 63 pacjentów w obu kohortach, reguła agregacji, wartości skrajne |
| `PYTANIA_DO_PROWADZACEGO.md` | kwestie do potwierdzenia na spotkaniu |
| `etap0_diagnostyka.py` | diagnostyka do etapu 0 |
| `etap0_agregacja.py`, `results/etap0_agregacja_*.csv` | porównanie reguł agregacji rekordów do pacjenta (punkt 0.3) |

Diagnostyka podziału (punkt 0.6) jest w `3-sgkf-split/`, razem z zamrożonym podziałem.

## Co dalej, w ogólnym zarysie

1. **Spotkanie: decyzje otwarte** (`04_PODSUMOWANIE_STANU_PROJEKTU.md`, Część IV):
   - kontrola rozjazdu czasowego kohort (0.2),
   - pochodzenie wyników NEURO: przed rozpoznaniem czy z hospitalizacji (0.2b),
   - jednostki kreatyniny i wartości niemożliwe (0.5),
   - status 63 pacjentów mieszanych (0.1).
2. **Nowy plan modelowania**, uwzględniający te decyzje. Niezależnie od ich wyniku obowiązują ustalenia, które już mamy:
   - foldy wyłącznie z `3-sgkf-split/results/pacjent_fold.csv` (`podzial.wczytaj_podzial()`), bez liczenia podziału od nowa,
   - imputacja i każde inne dopasowanie wyłącznie na części treningowej foldu,
   - ocena na poziomie pacjenta, nie rekordu,
   - brak grupy kontrolnej oznacza, że ocena opiera się na kontrolowanym ukrywaniu znanych pozytywnych (uwaga prowadzącego), a nie na klasycznych metrykach P-vs-N.
3. **Implementacja** według nowego planu.

## Odniesienie

Wstępny plan z września (`02_PLAN_PRZED_MODELOWANIEM.md`, `03_PLAN_MODELOWANIE.md`) i implementacja punktów 1–5 (foldy z walidacją zagnieżdżoną, ukrywanie pozytywnych, metryki pacjentowe, funkcja celu, 17 testów) są zachowane na branchu **`pu-pipeline-1-5-lk`**. Służą jako materiał do nowego planu, nie jako plan obowiązujący.
