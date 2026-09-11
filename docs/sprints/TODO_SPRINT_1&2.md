# TODO — Sprint 1

## Analiza wstępna danych (sprint 1 dla każdego)

## Dane — czyszczenie i walidacja

- [X] Usunąć kolumny z formatem `{'values': [...]}` 
- [X] Sprawdzić 1 duplikat (usunięty) `custom_id = '194143-2017-W52'`


- [ ] Skonsultować i ustalić co zrobić (DATA_ANALYSIS_SUMMARY.md):
  - WBC: 3 rekordy > 100 G/l 
  - KREA: wartości > 20 mg/dl 
  - Na < 100 i K > 9

### Na później: dotyczące niespójności między zbiorami KOR i NEURO
- Obsłużyć listę 358 powtarzających się pacjentów
- Ujednolicić kolumnę płci (KOR: `"0.0"`/`"1.0"` vs NEURO: `"0"`/`"1"`)
- pacjent PLT=0
- błędny wiek - pacjent miał 222 lata

## Analiza statystyczna (przed imputacją)

- [ ] Testy statystyczne porównujące rozkłady P vs U:
  - Mann-Whitney U
  - Kolmogorov-Smirnov
  - t-test Welcha
- [ ] Sprawdzić, czy zmiany tygodniowe są duże — ocena zasadności agregacji dwutygodniowej zamiast tygodniowej *(pytanie do konsultacji)*

## Podział danych

- [ ] Ustalić strategię podziału train/test (podział na poziomie `patient_id`)
- [ ] Zweryfikować, czy rekord pacjenta w NEURO zawsze kończy się na diagnozie

## Imputacja (po podziale train/test)

- [ ] KNN — *Liwia*
- [ ] MICE — *Lukasz*
- inna metoda *Adrian/Dominika*
- [ ] Porównanie metod imputacji — metryka Kullbacka-Leiblera lub Cosine Similarity *(pytanie jaka do sprawdznia)*


