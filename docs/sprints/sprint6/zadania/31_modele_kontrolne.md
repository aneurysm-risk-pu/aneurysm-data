# #31 Modele kontrolne

**Zadanie:** ustalić modele „ślepe”, z którymi porównujemy każdy właściwy model.

## Co to jest model kontrolny

To model, który dostaje tylko informację, która **nie** jest chorobą, np. ile badań miał pacjent albo z jakich lat są jego badania. Jeśli już on „znajduje” ukrytych chorych, to wynik da się osiągnąć skrótem. Działa to jak grupa placebo w badaniu leku: lek musi być lepszy od placebo, a nie od zera. Nasz model musi być wyraźnie lepszy od kontroli.

## Co zrobione

`modelowanie/kontrolne.py` i `modelowanie/uruchom_ocene.py`. Każda kontrola i model odniesienia są uczone tym samym modelem (HistGradientBoosting, ustawienia z etapu 3) na 4 układach roboczych × 5 list ukrytych = 20 przebiegów.

| Kontrola | Co widzi | Skąd pomysł |
|---|---|---|
| losowy | nic | sprawdza samą metrykę (powinno wyjść ok. 5%) |
| wiek i płeć | demografię | plan modelowania |
| wzorzec braków | które badania wykonano | plan modelowania |
| **liczba badań** | ile rekordów ma pacjent | nowe: KOR 60% z jednym badaniem, NEURO 33% |
| **okres badań** | datę pierwszego i ostatniego badania | nowe: KOR 2019–2021, NEURO 2000–2024 |
| nazwy badań | nazwy zleconych badań | pytanie prowadzącego; **czeka na #41** (Liwia) |

Model odniesienia (nie kontrola): 38 cech, mediana na pacjenta, braki bez imputacji (#30).

## Wyniki na danych syntetycznych (tylko sprawdzenie kodu)

RecallHidden@5%, średnio z 20 przebiegów; losowo oczekujemy 0,05.

| Model | `brak` | `wbudowany` | `realny` | `realny`, w grupach liczby badań | `realny`, tylko wspólne okno dat |
|---|---:|---:|---:|---:|---:|
| losowy | 0,053 | 0,053 | 0,061 | 0,062 | 0,052 |
| wiek i płeć | 0,047 | 0,047 | 0,116 | 0,120 | 0,138 |
| liczba badań | 0,052 | 0,052 | 0,197 | 0,100 | 0,144 |
| okres badań | 0,048 | 0,048 | **0,913** | 0,853 | 0,350 |
| wzorzec braków | 0,068 | 0,068 | 0,711 | 0,658 | 0,716 |
| **model odniesienia** | 0,074 | **0,178** | 0,537 | 0,491 | 0,528 |

Jak to czytać:
- **`brak`** (etykieta niezależna od danych): kontrole dają około 0,05, czyli kod nie ma wycieku. Model i wzorzec braków są minimalnie wyżej (0,074 i 0,068). Powód znaleźliśmy: uproszczony etap 1 w generatorze, tak jak prawdziwy etap 1, usuwa rzadko badanych pacjentów **tylko z KOR**. W danych zostaje więc około 1,4% pacjentów NEURO z bardzo małą liczbą badań, a w KOR takich nie ma. **Czyszczenie robione osobno dla grup samo tworzy skrót**, nawet gdy etykieta jest losowa. To ważne dla #46 (Liwia).
- **`wbudowany`** (znany sygnał w 6 cechach): model daje 0,178, czyli ponad 0,11 więcej niż najlepsza kontrola. Kod znajduje sygnał tam, gdzie on jest.
- **`realny`**: sam okres badań odróżnia grupy prawie idealnie (0,91). Model nie widzi dat, ale wartości badań mogą nieść informację o okresie (zmiany norm i aparatury, #45). We wspólnym oknie dat ta kontrola spada do 0,35.

Na danych syntetycznych model odniesienia wypada gorzej niż wzorzec braków (0,54 wobec 0,71). Na prawdziwych danych było odwrotnie: w etapie 3 (`3-sgkf-split/sgkf/results/kontrola_skrotow.json` w repo zespołu) wzorzec braków dał 0,56, a mediana 38 cech 0,69. Generator upraszcza wartości badań, więc **tych liczb nie interpretujemy**. Pokazują tylko, że kod działa.

## Na co uważać

- **Okres badań to najmocniejszy skrót.** Na prawdziwych danych jeszcze go nie liczyliśmy, ale na prawdziwych danych tylko ok. 9% chorych ma większość badań we wspólnym oknie dat (w syntetycznych 14%), więc spodziewamy się podobnie wysokiego wyniku. Wniosek dla #29: ocena we wspólnym oknie dat musi być obowiązkowa, choć jest mało precyzyjna: na prawdziwych danych to około 13 ukrytych na fold (na syntetycznych 21).
- **Przewagę liczymy nad każdą kontrolą osobno**, w tych samych przebiegach (`przewaga()` w `uruchom_ocene.py`).
- Kontrole muszą dostawać **te same pule i listy ukrytych** co model. Zapewnia to wspólna pętla w `uruchom_ocene.py`.

## Do sprawdzenia po zadaniach zespołu

- #41 (Liwia): dodać kontrolę „nazwy badań”.
- #49, #50 (Dominika): ocena w grupach o podobnych brakach, czyli wariant `ocena_w_grupach()` z grupą = wzorzec braków.
- #45 (Liwia): jeśli parametry wyraźnie zmieniają się w czasie, model może odtwarzać okres z wartości badań. Wtedy kluczowa jest ocena w oknie wspólnym.
- #46 (Liwia): czyszczenie osobno dla grup tworzy skrót (patrz tryb `brak`).
