# #29 Zasady oceny modeli (szkic na spotkanie 26.11)

**Zadanie:** ustalić, jak oceniamy modele, zanim ktokolwiek zacznie trenować. Każdy punkt ma kod w `modelowanie/`, sprawdzony na danych syntetycznych.

**Status:** propozycja do przyjęcia przez zespół. Punkty oznaczone „do potwierdzenia” czekają na prowadzącego (#34) albo na zadania zespołu.

## Zasady

| # | Zasada | Kod | Stan |
|---|---|---|---|
| 1 | **Fold 4 to zbiór testowy.** Pracujemy na 4 układach roboczych: 3 foldy uczą, czwarty sprawdza. Fold 4 dopiero raz na końcu (D1, D3) | `dane.uklady()` | do podmiany na #38 |
| 2 | **Ukrywamy 40% chorych**, osobno w każdym foldzie (równy udział), 5 list (ziarna 0–4). Ukryci w treningu mają etykietę 0 (D5) | `ukrywanie.listy_ukrytych()` | do podmiany na #39 |
| 3 | **Jednostka = pacjent**, profil = mediana cech (#30) | `profil.profil()` | propozycja |
| 4 | **Miara główna: RecallHidden@5%**, dodatkowo @1%, @10% i AUC. Remisy jako wartość oczekiwana. q nie dobieramy do wyniku (#33) | `ocena.ocena()` | gotowe |
| 5 | **Każdy wynik obok modeli kontrolnych**: losowy, wiek i płeć, liczba badań, okres badań, wzorzec braków, nazwy badań (#31). Podajemy przewagę nad każdą kontrolą | `kontrolne.py`, `uruchom_ocene.py` | bez „nazw badań” (#41) |
| 6 | **Ocena bez skrótów, obowiązkowo obok wyniku głównego:** w grupach liczby badań, we wspólnym oknie dat, a po #50 w grupach podobnych braków (o to prosił prowadzący) | `ocena.ocena_w_grupach()` | dwa z trzech gotowe |
| 7 | **Średnia i odchylenie z 20 przebiegów** (4 układy × 5 list), nigdy jeden przebieg | `uruchom_ocene.py` | gotowe |
| 8 | **Proporcje w teście naturalne**, wersja 50/50 tylko jako uzupełnienie (D6) | — | do potwierdzenia (#34) |
| 9 | **Strojenie:** Optuna na 4 układach roboczych, te same hiperparametry, cel = średnie RecallHidden@5% (D3) | — | Sprint 7 |

## Dlaczego tak, a nie prościej

Ocena PU bez potwierdzonych zdrowych jest trudna (szczegóły: `33_funkcja_oceny.md`). Ukryci chorzy mają wszystkie cechy kohorty NEURO, więc sama wysoka liczba niczego nie dowodzi. Na danych syntetycznych widać to wprost (`31_modele_kontrolne.md`):
- sam **okres badań** daje RecallHidden@5% około 0,91, więcej niż jakikolwiek model na wynikach badań,
- w trybie `brak` (etykieta losowa) model i tak wychodzi trochę ponad losowanie, bo **czyszczenie osobno dla grup** zostawiło w NEURO pacjentów z bardzo małą liczbą badań.

Dlatego wynik bez kontroli i bez oceny w grupach nie jest wynikiem.

## Co jest nowe względem planu modelowania (`4-pu-setup/PLAN_MODELOWANIA.md`)

- fold 4 jako zbiór testowy zamiast walidacji zagnieżdżonej,
- nowe kontrole: liczba badań, okres badań i (po #41) nazwy badań,
- ocena w grupach i we wspólnym oknie jako obowiązkowa część wyniku, a nie tylko analiza wrażliwości,
- remisy liczone jako wartość oczekiwana zamiast rozstrzygania po numerze pacjenta,
- testy kodu na danych syntetycznych w trybach `brak` i `wbudowany`.

## Pytania na spotkanie

1. Czy przyjmujemy zasady 1–7?
2. Okres badań odróżnia grupy prawie idealnie. Czy główny wynik ma być liczony we wspólnym oknie dat, mimo że na prawdziwych danych jest tam tylko około 13 ukrytych na fold?
3. 50/50 w teście: tylko jako uzupełnienie (czekamy na prowadzącego)?
