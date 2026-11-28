# #30 Profil pacjenta: wiele badań i jedno badanie

**Zadanie:** wybrać, jak łączymy rekordy pacjenta w jeden profil, i jak kontrolujemy to, że KOR częściej ma tylko jedno badanie.

## Propozycja

1. **Profil = mediana każdej cechy** z rekordów pacjenta; braki zostają brakami. Model drzewiasty obsłuży je sam, a dla regresji logistycznej użyjemy imputacji Dominiki (#47). Kod: `modelowanie/profil.py`.
2. **Liczbę badań kontrolujemy na trzy sposoby:**
   - model kontrolny „liczba badań” (#31): ile wyniku daje sama liczba rekordów,
   - ocena w grupach liczby badań (1 / 2 / 3–4 / 5+): ukrytych porównujemy tylko z KOR z tej samej grupy (`ocena_w_grupach()` w `ocena.py`),
   - analiza wrażliwości: profil tylko z pierwszego badania (`profil(..., "pierwszy_rekord")`).

## Dlaczego mediana

Analiza na prawdziwych danych była już zrobiona w repo zespołu (`4-pu-setup/etap0_agregacja.py`, wyniki w `4-pu-setup/results/`). Sprawdza, czy reguła łączenia „przemyca” liczbę badań: korelację cechy z liczbą rekordów, tylko wewnątrz KOR.

| Reguła | Cechy z korelacją ponad 0,2 z liczbą badań | Pacjenci z K poza zakresem 1,5–9 |
|---|---:|---:|
| **mediana** | **2** | **10** |
| średnia | 7 | 9 |
| maksimum | 19 | 57 |
| ostatnia wartość | 1 | 23 |

Maksimum wyraźnie przemyca liczbę badań: im więcej pomiarów, tym wyższe maksimum. Ostatnia wartość przemyca ją równie mało jak mediana, ale opiera się na jednym pomiarze i więcej skrajnych wartości przechodzi. Mediana jest najbezpieczniejsza.

## Na co uważać

- **Jedno badanie ma 60% pacjentów KOR i 33% NEURO.** Profil z jednego badania jest bardziej rozrzucony niż mediana z kilku, a model może to wykorzystać nawet bez kolumny „liczba badań”. Dlatego wynik zawsze pokazujemy też w grupach liczby badań.
- **Mediana miesza okresy.** Historia pacjenta NEURO może obejmować lata, także po leczeniu. Bez daty rozpoznania (pytanie do właściciela danych, #34) nie wybierzemy badań „sprzed rozpoznania”. Najbliższym przybliżeniem jest wariant „pierwsze badanie”.
- **Dane syntetyczne nie mają trendów w czasie,** więc różnice między regułami wychodzą na nich mniejsze niż na prawdziwych. Decyzję opieramy na analizie z repo zespołu.

## Do sprawdzenia po zadaniach zespołu

- #40 (Adrian): czy udział pacjentów z jednym badaniem jest podobny w każdym foldzie.
- #34: czy właściciel danych poda daty rozpoznania. Jeśli tak, wracamy do wyboru badań wchodzących do profilu.
