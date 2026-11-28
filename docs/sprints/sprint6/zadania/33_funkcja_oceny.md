# #33 Wspólna funkcja oceny

**Zadanie:** jedna funkcja oceny dla wszystkich modeli, z testem na ręcznie policzonym przykładzie.

## Co zrobione

`modelowanie/ocena.py`:
- `recall_hidden(wynik, ukryci, q)`: jaka część ukrytych chorych trafia do górnych q% puli,
- `ocena(wynik, ukryci)`: RecallHidden@1%, @5%, @10% i AUC (ukryci kontra reszta puli),
- `ocena_w_grupach(wynik, ukryci, grupa)`: to samo liczone osobno w grupach, np. według liczby badań; ukrytych porównujemy tylko z KOR z tej samej grupy.

**Pula oceny** w układzie roboczym to pacjenci z foldu walidacyjnego, których model widzi jako nieoznaczonych: wszyscy KOR i ukryci chorzy. Widoczni chorzy z tego foldu nie wchodzą do puli.

**Przykład z testu:** pula 10 pacjentów A–J posortowanych od najwyższego wyniku, ukryci to B i E. Przy q = 20% górne 2 miejsca to A i B, więc RecallHidden = 1 z 2 = 0,5.

## Dlaczego RecallHidden i dlaczego ocena jest trudna

Nie mamy potwierdzonych zdrowych, więc dokładności, precyzji ani swoistości policzyć się nie da. RecallHidden to jedyna miara, która w naszych danych sprawdza, czy model wypycha znanych chorych na górę listy.

Jej słabość, o której najpewniej mówił prowadzący: ukryci chorzy pochodzą z NEURO, więc mają wszystkie cechy tej kohorty, takie jak inne lata badań, inne zestawy badań i więcej rekordów. Wysoki wynik może więc oznaczać „rozpoznaję pacjentów NEURO”, a nie „rozpoznaję tętniaka”. Dlatego:
1. wynik czytamy zawsze obok modeli kontrolnych (#31), bo liczy się przewaga nad nimi,
2. liczymy go też w grupach, w których dany skrót nie działa (ta sama liczba badań, wspólne okno dat, a później podobne braki z #50),
3. AUC jest pomocnicze.

Inne miary z literatury PU nie rozwiązują tego problemu. AUC P kontra U jest wiarygodne tylko wtedy, gdy znani chorzy są losową próbką wszystkich chorych, a u nas nie są. Miara Lee i Liu (recall² / odsetek przewidzianych pozytywnych) służy do wyboru progu, a nie do rankingu. Prawdziwą swoistość dałaby tylko grupa z obrazowaniem bez tętniaka, o którą pytamy właściciela danych (#34).

## Sprawdzenie

- 9 testów w `modelowanie/tests/test_modelowanie.py`: przykład ręczny, remisy, wynik stały daje q, kolejność wierszy bez znaczenia, brak ukrytych daje brak miary (nie zero), losowy ranking daje średnio q, ocena w grupach usuwa skrót grupy.
- Losowy ranking na prawdziwych pulach z danych syntetycznych, 1000 powtórzeń: RecallHidden@5% = 0,051 (oczekiwane 0,050).
- Tryby `brak` i `wbudowany`: wyniki w `docs/31_modele_kontrolne.md`.

## Na co uważać

- **Remisy.** Jeśli na granicy górnych q% kilku pacjentów ma ten sam wynik, liczymy wartość oczekiwaną przy losowym rozstrzygnięciu. Wcześniejszy kod (`kontrola_skrotow.py`) rozstrzygał remisy po numerze pacjenta, co przy modelach z dużą liczbą remisów (np. „liczba badań”) mogło faworyzować jedną kohortę.
- **Rozdzielczość.** W puli jest około 7 900 pacjentów i około 159 ukrytych, więc jeden ukryty to 0,6 pp. Między przebiegami wynik waha się o 2–5 pp, dlatego zawsze podajemy średnią i odchylenie z 20 przebiegów.
- **q ustalamy z góry** (główne 5%) i nie dobieramy go do wyniku.

## Do sprawdzenia po zadaniach zespołu

- #39 (Adrian): podmienić `listy_ukrytych()` na zamrożone listy i sprawdzić, że pule się nie zmieniają.
- #38 (Adrian): podmienić `uklady()` na funkcję zespołu.
