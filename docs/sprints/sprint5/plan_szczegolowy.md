# Sprint 5: plan szczegółowy

**Status:** propozycje do omówienia w zespole i z prowadzącym. Zadania z uzasadnieniem są w `README.md` i na GitHubie (Sprint 5, issues #29–#51).
**Na podstawie:** uwag prowadzącego (`uwagi_prowadzacego.md`), notatek Liwii z OneNote (sekcja 3) i podziału ról w zespole.

---

## 1. Cel sprintu

W tym tygodniu **nie trenujemy jeszcze modeli „na wynik”**. Przygotowujemy wszystko, żeby pierwsze modele dały wynik, któremu da się zaufać:
- jeden zapisany sposób oceny,
- odłożony zbiór testowy,
- gotowe dane w foldach,
- jedna funkcja oceny dla wszystkich.

**Na koniec sprintu chcemy mieć:**
- [ ] zasady oceny modeli przyjęte przez zespół i wysłane prowadzącemu (#29–#33),
- [ ] fold 4 odłożony i chroniony w kodzie (#38),
- [ ] foldy robocze po imputacji, której imputer nie widział foldu 4 (#47),
- [ ] wspólną funkcję oceny i stałe listy ukrytych chorych (#33, #39),
- [ ] podział odtworzony na drugim komputerze (#36),
- [ ] skrypty do liczb, które dotąd były liczone jednorazowo (#37, #41–#44).

---

## 2. Proponowane decyzje

To propozycje z uzasadnieniem. Jeśli zespół albo prowadzący zdecyduje inaczej, zmieniamy zapisane zasady, a nie po cichu kod.

| # | Sprawa | Propozycja | Dlaczego |
|---|---|---|---|
| D1 | zbiór testowy | **fold 4 odkładamy do samego końca** (8 184 pacjentów, 364 chorych). Foldy 0–3 to zbiór roboczy (32 740 pacjentów, 1 459 chorych) | prosi o to prowadzący („wyodrębnić zbiór testowy”) i Liwia („testowy baaardzo z boku”). Korzystamy z istniejącego, zamrożonego podziału, więc nic nie przeliczamy. Fold wybrany regułą (ostatni), a nie po wynikach |
| D2 | decyzje oparte na danych | **liczymy je tylko na foldach 0–3** (np. które cechy odrzucić, jaki próg braków) | inaczej fold 4 przestaje być niezależny |
| D3 | strojenie | walidacja krzyżowa na 4 foldach roboczych: w każdym foldzie jeden model, **te same hiperparametry**; Optuna optymalizuje średnią z foldów, odchylenie raportujemy. Potem trening na całych foldach 0–3 i **jedna** ocena na foldzie 4 | schemat z notatek Liwii. Prostszy niż walidacja zagnieżdżona z `PLAN_MODELOWANIA.md` §15: jeden zestaw ustawień i jeden model końcowy |
| D4 | jednostka | jeden profil na pacjenta, np. mediana cech po imputacji (#30) | każdy pacjent waży tyle samo; mediana najsłabiej wiąże się z liczbą badań |
| D5 | metryka | główna: `RecallHidden@5%`; dodatkowo @1% i @10% oraz AUC. Ukrywamy 40% chorych, ziarna 0–4 (#29) | mówi, czy model wypycha ukrytych chorych na górę listy. Ustalona przed zobaczeniem wyników |
| D6 | proporcje w teście | naturalne (około 4,5% chorych); wersja 50/50 tylko jako uzupełnienie | 50/50 zmienia sens `RecallHidden@q` (mniejsza pula KOR); do potwierdzenia z prowadzącym |
| D7 | modele kontrolne | losowy ranking, wiek i płeć, sam wzorzec braków, same nazwy badań (#31) | wyznaczają poprzeczkę: sam wzorzec braków odzyskuje już 52% ukrytych |
| D8 | pierwsze modele | regresja logistyczna z regularyzacją, PU Bagging, jeden model drzewiasty (LightGBM albo XGBoost) (#32) | wystarczą do pierwszego uczciwego porównania; TabNet i AutoGluon później (sekcja 4) |
| D9 | imputacja | MICE bez zmian (ExtraTrees 7/78/10), bez strojenia | sprawdzone w etapach 2–3; strojenie imputera nie jest celem projektu |

**Konsekwencje D1:**
- Gotowe `foldy_imputowane_38/` z etapu 3 **nie nadają się do modelowania**. Imputer dla foldów 0–3 był dopasowany także na pacjentach foldu 4 (bez etykiet, ale jednak). Potrzebne są nowe foldy (#47).
- Na foldzie 4 będzie około 146 ukrytych chorych (40% z 364). Przy wyniku w okolicy 50% niepewność pojedynczego losowania to około ±8 pp. Dlatego obok wyniku na foldzie 4 podajemy wyniki walidacji krzyżowej.
- Diagnostyka z etapu 3 korzystała ze wszystkich foldów, ale nie wybierano na niej żadnego modelu ani ustawień. Zapisujemy to jako ograniczenie.

---

## 3. Notatki Liwii: co z czym robimy

Przepisane ze zrzutu OneNote. Fragmenty o niepewnym odczycie są oznaczone „(?)”.

| Punkt z notatek | Co z tym robimy | Zadanie |
|---|---|---|
| **Nie mamy danych z grupy kontrolnej → może centrum krwiodawstwa, badania z SOR** | nie w tym sprincie: wymaga nowych danych i zgód. Krwiodawcy też nie mieli obrazowania i są zdrowsi od pacjentów szpitalnych, więc model nauczyłby się „szpital kontra zdrowi”, czyli nowego skrótu. Najcenniejsza byłaby grupa **z obrazowaniem bez tętniaka** (np. po angio-TK) | #34 (pytanie do właściciela danych) |
| **Jak zmieniają się parametry u podobnych pacjentów na przestrzeni lat (fluktuacje)** | tak. Okres badania widać w danych samych chorych (AUC 0,83). Komentarze laboratorium mówią o zmianach norm: morfologia od 17.04.2015, kwas moczowy od 05.06.2015. Nazwy badań zmieniały się w 2013 i 2020–2021 | #45 |
| **Sprawdzić, czy nie ma wartości ujemnych** | w danych wejściowych do podziału nie ma żadnych (sprawdzone jednorazowo 22.11), a imputacja sprawdza to w każdym foldzie. Brakuje skryptu na danych surowych i przetworzonych | #43 |
| **Podział na train/val/test** | D1: fold 4 = test, foldy 0–3 = trening i walidacja | #38 |
| **10/5 foldów, stratified group k fold, standaryzacja per fold** | zostajemy przy 5 foldach (sekcja 4). Standaryzacja w foldzie jest zrobiona dla imputacji; dla modeli potrzebny jest osobny krok dopasowany tylko na treningu foldu | #51 |
| **Imputacja w foldach niezależnie na zbiorze testowym** | tak jest: imputer dopasowany na treningu, test tylko przekształcany. Osobnego imputera na samym teście nie dopasowujemy (to byłby wyciek i nie da się tak zrobić w praktyce). Dla D1 liczymy foldy od nowa | #47 |
| **Skrośna (?) cross validation** | D3 | #29 |
| **Zamiana etykiet (tętniaki wrzucamy do unlabeled)** | ukrywanie 40% chorych (D5), te same listy ukrytych dla wszystkich modeli | #39, #33 |
| **Jak pacjent wystąpi raz, to klasyfikujemy do badania? → dla unlabeled** | pacjenci z jednym rekordem zostają (decyzja z 23.04). Mają go jednak 60% pacjentów KOR i tylko 33% NEURO, a profil z jednego badania jest bardziej „rozrzucony” niż mediana z kilku, co model może wykorzystać. Sprawdzamy wyniki według liczby rekordów | #30, #40 |
| **Parametr odcięcia jako parametr → zmienna do modelowania** | do dopytania, o który próg chodzi. Jeśli o próg na liście (q): wartości (1/5/10%) ustalamy z góry i nie dobieramy ich na teście. Jeśli o próg braków (60%): to warianty zestawu cech | #29, #49 |
| **Modelowanie XGBoost, LGBM, TabNet, AutoGluon?** | D8: drzewa (LightGBM/XGBoost) tak, TabNet i AutoGluon później (sekcja 4) | #32 |
| **Kilka różnych możliwości dla positive unlabeled** | przegląd metod PU i wybór na start | #32 |
| **Obniżamy liczebność klasy pozytywnej; wybieramy k% pacjentów, ale musimy znać ilości; zmieniamy wtedy etykietę; żeby odzyskiwać dużo z poprawnymi etykietami** | to jest ukrywanie (D5). Przy 40%: w zbiorze roboczym 1 459 chorych, czyli około 584 ukrytych. W każdym podziale walidacji krzyżowej około 438 z nich jest w treningu, a około 146 w walidacji; w foldzie 4 około 146 | #29, #39 |
| **Odrzucić parametry, które mają dużą różnicę KOR vs NEURO w brakach danych (przynajmniej zbliżony odsetek)** | dobry wariant, zbieżny z uwagą prowadzącego o „podobnych brakach”. Na foldach 0–3 aż 9 cech różni się o ponad 20 pp: NRBC, %NRBC, NEUT, %NEUT, CRP, eGFR-MDRD, eGFRCKD, IG, %IG (liczone jednorazowo). Nawet bez nich morfologii brakuje w 12% rekordów NEURO i 2,5% KOR, więc skrót słabnie, ale nie znika | #49 |
| **Optymalizacja hiperparametrów: w foldzie 1 model, model per fold, te same hiperparametry, Optuna zwraca średnią metrykę z foldów, „jak najmniejsze (?) metryki dla x foldów”** | D3. Ostatni punkt rozumiemy jako stabilność: odchylenie między foldami raportujemy, a przy podobnej średniej wybieramy ustawienia stabilniejsze | #29 |
| **Inne zbiory (testowy baaardzo z boku)** | D1 i D2 | #38 |
| **Trenujemy model na wszystkich danych dla tego jednego modelu i ostatni test na zbiorze testowym** | D3: trening końcowy na foldach 0–3, jedna ocena na foldzie 4 | #29 |
| **Należy dobrze dobrać metrykę w Optunie** | definicja `RecallHidden@q` z ręcznie policzonym przykładem | #29, #33 |
| **Test set out (z rozkładem normalnym)** | rozumiemy jako „z naturalnymi proporcjami chorych” (D6); do potwierdzenia z Liwią | #29 |
| **Check, jakich modeli chcielibyśmy użyć** | przegląd modeli | #32 |
| **?PUBLIKACJA: T1–T4** | niejasne, do dopytania; nie w tym sprincie | — |

---

## 4. Czego nie robimy teraz (i dlaczego)

| Sprawa | Dlaczego nie teraz |
|---|---|
| grupa kontrolna (krwiodawcy, SOR) | nie mamy takich danych, a pozyskanie wymaga zgód. Krwiodawcy wprowadziliby nowy skrót „zdrowi kontra szpital”. Pytamy właściciela danych o pacjentów z obrazowaniem bez tętniaka |
| informacja o pęknięciu tętniaka | w danych jej nie widać (`diagnosis_id` puste); potwierdzenie w #42, pytanie do właściciela danych w #34 |
| TabNet | sieć do danych tabelarycznych, wymaga dużo danych i strojenia; 1 459 chorych w zbiorze roboczym to mało, a wynik trudno wyjaśnić |
| AutoGluon | sam składa wiele modeli i sam robi walidację krzyżową, która nie zna naszych pacjentów ani list ukrytych; łatwo o wyciek. Ewentualnie później jako „sufit” na profilach pacjentów |
| 10 foldów | obecne 5 foldów jest zamrożonych i sprawdzonych; zmiana oznaczałaby przeliczenie całego etapu 3 |
| walidacja zagnieżdżona z `PLAN_MODELOWANIA.md` §15 | zastępuje ją D1 + D3; wracamy do niej, jeśli wynik na foldzie 4 będzie wątpliwy |
| strojenie MICE, nowy benchmark imputacji | MICE bez zmian (D9) |
| dobieranie q albo progu na foldzie 4 | q ustalone z góry; pokazujemy wyniki dla kilku wartości |
| merge do `master` | po sprincie, gdy zespół przejrzy zmiany |
| publikacja („T1–T4”) | najpierw wynik; dopytać, co znaczy zapis |

---

## 5. Pytania (#34)

**Do prowadzącego:**
1. Czy fold 4 jako odłożony zbiór testowy (D1) i walidacja krzyżowa na pozostałych (D3) to to, o co chodziło?
2. „Po równo z tętniakiem i bez”: naturalne proporcje w teście czy 50/50?
3. Model na samych nazwach badań: czy chodziło o nazwy zleconych badań (`examination_type`)?
4. Optuna: wybór ustawień jednego modelu czy także wybór rodzaju modelu?
5. Limity historii 11/38 rekordów z etapu 1: zostawiamy jako ograniczenie czy ujednolicamy?
6. Co oznacza zapis „T1–T4” przy publikacji (jeśli pochodzi od prowadzącego)?

**Do właściciela danych:**
1. Czy można dosłać rozpoznanie pacjentów NEURO, np. kod ICD-10 (I60 = krwotok podpajęczynówkowy, czyli tętniak pęknięty; I67.1 = tętniak niepęknięty)?
2. Czy istnieje grupa pacjentów z obrazowaniem (angio-TK/MR) bez tętniaka?
3. Kiedy wykonano badania NEURO względem rozpoznania i leczenia; czy są daty rozpoznania?
4. Skąd pochodzi kohorta KOR (oddział, kryteria)?
5. Czy w systemie laboratoryjnym zmieniały się nazwy badań i normy (2013, 2015, 2020–2021)?

---

## 6. Zależności między zadaniami

- Zasady oceny (#29) korzystają z wariantów cech (#49) i porównania foldów (#40).
- Nowa imputacja (#47) korzysta z funkcji foldów roboczych (#38), ale może ruszyć wcześniej według tej samej reguły.
- Wspólna funkcja oceny (#33) korzysta ze stałych list ukrytych (#39).
- Model kontrolny „same nazwy badań” (#31) potrzebuje wyniku #41.

**Po sprincie:** pierwszy przebieg na foldach 0–3, czyli modele kontrolne, regresja logistyczna, PU Bagging i model drzewiasty. Potem ocena przy podobnych brakach (#50), strojenie Optuną (D3), a na końcu jedna ocena na foldzie 4.

---

## 7. Zasady wspólne

1. **Fold 4 jest zamknięty.** Czytają go tylko testy (#37, #38) i końcowa ocena.
2. **Decyzje oparte na danych podejmujemy tylko na foldach 0–3** (D2).
3. **Każda liczba do raportu ma skrypt w repo**, plik wyników (JSON/CSV) i log. Liczby policzone „na szybko” oznaczamy jako jednorazowe, dopóki nie mają skryptu.
4. **W każdym pliku wyników zapisujemy wersje bibliotek.**
5. **Nikt nie tworzy własnego podziału, imputacji ani metryki** w skrypcie modelu: wszyscy używają `wczytaj_podzial()`, foldów z #47 i funkcji oceny z #33.
6. **Każdą zmianę w kodzie przegląda druga osoba** przed scaleniem.
7. **Kod sprawdzamy na danych syntetycznych** (`data_syntetyczne/`), a wyników z nich nie commitujemy i nie raportujemy.
