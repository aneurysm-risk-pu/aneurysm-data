# Sprint 6 (23–26.11.2026): weryfikacja przed modelowaniem

**Cel:** zanim zaczniemy modelowanie, sprawdzić to, o co pytał prowadzący, i ustalić zasady trenowania i oceny modeli. Wyniki każdej osoby omawiamy na spotkaniu w czwartek 26.11.

**Zadania na GitHubie:** kamień milowy „Sprint 6”, issues #29–#51, tablica „aneurysm-risk-pu PROJECT” (widok SPRINT).

## Jak pracujemy w tym sprincie

- **Priorytety na tablicy.** P0: bez tego nie zaczniemy modelowania (#29, #33, #34, #35, #36, #38, #39, #47). P1: odpowiedzi dla prowadzącego. P2: jeśli starczy czasu. Najpierw P0.
- **Każde zadanie ma „Gotowe, gdy”.** Zadanie zamykamy, kiedy ten warunek jest spełniony, a nie wtedy, gdy „coś zostało zrobione”.
- **Zależności.** Część zadań czeka na inne (na GitHubie „Blocked by”, w treści „Zależy od”): #47 ← #38, #39 ← #29, #33 ← #39, #31 ← #41, #29 ← #30 i #49. Jeśli Twoje zadanie blokuje kogoś, zrób je w pierwszej kolejności.
- **Przed spotkaniem w czwartek** każdy dopisuje w swoich zadaniach krótki komentarz: co wyszło, co blokuje, co przechodzi na następny sprint. Spotkanie idzie po tablicy, od P0.
- **Przegląd.** Zmianę w kodzie przegląda druga osoba, zanim trafi do wspólnej gałęzi.

**Pliki w tym folderze:**
- `README.md` (ten plik): zadania z uzasadnieniem, dla zespołu,
- `plan_szczegolowy.md`: proponowane decyzje (D1–D9), co zrobiliśmy z notatkami Liwii, czego nie robimy, pytania,
- `uwagi_prowadzacego.md`: uwagi prowadzącego do raportu z października wraz z faktami i liczbami.

## Wspólne ustalenia

1. **Fold 4 odkładamy jako zbiór testowy.** Prowadzący prosił o wyodrębnienie zbioru testowego. Każde zajrzenie do niego w trakcie pracy, nawet tylko „żeby zobaczyć wynik”, sprawia, że modele i decyzje zaczynają się do niego dopasowywać. Wtedy końcowy wynik jest zawyżony. Dlatego fold 4 zostaje nietknięty aż do końcowej oceny.
2. **Wszystkie analizy, decyzje i strojenie robimy na foldach 0–3.** Dotyczy to także analiz danych (np. wyboru cech), bo decyzja podjęta po obejrzeniu foldu 4 też jest wyciekiem.
3. **Każda liczba, którą pokazujemy prowadzącemu, ma skrypt w repo.** Część liczb z ostatnich tygodni pochodzi z jednorazowych sprawdzeń i nie da się ich odtworzyć.

## Gdzie co jest w repo

| Folder | Zawartość |
|---|---|
| `data/` | `raw/`: dane surowe KOR i NEURO; `imputation-inputs/aneurysm_concatted.csv`: wynik etapu 1; `processed/aneurysm_sgkf_input.csv`: dane wejściowe podziału i modelowania |
| `data_syntetyczne/` | dane syntetyczne w takim samym układzie jak `data/` (niżej) |
| `1-data-preparation/` | etap 1: notebooki Liwii (`data_analysis_v1.ipynb`, `scripts-lf/`), skrypty czyszczące (`scripts-lk/`), `cleaning_summary.md` |
| `2-imputation/` | etap 2: porównanie metod imputacji, `RAPORT_IMPUTACJA.md`, testy KS/KL (`final/validate_imputed.py`) |
| `3-sgkf-split/przygotowanie/` | etap 3, część A: ostatnie poprawki danych (`przygotuj_dane.py`) i analizy decyzji |
| `3-sgkf-split/sgkf/` | etap 3, część B: podział na foldy, imputacja w foldach, kontrole skrótów, testy |
| `3-sgkf-split/uruchom_sgkf.py` | uruchamia cały etap 3 po kolei |
| `4-pu-setup/` | plan i opcje modelowania (`PLAN_MODELOWANIA.md`, `OPCJE_MODELOWANIA.md`), pierwsze analizy pod modelowanie |
| `narzedzia/dane_syntetyczne/` | generator danych syntetycznych |
| `1026_REALIZACJA_DO_SGKF.md` | główny opis etapów 1–3: co zrobiono i dlaczego |

Cała praca jest na gałęzi `master` (od 23.11, PR #52). Przed rozpoczęciem pracy:

```bash
git checkout master
git pull
```

Kto pracował na starej `master`, zobaczy nową strukturę folderów (`data/`, `3-sgkf-split/`, `4-pu-setup/` itd.). Własne zmiany można robić na osobnej gałęzi od `master` i scalać przez PR, ale nie jest to obowiązkowe. Gałąź `pu-modeling-setup-lk` została usunięta po scaleniu. Kto ma ją lokalnie, może ją skasować: `git branch -D pu-modeling-setup-lk`.

## Dane syntetyczne

W folderze `data_syntetyczne/` są sztuczne dane w takim samym układzie jak `data/` (`raw/`, `imputation-inputs/`, `processed/`). Wszyscy mamy te same pliki. Służą do uruchamiania i sprawdzania kodu, także z pomocą AI, bez dostępu do danych pacjentów. Żeby skrypt pracował na nich, wystarczy podmienić w nim ścieżkę `data` na `data_syntetyczne`.

- Wyników policzonych na danych syntetycznych **nie commitujemy**. Nadpisują prawdziwe pliki w `results/`, w tym zamrożony `pacjent_fold.csv`. Po pracy przywracamy je poleceniem `git checkout -- 3-sgkf-split/`.
- Liczb z danych syntetycznych **nie raportujemy**. Pokazują tylko, czy kod działa.
- Opis, tryby (`realny`, `brak`, `wbudowany`) i „klucz odpowiedzi” są w `narzedzia/dane_syntetyczne/README.md` i `data_syntetyczne/OPIS.json`.
- Każde zadanie na GitHubie ma na końcu sekcję „Dane syntetyczne”: do czego się w nim przydają i na co uważać. Wyjątki to #32 i #34, które nie dotyczą kodu.

---

## Łukasz: zasady modelowania

**Kod i dokumenty:** `4-pu-setup/PLAN_MODELOWANIA.md`, `4-pu-setup/OPCJE_MODELOWANIA.md`, `4-pu-setup/etap0_agregacja.py` (łączenie rekordów pacjenta), `3-sgkf-split/sgkf/kontrola_skrotow.py` (pierwsza wersja ukrywania chorych i metryki).

### #29 Ustalić, jak oceniamy modele
**Po co:** nie mamy potwierdzonych zdrowych. KOR to „nieoznaczeni”, a nie „na pewno bez tętniaka”, więc zwykła dokładność czy AUC nie mówi, czy model działa. Dlatego część chorych z NEURO ukrywamy (oznaczamy jako 0) i sprawdzamy, ilu z nich model umieszcza na szczycie listy, np. w górnych 5%. Przed treningiem trzeba ustalić:
- ilu chorych ukrywamy,
- która metryka jest główna,
- jak stroimy hiperparametry na foldach 0–3 (Optuna, o czym mówił prowadzący),
- czy zbiór testowy oceniamy w naturalnych proporcjach, czy dodatkowo „po równo z tętniakiem i bez”, o co pytał prowadzący.

Bez tych ustaleń wyniki różnych modeli nie będą porównywalne.

### #30 Ustalić, jak traktujemy pacjentów z wieloma badaniami i z jednym badaniem
**Po co:** model ocenia pacjenta, a pacjent ma od 1 do kilkudziesięciu rekordów. Trzeba wybrać sposób łączenia rekordów w jeden profil (pierwsze porównanie jest w `etap0_agregacja.py`). Jedno badanie ma 60% pacjentów KOR i tylko 33% NEURO, więc samo „ile badań” odróżnia grupy. Trzeba ustalić, jak to kontrolować (to też punkt z notatek Liwii).

### #31 Ustalić modele kontrolne
**Po co:** KOR i NEURO różnią się rzeczami, które nie mają nic wspólnego z tętniakiem:
- które badania wykonano (wzorzec braków),
- z jakich lat są dane,
- jak nazywały się badania.

Model może się nauczyć tych różnic zamiast choroby. Dlatego obok właściwego modelu liczymy modele „ślepe”: losowy, tylko wiek i płeć, tylko wzorzec braków i tylko nazwy badań (o ten pytał prowadzący). Jeśli właściwy model nie jest od nich wyraźnie lepszy, nie nauczył się niczego o chorobie.

### #32 Przegląd modeli i metod PU
**Po co:** w notatkach Liwii są XGBoost, LightGBM, TabNet i AutoGluon, do tego kilka sposobów uczenia PU. Chorych jest około 1 800, a to mało dla sieci neuronowych. AutoGluon sam dzieli dane i nie wie, że rekordy jednego pacjenta muszą być razem. Trzeba wybrać, od czego zaczynamy (np. regresja logistyczna, PU Bagging, LightGBM/XGBoost), a co zostawiamy na później.

### #33 Wspólna funkcja do oceny modeli
**Po co:** jeśli każdy policzy metrykę po swojemu (inaczej ukryje chorych albo inaczej połączy rekordy), wyników nie da się porównać. Potrzebna jest jedna funkcja w repo, z której korzystają wszyscy.

### #34 Pytania do prowadzącego i właściciela danych
**Po co:** części rzeczy nie sprawdzimy sami. Nie wiemy, skąd pochodzi grupa KOR. Nie mamy informacji o pęknięciu tętniaka ani grupy kontrolnej. Trzeba zapytać, zanim zaczniemy tego szukać. Lista pytań jest w `plan_szczegolowy.md`, sekcja 5.

---

## Adrian: podział na foldy (SGKF)

**Kod:**
- `3-sgkf-split/sgkf/podzial.py`: cały podział,
- `3-sgkf-split/uruchom_sgkf.py`: uruchamianie,
- `3-sgkf-split/sgkf/tests/test_podzial.py`: testy,
- `3-sgkf-split/sgkf/results/pacjent_fold.csv` i `podzial_meta.json`: zapisany przydział pacjentów do foldów,
- `3-sgkf-split/sgkf/kontrola_sgkf_czerwiec.py`: porównanie z podziałem z czerwca,
- `3-sgkf-split/archiwum/`: pierwsza wersja podziału (notebook Liwii).

**Opis:** `1026_REALIZACJA_DO_SGKF.md` (sekcje o podziale), `3-sgkf-split/RAPORT_SGKF_MICE.md`.

### #35 Zapoznać się z całym kodem podziału SGKF
**Po co:** to dla Ciebie nowy kod, a od teraz ta część jest Twoja. Trzeba rozumieć:
- jak powstaje podział: pacjent zawsze w jednym foldzie, foldy wyrównane m.in. pod względem liczby chorych, liczby badań i okresu,
- dlaczego jest własne losowanie zamiast wbudowanego w sklearn,
- co sprawdzają testy,
- jak z podziału korzystają imputacja (`aneurysm_sgkf_mice_pipeline.py`) i kontrola skrótów (`kontrola_skrotow.py`).

Niejasności warto zapisywać na bieżąco, przydadzą się na spotkaniu. Na danych syntetycznych można uruchomić cały podział bez obaw o prawdziwe dane (najlepiej na osobnej kopii repo).

### #36 Uruchomić podział na swoim komputerze
**Po co:** podział był dotąd liczony tylko na jednym komputerze. Wcześniej okazało się, że różne wersje sklearn dają różny przydział pacjentów (stąd własne losowanie). Trzeba potwierdzić, że na innym komputerze wychodzi dokładnie ten sam przydział. To zadanie robimy na prawdziwych danych, bo chodzi o porównanie z zapisanym `pacjent_fold.csv`.

### #37 Rozłączność i kompletność podziału w kodzie
**Po co:** prowadzący pytał, czy zbiory są rozłączne. Zostało to sprawdzone jednorazowo i wynik był poprawny, ale nie ma na to testu. Do tego funkcja wczytująca podział (`wczytaj_podzial`) sprawdza tylko, czy dane się nie zmieniły, a nie sam plik z przydziałem. Gdyby plik był uszkodzony albo ucięty, część pacjentów mogłaby po cichu wypaść z obliczeń. Oba sprawdzenia powinny być w kodzie i w testach.

Warto przy tym wiedzieć, że w danych było 63 pacjentów obecnych w obu grupach. Zostali jako chorzy, a ich rekordy KOR usunięto (`3-sgkf-split/przygotowanie/przygotuj_dane.py`).

### #38 Foldy robocze i zabezpieczenie foldu 4
**Po co:** skoro fold 4 jest testowy, łatwo przez nieuwagę użyć go w eksperymencie, np. pętlą po wszystkich foldach. Kod powinien:
- domyślnie dawać tylko układy robocze (trzy z foldów 0–3 do treningu, czwarty do walidacji),
- dawać układ końcowy (foldy 0–3 do treningu, fold 4 do testu) dopiero po świadomym włączeniu.

Z tej samej funkcji korzystają wtedy imputacja i strojenie modeli.

### #39 Stałe listy ukrytych chorych
**Po co:** ukrywanie chorych (#29) musi być takie samo dla wszystkich modeli i wszystkich osób, inaczej wyniki nie są porównywalne. Teraz w `kontrola_skrotow.py` ukryci są losowani przy każdym uruchomieniu z całego NEURO, więc ich liczba w foldach jest nierówna. Potrzebna jest lista ukrytych pacjentów dla każdego ziarna losowania:
- wylosowana tak, żeby w każdym foldzie był ten sam odsetek ukrytych,
- zapisana i zamrożona tak samo jak przydział do foldów.

Ile ukrywamy, ustala Łukasz. Imputacja nie korzysta z etykiety, więc ukrywanie nie wpływa na imputację Dominiki.

### #40 Czy foldy są porównywalne
**Po co:** prowadzący zwracał uwagę na pacjentów z długą historią badań. Fold 4 ma udawać „nowe dane”, więc powinien wyglądać jak pozostałe. Trzeba porównać w foldach:
- liczbę chorych,
- liczbę rekordów,
- pacjentów z jednym badaniem i z długą historią,
- udział danych z okresu wspólnego dla obu grup.

---

## Liwia: dane surowe i przygotowanie danych

**Dane i kod:**
- `data/raw/` (dane surowe, już zagregowane tygodniowo),
- `1-data-preparation/data_analysis_v1.ipynb` i `1-data-preparation/scripts-lf/` (Twoje notebooki z etapu 1),
- `1-data-preparation/scripts-lk/` (skrypty czyszczące),
- `1-data-preparation/cleaning_summary.md`,
- `3-sgkf-split/przygotowanie/` (późniejsze poprawki danych i analizy decyzji z etapu 1),
- `3-sgkf-split/sgkf/kontrola_epoki.py` (jak okres badania odróżnia grupy).

**Opis:** `1026_REALIZACJA_DO_SGKF.md`, sekcja 3.

### #41 Nazwy badań w KOR i NEURO
**Po co:** prowadzący zapytał o „model uczący się po samych nazwach”. Kolumna z nazwą badania (`examination_type`) została usunięta w etapie 1, ale w danych surowych wygląda inaczej w obu grupach. Część nazw występuje tylko w NEURO, np. „Kreatynina” tylko w latach 2010–2013. Nazwy mówią, jaki zestaw badań zlecono, a to przekłada się na to, których cech brakuje. Model może więc rozpoznać grupę „po nazwach” nawet bez tej kolumny. Trzeba sprawdzić:
- które nazwy są w której grupie i w których latach,
- czy różne nazwy oznaczają to samo badanie.

Wynik posłuży też do modelu kontrolnego „tylko nazwy” (#31).

### #42 Informacja o pęknięciu tętniaka w danych surowych
**Po co:** prowadzący pytał, czy w NEURO wiadomo, czy tętniak pękł. Sprawdzenie jednorazowe wykazało, że kolumna rozpoznania (`diagnosis_id`) jest pusta, a w komentarzach nie ma takiej informacji. Nie ma jednak na to skryptu, a znasz dane surowe najlepiej. Trzeba to potwierdzić skryptem albo wskazać miejsce, gdzie ta informacja jednak jest.

### #43 Wartości ujemne i niemożliwe
**Po co:** to punkt z Twoich notatek. W danych wejściowych do podziału nie ma wartości ujemnych, co zostało sprawdzone jednorazowo. Potrzebny jest skrypt, który sprawdza to na danych surowych i przetworzonych, żeby wiedzieć, czy coś takiego było i na którym etapie zniknęło.

### #44 Skąd mamy 38 cech
**Po co:** prowadzący pytał, czy cechy zostały wybrane na podstawie testów statystycznych. Według dokumentacji zdecydowała reguła 60% braków (spotkanie 07.05, Twój notebook). Usunięto wtedy m.in. ALT, AST, BUN i P-LCR. Późniejsze usunięcie CRP, MONO i %MONO na podstawie korelacji z etykietą zostało cofnięte. Trzeba to zebrać w jednym skrypcie:
- które kolumny odpadły,
- ile miały braków w każdej grupie,
- jak brzmiała dokładnie reguła.

Wtedy na pytanie „dlaczego te cechy” można odpowiedzieć liczbami.

### #45 Zmiany parametrów w czasie
**Po co:** to punkt z Twoich notatek („jak zmieniają się parametry na przestrzeni lat, fluktuacje”). KOR i NEURO pochodzą częściowo z różnych lat, a sam okres badania pozwala dość dobrze odróżnić grupy (`kontrola_epoki.py`). Jeśli wartości badań przesuwają się w czasie (zmiana aparatury, metody, norm), model może rozpoznawać rok, a nie chorobę. Trzeba sprawdzić, czy i które parametry się przesuwają. Liczymy tylko na foldach 0–3.

### #46 Czyszczenie z etapu 1 a różnice między grupami
**Po co:** czyszczenie było robione osobno dla każdej grupy, z różnymi progami i na całych danych, przed podziałem:
- progi wartości odstających liczone osobno: KOR stracił 0,8% rekordów, NEURO 3,2%,
- limit historii 11 rekordów w KOR i 38 w NEURO,
- usunięcie 1 793 pacjentów KOR z dużymi brakami.

Prowadzący pytał, na jakich danych były analizy. Trzeba umieć wyjaśnić każdy krok i ocenić, czy samo czyszczenie nie zwiększyło różnic między grupami. Inaczej model mógłby się uczyć skutków czyszczenia.

---

## Dominika: imputacja, braki danych i skalowanie

**Kod:**
- `3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py`: imputacja MICE w foldach,
- `3-sgkf-split/sgkf/ocena_imputacji.py`: ocena imputacji,
- `3-sgkf-split/sgkf/results/przebieg_imputacji_38.json`: przebieg ostatniej imputacji,
- `2-imputation/RAPORT_IMPUTACJA.md` (sekcja 4.6) i `2-imputation/final/validate_imputed.py`: testy KS/KL,
- `4-pu-setup/opcje_fakty.py`: wzorce braków danych.

### #47 Nowa imputacja pod nowy układ foldów
**Po co:** obecne dane po imputacji powstały, gdy każdy fold po kolei był testowy. Imputer dla foldów 0–3 uczył się więc także na foldzie 4. Skoro fold 4 jest teraz testowy, imputer nie może go widzieć. Potrzebna jest imputacja:
- dla każdego z czterech układów roboczych (trzy foldy uczą, jeden sprawdza),
- dla układu końcowego (foldy 0–3 uczą, fold 4 sprawdza).

Korzystamy z funkcji od Adriana (#38). Ustawienia MICE zostają takie jak w etapie 2.

### #51 Skalowanie i standaryzacja wewnątrz foldu
**Po co:** prowadzący zwrócił uwagę na wyciek przy standaryzacji. Jeśli skaler policzy średnie i zakresy na wszystkich danych, informacja z części testowej przecieka do treningu. W imputacji skalowanie jest liczone tylko na części treningowej. Trzeba to potwierdzić w kodzie i przygotować tak samo działający krok standaryzacji dla modeli, które go wymagają (np. regresji logistycznej).

### #48 Testy KS/KL: co sprawdzały
**Po co:** prowadzący pytał, czy testy KL/KS przed i po imputacji posłużyły do wyboru cech. Według raportu z etapu 2 sprawdzały one jakość imputacji, a nie wybór cech. Trzeba to potwierdzić i zdecydować, czy powtórzyć je na nowych foldach.

### #49 Różnice w brakach danych między KOR i NEURO
**Po co:** to punkt z notatek Liwii: odrzucić parametry, które mają bardzo różny odsetek braków w obu grupach. Sam wzorzec braków (które badania wykonano) dobrze odróżnia KOR od NEURO. Kilka cech, np. NRBC, NEUT, CRP i eGFR, różni się brakami o ponad 20 pp. Trzeba policzyć różnice dla wszystkich cech na foldach 0–3 i zaproponować kilka zestawów cech do późniejszego porównania (np. wszystkie 38, bez cech z największą różnicą, sama morfologia).

### #50 Pacjenci o podobnych brakach
**Po co:** prowadzący prosił, żeby sprawdzić model na danych z podobnymi brakami. Chodzi o pogrupowanie pacjentów według tego, które badania mają wykonane. Wtedy można ocenić model w grupach, w których KOR i NEURO mają te same badania, i model nie może „wygrać” tylko dlatego, że rozpoznaje, jakie badania komuś zlecono.

---

## Odkładamy na później

- **Grupa kontrolna z innego źródła (np. krwiodawcy):** to zupełnie inna populacja, więc model nauczyłby się odróżniać zdrowych od pacjentów szpitala. Najpierw pytamy właściciela danych.
- **Tętniak pęknięty / niepęknięty:** dopiero jeśli #42 albo właściciel danych wskaże, że ta informacja istnieje.
- **TabNet i AutoGluon:** dopiero gdy prostsze modele będą działały.
- **Zmiana na 10 foldów:** obecne 5 foldów jest zamrożonych i sprawdzonych; zmiana oznaczałaby przeliczenie całego etapu 3.

## Do dopytania Liwię (z jej notatek)

- „Parametr odcięcia jako zmienna do modelowania”: chodzi o próg w rankingu (np. górne 5%) czy o próg braków (60%)?
- „Test set out (z rozkładem normalnym)”: chodzi o naturalne proporcje chorych w zbiorze testowym?
- „PUBLIKACJA: T1–T4”: co oznacza?
