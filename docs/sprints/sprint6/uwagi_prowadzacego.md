# Uwagi prowadzącego do raportu z października

**Źródło:** analiza stanu projektu (06.11) i uwagi prowadzącego do raportu z października (19–20.11.2026).
**Do czego:** fakty i liczby, z których wyszły zadania Sprintu 6. Zadania są w `README.md` i na GitHubie (issues #29–#51), a decyzje w `plan_szczegolowy.md`.

**Oznaczenia przy liczbach:**
- **[repo]**: liczbę liczy skrypt w repozytorium; można cytować w raporcie,
- **[jednorazowo]**: policzone jednorazowo, bez skryptu w repo; **przed cytowaniem trzeba dodać skrypt** (zadania Sprintu 6),
- **[do ustalenia]**: pytanie lub decyzja, nie fakt.

---

## 0. Stan wyjściowy (06.11)

- Etap 3 jest technicznie zamknięty: podział `StratifiedGroupKFold` (5 foldów, `y` = 12 warstw, seed 42) jest zamrożony w `3-sgkf-split/sgkf/results/pacjent_fold.csv`. Testy przechodzą 18/18, a podział zbudowany od zera jest identyczny z zapisanym. 10 plików zaimputowanych foldów zgadza się z odciskami (5/5) **[repo + niezależne sprawdzenie 06.11]**.
- Etap 4 (modelowanie PU) ma plan (`4-pu-setup/PLAN_MODELOWANIA.md`) i mapę opcji (`4-pu-setup/OPCJE_MODELOWANIA.md`), ale **nie ma kodu**.
- Wszystkie analizy etapów 3–4 liczą na danych przetworzonych (`data/processed/aneurysm_sgkf_input.csv`: 78 052 rekordy, 40 924 pacjentów, 1 823 chorych). Żaden skrypt etapów 3–4 nie czyta danych surowych.

---

## 1. Uwagi prowadzącego

### 1.1 Model uczący się na samych nazwach badań

**O co chodzi.** W danych surowych kolumna `examination_type` zawiera listę nazw zleconych badań (np. „Morfologia”, „Kreatynina we krwi”). W etapie 1 usunęliśmy ją jako kolumnę tekstową. Raport przejściowy nazywa ją „zmienną czysto techniczną” (`docs/reports/0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md:35`), a raport z października tylko „kolumną tekstową” (krok 1). Pytanie brzmi: czy model rozpozna chorych po samych nazwach badań, bez ich wyników?

**Co wiemy:**
- Nazwy **nie były ujednolicone** między okresami. Zmieniały się w czasie, np. „Kreatynina” i „Azot Mocznika” występują tylko w latach 2010–2013, więc tylko w NEURO, bo KOR nie ma danych sprzed 2019. Nazwy zdradzają więc okres badania, a przez niego kohortę **[jednorazowo]**.
- Na naszym wejściu: KOR ma 63 nazwy, NEURO 47; wspólnych jest 43, tylko w KOR 20, tylko w NEURO 4. „Kreatynina” występuje w 10,4% rekordów NEURO (694), „Azot Mocznika” w 8,8% (584). CRP: 81,6% rekordów KOR wobec 54,2% NEURO **[jednorazowo]**.
- Wartości badań są we wspólnych kolumnach: oba pliki surowe mają te same 128 kolumn **[jednorazowo]**.
- Ślad nazw wraca do danych jako wzorzec braków (badania nie zlecono → brak wyniku). Sam wzorzec braków odróżnia kohorty z AUC 0,82, a w ocenie z ukrywaniem odzyskuje 52% ukrytych chorych w górnych 5% rankingu **[repo: `analiza_wynikow.py`, `kontrola_skrotow.py`]**.
- Model na pełnej liście nazw **nie został policzony**. Lista nazw jest szersza niż 38 cech, więc może dać więcej niż 0,82.

**Zadania:** #41 (nazwy badań według kohort i lat; rekordy wejścia dopasowują się do danych surowych po `custom_id` i dacie), #31 (model kontrolny „same nazwy badań”). Do raportu: zdanie, że kolumnę z nazwami usunięto, ale jej informacja wraca przez wzorzec braków, plus wynik modelu na samych nazwach.

**Do ustalenia:** czy prowadzący pytał właśnie o `examination_type`, czy o inne „nazwy” (jednostki `*-unit`, komentarze laboratorium).

### 1.2 Czy zbiory są rozłączne

**Wynik [jednorazowo, na `aneurysm_sgkf_input.csv`]:**
- wspólni pacjenci KOR i NEURO: **0** (w danych surowych było 63; ich rekordy KOR usunięto 04.10 **[repo]**),
- pacjenci z dwiema etykietami: 0; wspólne lub powtórzone `custom_id`: 0,
- identyczne rekordy KOR i NEURO przy różnych numerach pacjenta (data, wiek, płeć, 38 cech): 0,
- luźniej (ta sama data i płeć, wiek ±0,1 roku, wszystkie wspólnie zmierzone wartości równe, co najmniej 10): 0 par na 744 kandydatów,
- pacjent w więcej niż jednym foldzie: 0; rekord bez foldu: 0.

**Ograniczenie:** tej samej osoby pod dwoma różnymi numerami nie da się wykluczyć w 100%, bo nie mamy danych identyfikujących.

**Zadanie:** #37 (kontrola jako test w repo).

### 1.3 Tętniak pęknięty czy niepęknięty

**Wynik [jednorazowo]:** w danych **nie ma tej informacji**.
- Jedyne pole na rozpoznanie, `diagnosis_id`, jest puste we wszystkich wierszach: 7 655 w NEURO i 73 679 w KOR.
- W 551 komentarzach laboratoryjnych NEURO (`descriptive_result`) nie ma słów: tętniak, pęknięcie, krwotok, SAH, podpajęczynówkowy, udar, klipsowanie, embolizacja ani kodów I60–I67. Wśród nazw badań nie ma badania płynu mózgowo-rdzeniowego.
- Plik surowy NEURO jest ten sam co do bajtu od pierwszego commita (13.04). Kopie pośrednie też mają puste `diagnosis_id`.

**Dlaczego to ważne:** jeśli NEURO to głównie tętniaki pęknięte, profil może opisywać ostrą fazę krwotoku i leczenie, a nie samego tętniaka (`OPCJE_MODELOWANIA.md:387`).

**Zadania:** #42 (kontrola braku rozpoznania jako skrypt), #34 (pytanie do właściciela danych o rozpoznanie, np. ICD-10: I60 to krwotok podpajęczynówkowy, czyli tętniak pęknięty; I67.1 to tętniak niepęknięty). Nazwa pliku `merged_aggregated_1W_mean` sugeruje, że rozpoznanie mogło przepaść przy uśrednianiu tygodniowym. To przypuszczenie.

### 1.4 Testy statystyczne (KL, KS) a wybór cech

**Fakty:**
- KL i test Kołmogorowa–Smirnowa liczono w etapie 2, **do oceny imputacji, nie do wyboru cech** **[repo]**. KL była niska (KOR 0,003–0,005, NEURO 0,010–0,016). Test KS odrzucił zgodność dla 38 z 38 cech w obu kohortach, bo przy około 78 tys. rekordów wykrywa każdą różnicę, a kompletne wiersze nie są losową próbą. Raport uznał RMSE za właściwą miarę (`2-imputation/RAPORT_IMPUTACJA.md`, sekcja 4.6; `validate_imputed.py`, `validation_ks.csv`).
- **38 cech wybrano w etapie 1 według braków.** Z 42 kolumn wyników i demografii wypadły ALT, AST, BUN i P-LCR, bo w co najmniej jednej kohorcie brakuje ich w ponad 60% rekordów. P-LCR wypadł przez NEURO (62%), choć w KOR brakuje go tylko w 4% **[jednorazowo; reguła odtworzona z liczb, w dokumentach jest tylko „próg 60%”]**.
- Zmiana 38 → 35 (bez CRP, MONO, %MONO) wynikała z korelacji z etykietą, nie z KL. Cofnięto ją 04.10 (`docs/reports/1026_REALIZACJA_DO_SGKF.md`, sekcja 6.4) **[repo]**.

**Zadania:** #44 (skąd 38 cech: reguła 60% w skrypcie; dziś jest tylko w notebooku z etapu 1), #48 (co sprawdzały KS/KL). Do raportu: jedno zdanie o tym, skąd jest 38 cech, i że KL/KS służyły ocenie imputacji.

### 1.5 Trenowanie na foldach z Optuną i wybór najlepszego

**Rozumienie:** strojenie hiperparametrów Optuną w walidacji krzyżowej i wybór najlepszych ustawień.

**Pułapka:** strojenie i raportowanie na tych samych 5 foldach daje zawyżony wynik. Ten błąd był już w etapie 2: pierwszy przebieg benchmarku odrzucono, bo Optuna stroiła na masce, na której liczono wynik.

**Pierwotny plan (`PLAN_MODELOWANIA.md` §15):** walidacja zagnieżdżona. 5 zamrożonych foldów zewnętrznych służy tylko do oceny. W treningu każdego z nich są 3 foldy wewnętrzne (`podzial_wewnetrzny()` już jest i ma test), a Optuna stroi tylko na nich. Cel: `RecallHidden@q` na stałych maskach. Najlepsze ustawienia trenujemy na całym treningu i oceniamy raz na foldzie zewnętrznym.

**Propozycja Sprintu 6 (D1 + D3 w `plan_szczegolowy.md`):** zamiast walidacji zagnieżdżonej fold 4 odkładamy jako test, a Optuna stroi na 4 foldach roboczych (te same hiperparametry, średnia z foldów). MICE i skalowanie trzeba wtedy dopasować od nowa w każdym układzie roboczym (#47, #51). **Nie wolno dzielić gotowych `foldy_imputowane_38/`.** Zakresy parametrów, liczbę prób, ziarna i limit czasu zapisujemy przed startem (#29).

**Do ustalenia z prowadzącym:**
- Czy „wybrać najlepszy” znaczy najlepsze ustawienia (poprawnie), czy model z najlepszego foldu (błąd: wybór szczęśliwego podziału)?
- Czy Optuna ma wybierać też rodzinę modelu (logistyczna, PU Bagging, drzewa)? Wtedy ten wybór też musi być w pętli wewnętrznej.

### 1.6 Test na danych z podobnymi brakami

**O co chodzi:** porównać ukrytych chorych tylko z pacjentami KOR o tym samym zestawie badań. Wtedy sam wzorzec braków nic nie daje, a przewagę mogą dać tylko wartości badań. To najważniejszy test projektu, bezpośrednio związany z 1.1.

**Co wiemy [repo: `4-pu-setup/results/opcje_fakty.json`]:**
- 85,5% chorych ma wzorzec braków (na poziomie bloków badań), który występuje też u co najmniej 50 pacjentów KOR,
- 12,3% (225 chorych) ma wzorzec nieobecny w KOR; tych tak nie ocenimy,
- panele kompletne u pacjentów: sama morfologia 99% w KOR i NEURO; z nerkami i elektrolitami 82% w KOR i 74% w NEURO; wszystkie 36 cech laboratoryjnych 46% w KOR i 41% w NEURO.

**Zadania:** #50 (grupy pacjentów o podobnych brakach), #49 (warianty zestawu cech według różnic w brakach). Później: `RecallHidden@q` w warstwach wzorca braków (OPCJE H2), model na panelu kompletnym w obu kohortach (morfologia + wiek + płeć) bez imputacji, opis 225 chorych z wzorcem nieobecnym w KOR jako ograniczenie.

### 1.7 Proporcje w zbiorze testowym („po równo z tętniakiem i bez”)

**Fakty [repo]:**
- Między foldami: **tak, po równo**. Każdy fold ma 364–366 chorych na 8 184–8 185 pacjentów (4,45%), rozstęp 0,02 pp.
- W foldzie testowym: **nie 50/50**. Chorzy to 4,45% pacjentów i 8,5% rekordów. W ocenie z ukrywaniem pula testowa to około 7 800 pacjentów KOR i średnio około 146 ukrytych chorych na fold.

**Uwaga:** zbalansowanie 50/50 zmienia wynik `RecallHidden@q` (mniejsza pula KOR → łatwiej trafić do top q%). AUC nie zależy od proporcji.

**Do ustalenia:** które znaczenie miał prowadzący. Propozycja: wynik główny na naturalnych proporcjach, wersja 50/50 jako uzupełnienie (D6, #29, #34).

### 1.8 Pacjenci z dłuższą historią

**Fakty [repo]:**
- Podział: klasa liczby rekordów (1 / 2 / 3–4 / 5+) jest w stratyfikacji. Chorzy z 5+ rekordami to 20,6–20,9% chorych w każdym foldzie.
- Plan: jeden profil na pacjenta (mediana cech). Mediana najsłabiej wiąże się z liczbą badań: korelacja z liczbą rekordów przy 2 z 38 cech, przy maksimum 19 (`4-pu-setup/etap0_agregacja.py`).
- Liczba rekordów różni kohorty: NEURO średnio 3,65 na pacjenta, KOR 1,83.
- Etap 1 ograniczył historię osobno dla kohort: do 11 rekordów w KOR i 38 w NEURO. Wspólny limit 11 usunąłby 943 rekordy 94 chorych (`analiza_decyzji.json`, sekcja L).
- MICE nadal uczy się na rekordach, więc pacjenci badani częściej mają większy wpływ na imputację.

**Zadania:** #30 (jak traktujemy pacjentów z wieloma i jednym badaniem), #40 (porównanie foldów), #46 (limity historii w czyszczeniu z etapu 1). Później: wyniki modeli według liczby rekordów i wariant „tylko pierwszy rekord pacjenta”.

**Do ustalenia:** różne limity historii (11/38) zostawiamy jako ograniczenie czy ujednolicamy przy ponownym zamrożeniu danych?

### 1.9 Wyodrębniony zbiór testowy

**Stan:** 5 foldów, każdy pacjent raz w teście; nie ma zbioru odłożonego do końca projektu.

**Opcje:**
- walidacja zagnieżdżona (1.5): fold zewnętrzny nie bierze udziału w strojeniu, więc działa jak odłożony test, powtórzony 5 razy,
- odłożony fold, np. fold 4 (8 184 pacjentów, 364 chorych): zamknięty do końca, strojenie na foldach 0–3, jedna ocena na końcu. Kosztem jest mniej danych do nauki.

**Uwaga:** diagnostyka etapu 3 (AUC, kontrola skrótów) korzystała ze wszystkich foldów, ale żadnego modelu ani ustawień jeszcze na nich nie wybrano.

**Propozycja Sprintu 6:** fold 4 odłożony, walidacja krzyżowa na foldach 0–3 (D1, D3; #38).

### 1.10 Standaryzacja w foldzie (wyciek danych)

**Fakty [repo: `3-sgkf-split/sgkf/aneurysm_sgkf_mice_pipeline.py:123–157`]:**
- normalizacja `MinMaxScaler` do [0, 1] jest dopasowana tylko na kompletnych wierszach treningu foldu,
- MICE jest dopasowane tylko na treningu, test jest tylko przekształcany,
- po imputacji skalowanie jest odwracane: zapisane foldy mają oryginalne jednostki.

**Czego nie ma:** standaryzacji dla modeli, bo nie ma modeli. Plan (`PLAN_MODELOWANIA.md` §11.2): `StandardScaler` dla regresji logistycznej, dopasowany tylko na profilach pacjentów treningowych (#51).

**Miejsca wycieku do pilnowania lub opisania:**
- etap 1: progi kwantylowe i limity historii ustalone na całych danych przed podziałem, osobno dla kohort (ograniczenie, `1026_REALIZACJA`, sekcja 9, pkt 8),
- strojenie: skalowanie i MICE w każdym układzie roboczym trzeba dopasować od nowa (1.5, #47),
- etap 2: skalowanie na całym zbiorze, ale tylko w benchmarku; plik zaimputowany globalnie jest „nie do modeli”.

### 1.11 Ukrycie tętniaków jako „?” i wrzucenie do 0

**To nasz sposób oceny** (pomysł prowadzącego, `PLAN_MODELOWANIA.md` §1 i §12):
1. losujemy 40% pacjentów NEURO (całych pacjentów, ze wszystkimi rekordami),
2. w treningu dostają etykietę 0 i trafiają do nieoznaczonych; w puli U treningu jest ich średnio 583 na 31 864 osoby (1,83%) **[repo: `opcje_fakty.json`]**,
3. prawdziwa etykieta zostaje osobno, tylko do oceny: jaki odsetek ukrytych z części testowej trafia do górnych q% rankingu (`RecallHidden@q`),
4. maski są stałe (ziarna 0–4) i wspólne dla wszystkich modeli.

**Co już jest [repo: `3-sgkf-split/sgkf/kontrola_skrotow.py`]:** wynik dla modeli kontrolnych przy q = 5%: losowy ranking 4%, wiek i płeć 12%, sam wzorzec braków 52%, 38 cech 66–69%.

**Uwaga:** ukryci zachowują cechy kohorty NEURO, więc sam wysoki wynik niczego nie dowodzi. Liczy się przewaga nad modelami kontrolnymi, najlepiej przy podobnych brakach (1.6).

**Zadania:** #29 (zasady oceny), #39 (stałe listy ukrytych), #33 (wspólna funkcja oceny: `original_label` oddzielnie od `observed_label`, ręcznie sprawdzony przykład metryki, remisy w rankingu). Można wykorzystać wrześniowy prototyp (`4-pu-setup/pu/ukrywanie.py`, `metryki.py` w commicie `2cb6d1a`), ale **najpierw trzeba zabezpieczyć ten commit tagiem** (sekcja 2).

---

## 2. Zaległości techniczne (z analizy 06.11)

| Pilność | Sprawa | Co zrobić / zadanie |
|---|---|---|
| **pilne** | prototyp PU (commit `2cb6d1a`, 25.09) nie należy do żadnego brancha ani taga, lokalnie ani na GitHubie; trzyma go tylko lokalna historia gita, którą git może wyczyścić | tag + `git push` taga (Łukasz, poza issues) |
| wysoka | podział nie był uruchomiony na drugim komputerze (sklearn 1.8) | #36 |
| wysoka | `wczytaj_podzial()` sprawdza tylko odcisk danych, nie sam plik przydziału (przyjąłby fold 99; przy uciętym pliku pipeline MICE po cichu pominie pacjentów) | #37 |
| wysoka | zaimputowane foldy czytane po nazwie pliku, bez manifestu | przy #47 |
| średnia | brak wspólnych, przypiętych wersji bibliotek (łącznie z PyArrow); odcisk danych zależy od wersji pandas | odłożone; wersje zapisujemy w plikach wyników |
| średnia | `master` nie ma prac od 11.09 (35 commitów) | po Sprincie 5 |
| niska | drobne niezgodności w dokumentach: `docs/reports/1026_REALIZACJA_DO_SGKF.md:357` podaje rozstęp 0,55 pp (to liczba wariantu testowanego; dla SGKF 0,52); „0,05 pp” dla rekordów NEURO w trzech dokumentach, a CSV daje 0,0448 (0,04); komunikat błędu w `podzial.py:399` ma ścieżkę bez `sgkf/` | poprawić przy najbliższej zmianie dokumentów |
| niska | etap 1 nieodtwarzalny jednym skryptem (notebooki, ścieżki z Colab, kopie danych w kilku miejscach) | opisane jako ograniczenie; porządki przy ponownym zamrożeniu danych |

---

## 3. Pytania zebrane w jednym miejscu

Ostateczna lista do wysłania: `plan_szczegolowy.md`, sekcja 5 (#34).

**Do prowadzącego:**
1. „Model na samych nazwach”: chodziło o nazwy zleconych badań (`examination_type`)?
2. „Po równo z tętniakiem i bez”: między foldami czy 50/50 w zbiorze testowym?
3. Odłożony fold testowy czy tylko walidacja zagnieżdżona?
4. Optuna: wybór ustawień czy też rodziny modelu?
5. Limity historii 11/38: ograniczenie czy ujednolicenie?
6. Wcześniej otwarte (z `docs/reports/1026_SPOTKANIE_PROWADZACY.md`): cel predykcji, skąd pochodzi KOR, walidacja zewnętrzna, decyzja o NRBC/%NRBC.

**Do właściciela danych:**
1. Rozpoznanie dla pacjentów NEURO: pęknięty czy niepęknięty (kod ICD-10)?
2. Kiedy wykonano badania NEURO względem rozpoznania i leczenia; czy są daty rozpoznania?
3. Skąd pochodzi kohorta KOR (oddział, kryteria)?
4. Czy nazwy badań zmieniały się w systemie laboratoryjnym (2013, 2020–2021)?

---

## 4. Zadania

Lista zadań z tych notatek trafiła do Sprintu 6 (issues #29–#51, opis w `README.md`). Kolejne kroki po sprincie są w `plan_szczegolowy.md`, sekcja 6.
