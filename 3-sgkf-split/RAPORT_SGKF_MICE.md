# Etap 3: przygotowanie danych i podział SGKF z imputacją w foldach

**Wersja 1:** 11.06.2026, StratifiedGroupKFold na rekordach, materiał do raportu przejściowego (`archiwum/`)
**Wersja 2:** 04.10.2026, decyzje dotyczące danych, zamrożony podział po pacjentach, pełna imputacja w foldach
**Wejście etapu:** `data/imputation-inputs/aneurysm_concatted.csv` (78 197 rekordów, 38 cech, przed imputacją)
**Wyjście etapu:** zamrożony przydział `sgkf/results/pacjent_fold.csv` oraz zaimputowane foldy (38 cech)

---

## W skrócie

**Co robi ten etap.** Najpierw porządkuje dane według decyzji z 04.10.2026 (część A). Potem dzieli 40 924 pacjentów na 5 foldów tak, żeby żaden pacjent nie był jednocześnie w treningu i w teście. Na koniec w każdym foldzie uzupełnia braki (MICE), ucząc imputer wyłącznie na części treningowej (część B). Wynik to 5 par train/test bez braków i bez wycieku informacji, gotowych do modelowania.

| Element | Stan | Gdzie |
|---|---|---|
| przygotowane dane (decyzje 04.10.2026) | **gotowe**, 78 052 rekordy, spis każdej zmiany | `data/processed/aneurysm_sgkf_input.csv`, `przygotowanie/results/` |
| liczby uzasadniające decyzje | gotowe | `przygotowanie/results/analiza_decyzji.json` |
| przydział pacjent → fold | **zamrożony**, z odciskiem danych | `sgkf/results/pacjent_fold.csv`, `podzial_meta.json` |
| zestaw cech: **38, jeden wariant** | selekcja z etapu 1 (usunięcie CRP, MONO, %MONO) cofnięta | `przygotowanie/results/analiza_selekcji_cech.json`, sekcja 2.6 |
| imputacja MICE w foldach | **pełny przebieg wykonany** | `sgkf/results/przebieg_imputacji_38.json` |
| zaimputowane foldy | lokalnie, poza gitem (46 MB), odtwarzalne jednym poleceniem; dwa przebiegi dały identyczne odciski | `sgkf/results/foldy_imputowane_38/` |
| jakość imputacji (protokół z etapu 2, w foldach) | jak w benchmarku przy losowych brakach; przy brakach całymi panelami uzupełnienia ≈ średnia | `sgkf/results/ocena_imputacji.json`, sekcja 4.3 |
| analiza wyników: na czym model mógłby się „przejechać” | kohorty rozdzielne z AUC 0,93; sam wzorzec braków 0,82; epoka w wynikach NEURO 0,83; KOR wygląda na populację szpitalną | `sgkf/results/analiza_wynikow.json`, sekcja 5 |
| porównanie 38 vs 35 (jednorazowe, podkładka pod decyzję) | brak istotnego wpływu na imputację | `sgkf/results/porownanie_38_35/`, sekcja 4.5 |
| testy | 6/6 (część A), 10/10 (część B) | `przygotowanie/tests/`, `sgkf/tests/` |

**Jak użyć wyniku w kolejnym etapie:**

```python
import sys; sys.path.insert(0, "3-sgkf-split/sgkf")
from podzial import wczytaj_podzial           # przydział pacjent -> fold (sprawdza odcisk danych)
przydzial, meta = wczytaj_podzial()

import pandas as pd                            # albo gotowe zaimputowane foldy
train0 = pd.read_parquet("3-sgkf-split/sgkf/results/foldy_imputowane_38/fold0_train.parquet")
```

**Jak odtworzyć cały etap:** `python 3-sgkf-split/uruchom_sgkf.py` (około 35 min na Macu z procesorem M, w tym 15 min imputacji i 15 min jej oceny). Opcja `--czesc A` albo `--czesc B` uruchamia jedną część, a `--bez-mice` pomija imputację.

---

## 1. Problem i motywacja

**Pacjent w obu zbiorach.** Dane zawierają wielokrotne pomiary tych samych pacjentów: od 1 do 38 rekordów na osobę, średnio 1,91. Losowy podział rekordów (`train_test_split`) umieściłby różne pomiary tego samego pacjenta w treningu i w teście. Model oceniany byłby wtedy na osobie, którą już widział. To **data leakage**, który sztucznie zawyża metryki.

**Niezbalansowane klasy.** 4,45% pacjentów to `label = 1` (NEURO), więc losowy podział mógłby zaburzyć proporcje klas w foldach.

**Uczący się preprocessing.** Scaler i imputer też uczą się z danych. Gdyby imputacja była wykonana **przed** podziałem, statystyki z części testowej wpłynęłyby na uzupełnianie braków w treningu, co również jest wyciekiem. Dlatego imputacja odbywa się wewnątrz foldu.

**Dane z błędami i niejednoznacznościami.** Weryfikacja po raporcie przejściowym (etap 0) wykazała:
- pacjentów obecnych w obu kohortach,
- wartości niemożliwe fizjologicznie,
- mieszane jednostki kreatyniny.

To trzeba rozstrzygnąć **przed** zamrożeniem podziału, bo zmiana danych po fakcie unieważnia foldy.

---

## 2. Część A: przygotowanie danych do SGKF

**Kod:** `przygotowanie/przygotuj_dane.py` · **testy:** `przygotowanie/tests/test_przygotowanie.py` · **podkładka liczbowa:** `przygotowanie/analiza_decyzji.py`, `przygotowanie/analiza_selekcji_cech.py`

Część A jest osobnym krokiem. Czyta plik źródłowy i zapisuje przygotowany plik `data/processed/aneurysm_sgkf_input.csv`, który jest jedynym wejściem części B. Plik źródłowy zostaje nietknięty. Każdą zmianę zapisuje się do spisu: `przygotowanie/results/usuniete_rekordy.csv` i `usuniete_wartosci.csv`.

### 2.1 Decyzje z 04.10.2026

| # | Problem | Decyzja | Skutek w danych |
|---|---|---|---|
| 1 | 63 pacjentów ma rekordy i w KOR, i w NEURO | zostają jako **pozytywni**, ich rekordy KOR są usuwane; kto to był, zapisuje spis `przygotowanie/results/usuniete_rekordy.csv` (dane nie mają dodatkowej kolumny) | −145 rekordów KOR; pacjentów bez zmian (40 924) |
| 2 | KREA > 50 mg/dl | wartość → brak | 210 wartości (KOR 195, NEURO 15) |
| 3 | KREA > 10 mg/dl przy eGFR-MDRD ≥ 30 | wartość → brak | 516 wartości (KOR 478, NEURO 38) |
| 4 | K > 15 mmol/l | wartość → brak | 25 wartości (wszystkie KOR, maks. 28,3) |
| 5 | Na < 80 mmol/l | wartość → brak | 1 wartość (NEURO, 74) |
| 6 | rozjazd czasowy kohort | **bez korekty**: dane są, jakie są; opisane jako ograniczenie; stratyfikacja podziału wyrównuje epokę między foldami | — |
| 7 | asymetria braków między kohortami | **bez korekty**: ograniczenie, raportowane w każdym foldzie | — |
| 8 | zestaw cech (w etapie 1 usunięto CRP, MONO, %MONO) | **38 cech, jeden wariant**: selekcja z etapu 1 cofnięta (sekcja 2.6) | CRP, MONO, %MONO wracają |

Łącznie **752 wartości zamienione na brak** (726 KREA, czyli 1,1% jej pomiarów; 25 K, czyli 0,04%; 1 Na) i **145 usuniętych rekordów**. Wynik to 78 052 rekordy, 40 924 pacjentów, w tym 1 823 pozytywnych.

**Zasada dla wartości niemożliwych:** usuwamy tylko wartość, która **nie może** być prawdziwym wynikiem żywego pacjenta albo przeczy innemu pomiarowi z tego samego rekordu. Wartości skrajne, ale możliwe w ciężkich stanach, zostają:

| Zostaje | Liczba | Dlaczego |
|---|---:|---|
| K 9–15 mmol/l | 61 | ciężka hiperkaliemia, w skrajnych przypadkach przeżywalna |
| Na 80–100 mmol/l | 7 | ciężka hiponatremia, przeżywalna |
| WBC > 200 G/l (maks. 253) | 7 | możliwa przy białaczce |
| KREA > 10 przy eGFR < 30 | 252 | spójne z niewydolnością nerek (dializa) |
| KREA > 10 bez eGFR | 21 | brak podstaw, by uznać je za sprzeczne |

Wartości są średnimi tygodniowymi. Niemożliwa średnia (np. K = 26) oznacza, że co najmniej jeden pomiar w tym tygodniu był niemożliwy. Możliwa średnia może jednak ukrywać pojedynczy błędny pomiar, i tego nie da się wykryć.

### 2.2 Dlaczego reguła dla KREA odwołuje się do eGFR

eGFR (szacowany wskaźnik filtracji kłębuszkowej) laboratorium **wylicza z kreatyniny**, wieku i płci: im wyższa kreatynina, tym niższe eGFR. Rekord z wysoką kreatyniną i prawidłowym eGFR zawiera więc dwie sprzeczne liczby. Najprostsze wyjaśnienie: laboratorium liczyło eGFR z prawidłowej kreatyniny, a do kolumny trafiła wartość w innej jednostce (µmol/l zamiast mg/dl, przelicznik 88,4) albo z błędem wpisu.

Dane potwierdzają ten obraz (`analiza_decyzji.json`, sekcja D, dane przed przygotowaniem):

| KREA [mg/dl] | Wartości | Mediana eGFR-MDRD | eGFR ≥ 60 | eGFR ≥ 30 |
|---|---:|---:|---:|---:|
| ≤ 1,2 | 49 108 | 60,0 | 87% | 97% |
| 1,2–5 | 13 421 | 37,5 | 3% | 68% |
| 5–10 | 1 026 | **8,0** | 4% | 9% |
| 10–20 | 362 | 30,8 | 28% | **51%** |
| 20–50 | 427 | **60,0** | 53% | **78%** |
| > 50 | 210 | 60,0 | 66% | 80% |

Do KREA ≈ 10 mg/dl zależność jest taka, jak powinna: wyższa kreatynina oznacza niższe eGFR. Powyżej 10 zależność się odwraca, a mediana eGFR wraca do wartości prawidłowej. Stąd reguła:

- **KREA > 50 mg/dl → brak.** Takiej wartości w mg/dl nie ma u żywego człowieka, więc to błąd jednostki.
- **KREA > 10 mg/dl przy eGFR-MDRD ≥ 30 → brak.** To sprzeczność z eGFR. Próg 10 jest zachowawczy: w przedziale 5–10 zostaje około 90 wartości z eGFR ≥ 30, ale tam część pacjentów może być w trakcie dializy, co zaburza zależność.
- **Pozostałe KREA > 10 zostają**: 252 przy eGFR < 30 to wiarygodna niewydolność nerek, a 21 bez eGFR nie da się sprawdzić.

Prosty próg „KREA > 20” byłby łatwiejszy do opisania, ale usunąłby 83 wiarygodne wartości (eGFR < 30) i zostawił sprzeczne wartości 10–20. Dlatego go nie wybraliśmy.

Wartości usunięte nie są przeliczane ÷ 88,4, choć wiele z nich to najpewniej µmol/l. Jednostki konkretnego pomiaru nie znamy, więc przeliczenie zgadywałoby. Brak uzupełniony przez MICE korzysta m.in. z eGFR tego samego rekordu, więc odtwarza wartość spójną z resztą profilu.

### 2.3 Dlaczego te wartości przetrwały wcześniejsze czyszczenie

Na etapie 1 (notebook `1-data-preparation/scripts-lf/aneurysm_data_analysis (1).ipynb`) wartości odstające usuwano **kwantylami, parametr po parametrze, całymi rekordami** (`analiza_decyzji.json`, sekcja F):

| Kohorta | Rekordy przed | Usunięte | Udział | Zakres kwantyli |
|---|---:|---:|---:|---|
| KOR | 72 718 | 585 | 0,8% | 0,0001–0,0007 |
| NEURO | 7 053 | 228 | 3,2% | 0,0002–0,001 |

Granice były statystyczne, nie fizjologiczne: dla KOR górna granica K wynosiła 29,15 mmol/l, a KREA 211 mg/dl. Dlatego wartości niemożliwe zostały w danych. Reguła z części A nie dubluje tamtego kroku, tylko uzupełnia go o kryterium fizjologiczne i usuwa pojedyncze komórki, a nie całe rekordy.

Przy okazji widać drugą asymetrię: **NEURO było czyszczone ostrzej niż KOR** (3,2% wobec 0,8% rekordów). Tego kroku nie powtarzamy, bo to stan zastany etapu 1, ale trafia do ograniczeń.

### 2.4 63 pacjentów w obu kohortach

| Układ czasowy rekordów | Pacjentów |
|---|---:|
| wszystkie rekordy KOR **przed** NEURO | 55 (mediana odstępu 546 dni; 35–1 365) |
| wszystkie NEURO przed KOR | 6 |
| przeplatane | 2 |

Pacjent nie może być jednocześnie pozytywny (NEURO) i nieoznaczony (KOR), bo podział i etykieta są pacjentowe. Rozpoznanie tętniaka jest faktem, a „nieoznaczony” oznacza tylko brak informacji, więc pacjent zostaje pozytywny. Jego rekordy KOR są usuwane, bo bez daty diagnozy nie wiadomo, czy pochodzą sprzed choroby. Gdyby zostały z etykietą 1, model uczyłby się, że rutynowy wynik sprzed lat jest „profilem chorego”. Skala: 145 rekordów, 0,19% danych.

### 2.5 Kontrole jakości po przygotowaniu

`analiza_decyzji.json`, sekcja K, wszystkie spełnione:
- `custom_id` (pacjent-tydzień) jest unikalny,
- każdy pacjent jest w jednej kohorcie,
- brak wartości ujemnych, procenty w zakresie 0–100,
- płeć i wiek bez braków (wiek 18–100), więc MICE ich nie uzupełnia i nie powstaje „ułamkowa płeć”,
- żaden rekord nie jest pusty.

### 2.6 Zestaw cech: dlaczego 38, a nie 35

W etapie 1 (notebook `1-data-preparation/scripts-lf/aneurysm_data_merge.ipynb`, 08.06.2026) usunięto CRP, MONO i %MONO. Skrypt `przygotowanie/analiza_selekcji_cech.py` odtwarza tę procedurę na danych źródłowych. Wynik jest identyczny z notebookiem: te same trzy cechy, wartości korelacji zgodne co do cyfry we wszystkich czterech miarach.

**Jak wybrano te cechy.** Dla każdej z czterech miar zależności z etykietą (Pearson, Spearman, Kendall, Phi-K) wzięto 10 cech najsłabiej skorelowanych. Zliczono, ile razy każda cecha znalazła się na tych listach, i usunięto trzy najczęstsze.

| Cecha | „Głosy” (na 4 możliwe) |
|---|---:|
| CRP | 4 |
| MONO | 4 |
| %MONO | 3 |
| HCT, MPV, `patient_age` | po 3 |

**Dlaczego tej selekcji nie utrzymujemy:**
1. **Użyto etykiety na całym zbiorze**, czyli także na pacjentach, którzy później trafiają do testu. To formalny wyciek: decyzja o cechach „widziała” wynik oceny. Realny wpływ na wyniki był jednak znikomy, bo usunięte cechy miały korelację z etykietą |ρ| ≤ 0,02. Zmiana porządkuje metodę, a nie naprawia błąd w czerwcowych wynikach.
2. **W PU etykieta oznacza kohortę, nie chorobę.** Kryterium „słaba korelacja z etykietą” usuwa cechy, które nie odróżniają KOR od NEURO, a zostawia te, które odróżniają, łącznie z różnicami epoki i sposobu pozyskania danych. To odwrotność tego, czego potrzebujemy.
3. **Korelacja jednowymiarowa nie mierzy przydatności cechy** w modelu wielowymiarowym.
4. **Wybór na granicy był przypadkowy.** Trzecie miejsce to remis czterech cech po 3 głosy, rozstrzygnięty sortowaniem alfabetycznym („%” jest przed literami). Ta sama procedura z inną kolejnością usunęłaby HCT, MPV albo wiek pacjenta.

**Co mówią same cechy, niezależnie od etykiety** (`analiza_selekcji_cech.json`, sekcja 5):
- %MONO jest **w pełni redundantne**: równa się 100 − (%NEUT + %LYMPH + %EO + %BAZO), korelacja 0,9995, mediana różnicy 0,0 pp.
- MONO jest **prawie redundantne**: MONO ≈ %MONO × WBC / 100, mediana błędu względnego 0,4%.
- CRP **nie jest redundantne i nie odróżnia kohort**: mediana 10,7 mg/l w KOR i 12,1 w NEURO, Spearman z etykietą 0,008. Brakuje go w 18,6% rekordów KOR i 47,7% NEURO, ale podobną asymetrię braków mają cechy, których nikt nie proponował usuwać (NRBC, NEUT).

**Decyzja (04.10.2026): jeden zestaw, 38 cech.** Selekcja z etapu 1 jest cofnięta, a ewentualny wybór cech przechodzi do modelowania, gdzie może odbywać się wewnątrz walidacji krzyżowej. Za takim wyborem przemawiają też dwie rzeczy praktyczne:
- benchmark imputacji z etapu 2 był liczony na 38 cechach, więc parametry MICE pasują do tego zestawu bez ponownego benchmarku,
- jednorazowe porównanie nie wykazało istotnej różnicy w jakości imputacji (sekcja 4.5).

Dokument czerwcowy (`0626_PODSUMOWANIE_RAPORT_PRZEJSCIOWY.md`) już wtedy zastrzegał, że przed analizą potwierdzającą trzeba albo wrócić do 38 cech, albo przenieść selekcję do walidacji krzyżowej. Ta decyzja domyka to zastrzeżenie.

Redundancja MONO i %MONO zostaje. Procenty rozmazu sumują się do 100%, więc w modelach liniowych są dokładnie współliniowe. W modelowaniu trzeba to obsłużyć regularyzacją albo pominięciem jednego procentu; modelom drzewiastym to nie przeszkadza. CRP ma 48% braków w NEURO i zawyżone uzupełnienia (sekcja 5.1), więc jego wpływ sprawdzi analiza wrażliwości na cechach o niskim odsetku braków.

---

## 3. Część B: podział na foldy

**Kod:** `sgkf/podzial.py` · **testy:** `sgkf/tests/test_podzial.py`

### 3.1 Zasada

| Warunek | Co zapewnia | Jak |
|---|---|---|
| **Group** | wszystkie rekordy pacjenta trafiają albo do train, albo do test | podział na tabeli pacjentów (jeden wiersz = jedna osoba) |
| **Stratified** | podobne proporcje w każdym foldzie | `StratifiedKFold` po kluczu **etykieta × okno czasowe KOR × liczba rekordów** (12 warstw) |
| **Zamrożony** | ten sam podział na każdym komputerze i przy każdej wersji bibliotek | `shuffle=True`, seed 42, przydział zapisany do `pacjent_fold.csv` z odciskiem pliku wejściowego |

Pacjent jest „w oknie”, jeśli co najmniej połowa jego rekordów leży we wspólnym oknie dat obu kohort (2019-12-29 – 2021-12-26). Klasy liczby rekordów to 1 / 2 / 3–4 / 5+. Warstwa mniejsza niż 20 pacjentów byłaby scalana do `<etykieta>_inne`; najmniejsza liczy 24 osoby, więc scalenie nie zachodzi.

Etykieta, okno i liczba rekordów służą tylko do **konstrukcji** podziału. Żadna z nich nie jest cechą modelu.

### 3.2 Audyt wersji 1 (11.06.2026)

| # | Problem | Skutek |
|---|---|---|
| 1 | `StratifiedGroupKFold(n_splits=5)` bez `shuffle` | podział zależał od kolejności wierszy w CSV (plik posortowany: najpierw KOR, potem NEURO); `RANDOM_STATE = 42` działał tylko w MICE |
| 2 | przydział nie był zapisywany | każdy etap liczyłby podział od nowa; przesortowanie pliku albo inna wersja sklearn po cichu zmienia foldy |
| 3 | `shuffle=True` w SGKF zależy od wersji sklearn | ta sama konfiguracja (seed 42, dane z 09.2026) daje rozstęp udziału pozytywnych 0,11 pp w sklearn 1.8 i 0,80 pp w sklearn 1.6, czyli **inne podziały na dwóch komputerach zespołu**; `StratifiedKFold` na pacjentach odtworzył się w 100% |
| 4 | stratyfikacja tylko po etykiecie rekordu | foldy niewyrównane pod względem epoki badania i liczby rekordów, dwóch znanych źródeł fałszywego sygnału |
| 5 | 63 pacjentów w obu kohortach | etykieta pacjenta nieokreślona (rozwiązane w części A) |
| 6 | notebook v1: `X = df.drop(columns=['label'])` | w `X` zostają `patient_id`, `custom_id`, `examination_date`; nie wpływa na podział, ale `X` nie nadaje się wprost do modelu |

Wersja 1 była poprawna co do najważniejszego: grupowała po pacjencie i imputowała wewnątrz foldu. Problemy dotyczyły powtarzalności i balansu.

### 3.3 Porównanie strategii

Dane po przygotowaniu (40 924 pacjentów, 1 823 pozytywnych), 5 foldów. Rozstęp to różnica między najbardziej a najmniej obciążonym foldem. Dla strategii tasowanych podany jest **najgorszy** wynik z 5 ziaren (42, 7, 13, 101, 2024). Źródło: `sgkf/results/podzial_porownanie_strategii.csv`.

| Strategia | Udział poz. (pacjenci) | Udział poz. (rekordy) | Rozmiar foldu (rekordy) | NEURO z okna KOR | NEURO z 5+ rekordami |
|---|---:|---:|---:|---:|---:|
| SGKF rekordowy, `shuffle=False` (v1) | 0,02 pp | 0,01 pp | 0,01% | **4,06 pp** | 0,27 pp |
| SGKF rekordowy, `shuffle=True` (sklearn 1.6) | 0,66 pp | 2,82 pp | 4,43% | 5,01 pp | 5,64 pp |
| pacjenci, stratyfikacja: etykieta (rekomendacja z 09.2026) | 0,01 pp | 2,04 pp | 4,12% | **4,91 pp** | **8,49 pp** |
| **pacjenci, etykieta × okno × liczba rekordów (v2)** | **0,01 pp** | 1,15 pp | 2,36% | **0,55 pp** | **0,22 pp** |

Wnioski:
1. Wersja 2 jest jedyną strategią, która jednocześnie wyrównuje klasy i oba czynniki zakłócające, przy każdym z 5 ziaren.
2. Stratyfikacja tylko po etykiecie (rekomendacja z września) idealnie wyrównuje klasy, ale przy niektórych ziarnach rozjeżdża foldy do 5 pp na epoce i 8 pp na liczbie rekordów. Siatka stabilności z września (`sgkf/etap0_sgkf_stabilnosc.py`) tego nie wychwyciła, bo mierzyła udział lat 2020–2021 we **wszystkich** pacjentach foldu, z których 96% to KOR. Problem siedział w podgrupie NEURO.
3. Wersja 1 wypada dobrze na rekordach, bo SGKF optymalizuje właśnie proporcję rekordów. Przy ocenie pacjentowej ważniejsza jest proporcja pacjentów. Rozstęp 1,15 pp na rekordach w v2 wynika z różnej liczby rekordów na pacjenta i nie przenosi się na metryki pacjentowe.

### 3.4 Zamrożony podział (v2, seed 42)

`sgkf/results/podzial_diagnostyka.csv`:

| Fold | Pacjenci | Rekordy | Pozytywni | Udział poz. | NEURO z okna KOR | NEURO z 5+ rek. | Mediana roku NEURO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 185 | 15 722 | 364 | 4,45% | 8,5% | 20,6% | 2019 |
| 1 | 8 185 | 15 540 | 365 | 4,46% | 8,5% | 20,8% | 2019 |
| 2 | 8 185 | 15 601 | 365 | 4,46% | 9,0% | 20,8% | 2018 |
| 3 | 8 185 | 15 608 | 365 | 4,46% | 9,0% | 20,8% | 2019 |
| 4 | 8 184 | 15 581 | 364 | 4,45% | 8,5% | 20,6% | 2018 |

Pacjentów, którzy byli w obu kohortach (63), nie dodano do klucza stratyfikacji, bo dałoby to warstwy po kilka osób. Ich listę zawiera spis części A.

### 3.5 Odtwarzalność

- `podzial_meta.json` przechowuje konfigurację, odcisk SHA-256 pliku wejściowego (`5b822c09ebc03011`), metadane części A, wersje bibliotek i datę.
- `wczytaj_podzial()` przelicza odcisk i **odmawia działania**, jeśli plik wejściowy się zmienił. Każda zmiana w części A (reguły, dane źródłowe) unieważnia więc zapisany podział i wymusza świadome przebudowanie.
- Odcisk liczony jest po ujednoliceniu końców linii, więc ten sam podział wczytuje się na Windowsie (CRLF po checkoucie) i na macOS.

---

## 4. Część B: imputacja MICE wewnątrz foldów

**Kod:** `sgkf/aneurysm_sgkf_mice_pipeline.py` · **porównanie wariantów:** `sgkf/porownanie_cech.py`

### 4.1 Procedura

```
dla każdego z 5 foldów (przydział z pacjent_fold.csv):
  1. split → X_train, X_test (z brakami)
  2. MinMaxScaler.fit()           ← tylko kompletne wiersze X_train
  3. IterativeImputer.fit()       ← tylko X_train (po skalowaniu)
  4. transform(X_train), transform(X_test)
  5. scaler.inverse_transform()   ← powrót do oryginalnych jednostek
  6. kontrole: 0 wspólnych pacjentów, 0 NaN, 0 wartości ujemnych
po pętli: każdy pacjent dokładnie raz w części testowej
```

Parametry MICE wybrano w benchmarku imputacji (`2-imputation/RAPORT_IMPUTACJA.md`, przebieg 4 + walidacja cross-param):

| Parametr | Wartość |
|---|---|
| `estimator` | `ExtraTreesRegressor` |
| `max_iter` / `n_estimators` / `max_depth` | 7 / 78 / 10 |
| `min_value` / `max_value` | 0 / 1 (skala po MinMaxScaler) |
| `random_state` | 42 |

Uwagi metodologiczne:
- **Imputer nie widzi etykiety.** Uzupełnianie korzysta wyłącznie z cech, więc w teście nie wymaga znajomości `label`.
- **Scaler na kompletnych wierszach**, zgodnie z benchmarkiem. Poza zakresem kompletnych wierszy leży 209 obserwowanych wartości (0,008%), więc różnica względem skalowania na wszystkich wartościach jest pomijalna. `min/max_value` ograniczają tylko wartości **uzupełniane**, obserwowane zostają nietknięte.
- **Imputacja deterministyczna** (`sample_posterior=False`), a nie *multiple imputation*. Niepewność uzupełnienia nie jest propagowana.
- **Rekord imputowany jest niezależnie od innych rekordów tego samego pacjenta.** Nie ma przepływu informacji między wierszami testowymi.

### 4.2 Wyniki pełnego przebiegu

`python 3-sgkf-split/uruchom_sgkf.py`, 04.10.2026: wszystkie kroki OK (Mac M-series, Python 3.9.6, sklearn 1.6.1). Logi każdego kroku: `przygotowanie/results/logi/`, `sgkf/results/logi/`.

**Kontrole w każdym foldzie:** 0 wspólnych pacjentów train/test, 0 braków i 0 wartości ujemnych po imputacji; każdy z 40 924 pacjentów trafił do części testowej dokładnie raz.

`sgkf/results/przebieg_imputacji_38.json`, łącznie 15,6 min:

| Fold | Rekordy train / test | Pacjenci test | Udział `label=1` w teście (rekordy) | Uzupełnione komórki w teście: KOR / NEURO | Czas | Odcisk wyniku |
|---:|---|---:|---:|---|---:|---|
| 0 | 62 330 / 15 722 | 8 185 | 9,18% | 12,1% / 27,5% | 2,9 min | `cf84e0d1d6e15e45` |
| 1 | 62 512 / 15 540 | 8 185 | 8,07% | 12,3% / 25,5% | 3,2 min | `a42c7f1088cff2eb` |
| 2 | 62 451 / 15 601 | 8 185 | 8,21% | 12,3% / 25,8% | 3,1 min | `71a04bd6f5aae1f7` |
| 3 | 62 444 / 15 608 | 8 185 | 8,70% | 12,3% / 25,3% | 3,2 min | `b51c96cf1e45b725` |
| 4 | 62 471 / 15 581 | 8 184 | 8,43% | 12,5% / 25,8% | 3,2 min | `3712e06c85c32b52` |

**Powtarzalność.** Dwa niezależne pełne przebiegi z 04.10.2026 (przed i po usunięciu kolumny meta) dały **identyczne odciski wyników we wszystkich 5 foldach**. Imputacja jest więc deterministyczna na danym środowisku. Na innym komputerze lub przy innej wersji sklearn porównanie odcisków od razu pokaże ewentualną różnicę.

Udział `label=1` liczony na **rekordach** waha się między foldami (8,1–9,2%), bo pacjenci NEURO mają różną liczbę rekordów. Na **pacjentach**, czyli tam, gdzie będą liczone metryki, foldy są wyrównane do 0,01 pp (sekcja 3.4).

Zaimputowane foldy mają 46 MB w formacie parquet: `sgkf/results/foldy_imputowane_38/fold{0..4}_{train,test}.parquet`. Każdy plik zawiera kolumny meta (`patient_id`, `custom_id`, `examination_date`, `label`) i 38 cech po imputacji, w oryginalnych jednostkach. Pliki są poza gitem; na innym komputerze odtwarza je `uruchom_sgkf.py`.

### 4.3 Jakość imputacji: protokół z etapu 2 w foldach

**Kod:** `sgkf/ocena_imputacji.py` · **wynik:** `sgkf/results/ocena_imputacji.json`

Błąd imputacji da się zmierzyć tylko na wartościach, które znamy. Dlatego, jak w benchmarku z etapu 2, ukrywamy część wartości w kompletnych wierszach i porównujemy uzupełnienia z prawdą. Funkcje maskowania i metryk są importowane wprost z `2-imputation/final/evaluate.py`. Różnica względem etapu 2: ocena jest **out-of-fold**. Scaler i MICE dopasowano na treningu foldu dokładnie jak w produkcji, a oceniane są kompletne wiersze części testowej, czyli pacjenci niewidziani przy dopasowaniu: około 4 000 wierszy KOR i 142–176 wierszy NEURO na fold.

Dwie maski:
- **etap 2:** losowe 10% komórek (seed 42), czyli dokładnie protokół benchmarku,
- **realistyczne wzorce:** każdy kompletny wiersz dostaje wzorzec braków losowo wybranego niekompletnego wiersza tej samej kohorty z treningu. Braki idą wtedy całymi panelami, jak w prawdziwych danych. Ukrytych jest średnio 17% komórek w KOR i 29–34% w NEURO.

Obok RMSE podajemy **stosunek do uzupełniania średnią z treningu** na tych samych maskach. Jest niezależny od skali; 1,0 oznacza, że MICE nie jest lepsze od średniej.

| Maska | Kohorta | RMSE [0,1] (5 foldów) | MAE | RMSE względem średniej | Etap 2 (MICE, parametry produkcyjne) |
|---|---|---:|---:|---:|---:|
| losowe 10% | KOR | 0,0829 ± 0,0019 | 0,031 | **0,61** | 0,0861 |
| losowe 10% | NEURO | 0,0801 ± 0,0078 | 0,031 | **0,65** | 0,0984 |
| realistyczne wzorce | KOR | 0,1080 ± 0,0014 | 0,060 | **0,98** | — |
| realistyczne wzorce | NEURO | 0,0792 ± 0,0058 | 0,045 | **0,89** | — |

**Wnioski:**
1. **MICE w ustawieniu produkcyjnym działa co najmniej tak dobrze jak w benchmarku.** Dla KOR liczby są wprost porównywalne: 0,083 wobec 0,086. Dla NEURO RMSE wyszło niższe niż w benchmarku, ale nie jest wprost porównywalne. W etapie 2 NEURO miało własny skaler z 780 wierszy, a tu skaler pochodzi z treningu, w ~96% z KOR. Miara względna pokazuje, że **NEURO jest uzupełniane praktycznie tak samo dobrze jak KOR** (0,65 wobec 0,61).
2. **Przy brakach całymi panelami imputacja prawie nic nie wnosi**: w KOR jest o 2% lepsza od średniej, w NEURO o 11%. Gdy brakuje całego panelu, pozostałe cechy niosą o nim mało informacji i uzupełnienie jest bliskie średniej z treningu. Dotyczy to zwłaszcza koagulogramu (PT, INR, APTT, WAPTT) i glukozy, których brakuje w około połowie rekordów obu kohort. **Benchmark z etapu 2, maskujący losowe komórki, przeszacowywał więc użyteczność imputacji dla tych danych.** Skutki opisuje sekcja 5.1.
3. Najtrudniejsze dla NEURO w protokole z etapu 2 są płeć i wiek, co jest artefaktem protokołu, bo w prawdziwych danych tych cech nigdy nie brakuje. Dalej MPV, CRP i eGFR. Przy realistycznych wzorcach najtrudniejsze są eGFR, MPV i parametry czerwonokrwinkowe (HGB, RBC, HCT).
4. Wyniki NEURO są bardziej zmienne między foldami (odch. 0,008), bo opierają się na 142–176 kompletnych wierszach na fold. KL dla NEURO (0,06 przy maskach losowych) jest przy tej liczbie wierszy mało wiarygodne.

### 4.4 Asymetria braków między kohortami

`analiza_decyzji.json`, sekcja G, wariant 38 cech po przygotowaniu:

| | KOR | NEURO |
|---|---:|---:|
| komórki cech z brakiem (uzupełniane przez MICE) | 12,3% | **26,0%** |
| kompletne wiersze (podstawa skalera) | 20 383 | 772 |
| udział KOR wśród kompletnych wierszy | 96,4% | |
| braki NRBC / %NRBC | 2,6% | 47,7% |
| braki NEUT / %NEUT | 2,9% | 42,3% |
| braki CRP | 18,6% | 47,7% |
| braki eGFR (oba wzory) | 17,2% | 41,7% |

Około jednej czwartej profilu NEURO odtwarza imputer, który uczy się na treningu w 91% złożonym z rekordów KOR. Możliwe skutki idą w przeciwnych kierunkach:
- wartości uzupełnione w NEURO mogą być „podobne do KOR” i zacierać różnice,
- albo sam wzorzec braków może zostawić ślad odróżniający kohorty.

**Decyzja (04.10.2026): bez korekty, opisujemy jako ograniczenie.** Udział uzupełnionych komórek per kohorta jest raportowany w każdym foldzie. Jako analiza wrażliwości w modelowaniu: model na cechach o niskim odsetku braków w obu kohortach.

### 4.5 Jednorazowe porównanie 38 vs 35 cech (podkładka pod decyzję)

**Po co.** Zanim zapadła decyzja o jednym zestawie cech (sekcja 2.6), oba warianty policzono na tym samym podziale i porównano ich wpływ na imputację. Porównanie nie jest częścią standardowego przebiegu. Wyniki i logi są w `sgkf/results/porownanie_38_35/`, a polecenia do odtworzenia w `sgkf/porownanie_cech.py`. Przebieg z 04.10.2026: MICE na 35 cechach 13,5 min, te same kontrole spełnione w każdym foldzie.

**Co naprawdę różni warianty.** MONO i %MONO są prawie w całości wyliczalne z innych cech: %NEUT + %LYMPH + %EO + %BAZO + %MONO = 100,0% (odch. std. 0,11), a MONO ≈ %MONO × WBC / 100 (mediana błędu względnego 0,4%). Realną nową informację wnosi tylko **CRP**.

**Test kontrolowany:** kompletne wiersze treningu foldu 0 (16 889, w tym 630 NEURO), maska 10% komórek wyłącznie w 35 wspólnych cechach (59 011 komórek), MICE z parametrami produkcyjnymi. RMSE i MAE w skali [0,1]:

| | RMSE | RMSE KOR | RMSE NEURO | MAE |
|---|---:|---:|---:|---:|
| 38 cech | **0,0846** | 0,0849 | 0,0750 | 0,0312 |
| 35 cech | 0,0859 | 0,0863 | 0,0760 | 0,0320 |

Najbardziej zyskują %NEUT (RMSE 0,020 wobec 0,032), %LYMPH (0,020 wobec 0,029), %EO i WBC, czyli cechy związane z %MONO przez sumę rozmazu. Ten test **faworyzuje wariant 38**, bo dodatkowe kolumny nigdy nie są w nim maskowane. W prawdziwych danych %MONO brakuje zawsze, gdy brakuje %LYMPH, i w 59% przypadków, gdy brakuje %NEUT, bo to ten sam panel badań.

**Faktyczne foldy:** porównanie wartości uzupełnionych w obu wariantach, w odchyleniach standardowych cechy:

| | Średnia różnica | Mediana różnicy | Komórki z różnicą > 0,5 SD |
|---|---:|---:|---:|
| części testowe, 5 foldów | 0,11 SD | 0,03–0,05 SD | 2,8% |
| w tym KOR / NEURO | 0,11–0,13 / 0,06–0,07 SD | | |

Wartości obserwowane są w obu wariantach identyczne; różnią się tylko uzupełnienia.

**Wniosek.** Wybór 38 czy 35 cech **nie ma istotnego wpływu na imputację**. Drobna przewaga 38 w teście kontrolowanym wynika częściowo z jego konstrukcji. O **jednym zestawie 38 cech** przesądza więc ocena samej selekcji (sekcja 2.6), a nie jakość imputacji.

---

## 5. Analiza wyników: na czym model mógłby się „przejechać”

**Kod:** `sgkf/analiza_wynikow.py` · **wynik:** `sgkf/results/analiza_wynikow.json`

To nie jest model ryzyka tętniaka, tylko diagnostyka zbioru przygotowanego do modelowania. Sprawdza, jakie różnice między kohortami istnieją niezależnie od choroby i mogłyby posłużyć modelowi za „skrót”.

### 5.1 Jak wyglądają uzupełnione wartości

- **Uzupełnienia brakujących paneli skupiają się wokół średniej z treningu, tak samo w obu kohortach.** APTT: uzupełniona mediana 32,7 wobec średniej zmierzonych 32,5 (mediana zmierzonych 30,0). GLU: 120,8 wobec średniej 124,7 (mediana 107). CRP: 26,6 wobec średniej 41,7 (mediana 10,8). Przy skośnych rozkładach (CRP, GLU) uzupełnienia są więc **systematycznie wyższe niż typowa zmierzona wartość**. To bezpośredni skutek tego, że przy brakach panelowych imputacja ≈ średnia (sekcja 4.3).
- **Podpis imputacji NRBC i %NRBC.** Zmierzone NRBC wynosi dokładnie 0 w 77,6% rekordów. Uzupełnione nigdy nie jest zerem: 99,5% uzupełnień to wartości z przedziału (0; 0,05). NRBC brakuje w 47,7% rekordów NEURO i w 2,6% KOR. „Mała niezerowa wartość NRBC” oznacza więc w praktyce „rekord NEURO”, a model drzewiasty łatwo to wykorzysta. Tak samo zachowuje się %NRBC.
- **Imputacja jest stabilna między foldami.** Mediany uzupełnień różnią się między foldami o ułamek jednostki (APTT o 1,1–1,2 s, GLU o 1,3–2,1 mg/dl).

### 5.2 Diagnostyka skrótów

Na zamrożonych foldach (HistGradientBoosting, ocena na części testowej każdego foldu) mierzymy, jak dobrze da się odróżnić grupy, które nie powinny być odróżnialne, gdyby dane różniły się wyłącznie chorobą:

| Test | AUC (rekordy) | AUC (pacjenci) |
|---|---:|---:|
| B1 NEURO vs KOR, 38 cech po imputacji | 0,906 (0,891–0,916) | **0,928** |
| B2 NEURO vs KOR, **tylko wzorzec braków** (które badania zlecono) | 0,819 | **0,823** |
| B3 NEURO vs KOR, tylko rekordy ze wspólnego okna dat | 0,793 (0,741–0,837) | 0,801 |
| B4 **NEURO z lat 2020–2021 vs NEURO z innych lat** (ta sama choroba) | **0,832** (0,807–0,854) | — |

Wnioski:
1. **Kohorty są bardzo łatwe do odróżnienia (0,93 na pacjentach).** To nie jest miara wykrywania tętniaka, bo etykieta oznacza kohortę. Wysokie AUC P-vs-U w modelowaniu będzie się należało przede wszystkim temu.
2. **Sam wzorzec braków odróżnia kohorty z AUC 0,82.** Informacja o tym, *które* badania zlecono, jest silnym sygnałem kohorty. Imputacja jej nie usuwa: zostaje w podpisie NRBC i w uzupełnieniach równych średniej.
3. **W tym samym okresie rozdzielność spada do 0,80, ale zostaje wysoka.** Różnica epok podnosi rozdzielność (0,93 wobec 0,80), lecz jej nie tłumaczy w całości.
4. **Epokę widać w wynikach samych chorych (AUC 0,83).** Rozjazd czasowy nie jest więc problemem teoretycznym: wyniki laboratoryjne niosą wyraźny ślad okresu, z którego pochodzą. Decyzja „bez korekty” (sekcja 2.1) wymaga w modelowaniu przynajmniej analizy wrażliwości we wspólnym oknie dat.

### 5.3 Profil kohort względem zakresów referencyjnych

Odsetek rekordów poza typowym zakresem referencyjnym dla dorosłych, wśród wartości zmierzonych:

| Parametr | Granica | KOR | NEURO |
|---|---|---:|---:|
| CRP | > 5 mg/l | **61,4%** | 62,2% |
| WBC | > 10 G/l | 36,9% | 32,5% |
| NEUT | > 7 G/l | 39,5% | 33,9% |
| GLU | > 99 mg/dl | 64,0% | 57,7% |
| HGB | < 12 g/dl | 39,6% | 43,2% |
| PLT | < 150 G/l | 12,1% | 7,3% |

KOR ma profil zapalny i „szpitalny”, w części parametrów nawet bardziej odbiegający od normy niż NEURO. **KOR nie wygląda na populację ogólną ani przesiewową, tylko raczej na pacjentów szpitalnych z lat 2020–2021**, czyli z okresu pandemii. Zastrzeżenie: to średnie tygodniowe z badań zlecanych z powodów klinicznych. Jeśli się to potwierdzi, zmienia interpretację klasy U i całego zastosowania „przesiewowego”. To pytanie do prowadzącego lub właściciela danych.

---

## 6. Przegląd merytoryczny: co sprawdzono

| Obszar | Sprawdzenie | Wynik |
|---|---|---|
| wyciek przez pacjenta | wspólni pacjenci train/test w każdym foldzie; każdy pacjent raz w teście | 0; spełnione (asercje w pipeline, testy) |
| wyciek przez preprocessing | scaler i MICE dopasowywane tylko na treningu foldu | spełnione (struktura `prepare_fold`) |
| wyciek przez etykietę | etykieta nie jest cechą imputera ani modelu; używana tylko do stratyfikacji | spełnione |
| wyciek przez selekcję cech | CRP, MONO, %MONO usunięte w etapie 1 na podstawie korelacji z etykietą na całym zbiorze | selekcja cofnięta, jeden zestaw 38 cech (sekcja 2.6); wpływ na imputację zmierzony (sekcja 4.5) |
| powtarzalność podziału | ziarno działa, wynik nie zależy od kolejności wierszy ani od wersji sklearn | testy `test_podzial_nie_zalezy_od_kolejnosci_wierszy`, `test_ziarno_*`; przydział zapisany |
| zmiana danych po zamrożeniu | odcisk pliku wejściowego | `test_zmiana_danych_uniewaznia_podzial` |
| spójność przygotowania | plik w repozytorium = wynik kodu na danych źródłowych | `test_plik_w_repozytorium_zgodny_z_kodem` |
| jakość danych | unikalność `custom_id`, zakresy, brak ujemnych, płeć i wiek bez braków | spełnione (sekcja 2.5) |
| powtarzalność imputacji | ponowny przebieg na tym samym komputerze daje identyczne odciski wyników | sekcja 4.2 |
| jakość imputacji w produkcji | protokół z etapu 2 out-of-fold + realistyczne wzorce braków | jak w benchmarku przy losowych brakach; przy brakach panelowych ≈ średnia (sekcja 4.3) |
| skróty dla modelu | rozdzielność kohort na cechach, na wzorcu braków, we wspólnym oknie; epoka w NEURO | AUC 0,93 / 0,82 / 0,80 / 0,83 (sekcja 5.2) |
| redundancja cech | %NEUT + %LYMPH + %EO + %BAZO + %MONO = 100,0 (odch. 0,11); MONO ≈ %MONO × WBC / 100 (mediana błędu 0,4%) | MONO i %MONO zostają: pomagają imputacji; kolinearność do obsłużenia w modelowaniu |
| cenzurowanie eGFR | eGFR-MDRD = 60 w 56% rekordów, eGFR-CKD = 90 w 33% | laboratorium raportuje „≥ 60” / „≥ 90”; reguła KREA (eGFR ≥ 30) działa mimo cenzury; do opisu |
| zgodność między komputerami | odcisk wartości po imputacji w każdym foldzie (`przebieg_imputacji_*.json`) | pozwala sprawdzić, czy Windows (sklearn 1.8) liczy to samo co Mac (sklearn 1.6) |

---

## 7. Ograniczenia tego etapu

1. **Rozjazd czasowy kohort** (99,5% KOR to lata 2020–2021, NEURO to lata 2000–2024) nie jest korygowany. Stratyfikacja wyrównuje go między foldami, ale model nadal może rozpoznawać epokę.
2. **Asymetria braków** (NEURO 26%, KOR 12% komórek uzupełnianych) i **asymetryczne czyszczenie kwantylowe z etapu 1** (NEURO 3,2%, KOR 0,8% usuniętych rekordów).
3. **Imputacja deterministyczna**, oceniana w benchmarku na kompletnych wierszach; mechanizm braków (MCAR/MAR/MNAR) niezbadany.
4. **Reguły wartości niemożliwych są zachowawcze.** Usuwają tylko oczywiste przypadki; pojedyncze błędy ukryte w średniej tygodniowej zostają.
5. **eGFR jest cenzurowane** (60 / 90), co ogranicza jego informatywność w górnym zakresie.
6. **Zaimputowane foldy są poza gitem** (rozmiar). Odtwarza je jedno polecenie, a ewentualne różnice między wersjami sklearn wykrywa odcisk wyniku.
7. **Przy brakach całymi panelami uzupełnienia ≈ średnia z treningu** (sekcja 4.3). Nie niosą informacji o pacjencie, a przy skośnych rozkładach są zawyżone (CRP, GLU).
8. **Podpis imputacji NRBC/%NRBC**: uzupełnione wartości są rozpoznawalne i w praktyce oznaczają NEURO (sekcja 5.1).
9. **Kohorty różnią się wieloma rzeczami poza chorobą.** Sam wzorzec braków daje AUC 0,82, a epokę widać w wynikach NEURO z AUC 0,83 (sekcja 5.2). Profil KOR sugeruje populację szpitalną, a nie ogólną (sekcja 5.3).

---

## 8. Pliki i uruchomienie

```
3-sgkf-split/
├── RAPORT_SGKF_MICE.md               ten raport
├── uruchom_sgkf.py                   cały etap: część A, potem B
├── przygotowanie/                    CZĘŚĆ A
│   ├── przygotuj_dane.py             decyzje → data/processed/aneurysm_sgkf_input.csv + spis zmian
│   ├── analiza_decyzji.py            podkładka liczbowa pod decyzje
│   ├── analiza_selekcji_cech.py      dlaczego 38 cech (odtworzenie selekcji z etapu 1)
│   ├── etap0_diagnostyka.py          diagnostyka z 09.2026 (stan przed decyzjami)
│   ├── tests/test_przygotowanie.py   6 testów
│   └── results/                      usuniete_rekordy.csv, usuniete_wartosci.csv, podsumowanie.json,
│                                     analiza_decyzji.json, analiza_selekcji_cech.json, logi/
├── sgkf/                             CZĘŚĆ B
│   ├── podzial.py                    zamrożony przydział pacjent → fold
│   ├── aneurysm_sgkf_mice_pipeline.py  MICE wewnątrz foldów (--zapisz)
│   ├── ocena_imputacji.py            jakość imputacji protokołem z etapu 2 (w foldach)
│   ├── analiza_wynikow.py            diagnostyka: uzupełnienia, skróty, profil kohort
│   ├── porownanie_cech.py            jednorazowe porównanie 38 / 35 (podkładka pod decyzję)
│   ├── etap0_sgkf_stabilnosc.py      siatka stabilności z 09.2026
│   ├── tests/test_podzial.py         10 testów
│   └── results/                      pacjent_fold.csv, podzial_meta.json, podzial_diagnostyka.csv,
│                                     podzial_porownanie_strategii.csv, przebieg_imputacji_38.json,
│                                     ocena_imputacji.json, analiza_wynikow.json, logi/,
│                                     porownanie_38_35/, foldy_imputowane_38/ (poza gitem)
└── archiwum/
    └── aneurysm_data_StratifiedGroupKFold.ipynb   wersja 1 (Liwia, 08.06.2026)
```

```bash
python 3-sgkf-split/uruchom_sgkf.py                      # całość (~35 min)
python 3-sgkf-split/uruchom_sgkf.py --czesc A            # tylko przygotowanie danych (sekundy)
python 3-sgkf-split/uruchom_sgkf.py --czesc B --bez-mice # tylko podział i testy
python 3-sgkf-split/sgkf/podzial.py --sprawdz            # czy zapisany podział pasuje do danych
```
