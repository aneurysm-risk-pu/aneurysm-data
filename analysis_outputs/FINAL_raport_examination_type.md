# Analiza zleconych badań (`examination_type`) między grupami KOR i NEURO [FINAL]

## 1. Podsumowanie pokrycia nazw zleceń

| Kategoria | Liczba unikalnych nazw | Udział (%) |
| :--- | :---: | :---: |
| `BOTH` | 47 | 68.1% |
| `ONLY_KOR` | 16 | 23.2% |
| `ONLY_NEURO` | 6 | 8.7% |
| **RAZEM** | **69** | **100.0%** |

## 2. Wykryte synonimy (różne nazwy oznaczające to samo badanie)

### A. Synonimy potwierdzone identyczną sygnaturą mierzonych parametrów (>80% zgodności)

- **Parametry `[eGFR-MDRD, eGFRCKD, exam_list]`**: `Antystreptolizyna O (ASO)` (ONLY_KOR), `Bilirubina całkowita we krwi` (BOTH), `Wapń całkowity we krwi` (BOTH)
- **Parametry `[K, Na, exam_list]`**: `Elektrolity` (BOTH), `Elektrolity (potas, sód)` (BOTH)
- **Parametry `[KREA, exam_list]`**: `Kreatynina` (ONLY_NEURO), `Kreatynina w moczu` (BOTH), `Wydalanie z moczem` (BOTH)
- **Parametry `[%BAZO, %EO, %IG, %LYMPH, %MONO, %NEUT, %NRBC, BAZO, EO, HCT, HGB, IG, LYMPH, MCH, MCHC, MCV, MONO, MPV, NEUT, NRBC, P-LCR, PLT, RBC, RDW, WBC, exam_list]`**: `Morfologia` (BOTH), `Morfologia z retikulocytami` (BOTH)

### B. Kandydaci leksykalni (podobna nazwa po normalizacji)

- **Rdzeń `albumina`**: `Albumina` (BOTH), `Albumina w moczu` (ONLY_KOR), `Albumina w surowicy` (BOTH)
- **Rdzeń `azot mocznika`**: `Azot Mocznika` (ONLY_NEURO), `Azot mocznika w moczu` (BOTH), `Azot mocznika we krwi` (BOTH)
- **Rdzeń `beta-2-mikroglobulina`**: `Beta-2-mikroglobulina` (BOTH), `Beta-2-mikroglobulina we krwi` (ONLY_KOR)
- **Rdzeń `chlorki`**: `Chlorki (Cl)` (BOTH), `Chlorki we krwi` (BOTH)
- **Rdzeń `elektrolity`**: `Elektrolity` (BOTH), `Elektrolity (potas, sód)` (BOTH)
- **Rdzeń `kreatynina`**: `Kreatynina` (ONLY_NEURO), `Kreatynina w moczu` (BOTH), `Kreatynina we krwi` (BOTH)
- **Rdzeń `kwas moczowy`**: `Kwas moczowy` (BOTH), `Kwas moczowy we krwi` (BOTH)
- **Rdzeń `magnez`**: `Magnez (Mg)` (BOTH), `Magnez we krwi` (BOTH)
- **Rdzeń `potas`**: `Potas` (ONLY_NEURO), `Potas (K) w moczu` (BOTH), `Potas w moczu` (BOTH)
- **Rdzeń `przeciwciała przeciwko cytrulinowanym peptydom`**: `Przeciwciała przeciwko cytrulinowanym peptydom` (ONLY_KOR), `Przeciwciała przeciwko cytrulinowanym peptydom (anty-CCP)` (ONLY_KOR)
- **Rdzeń `sod`**: `Sód` (ONLY_NEURO), `Sód (Na) w moczu` (BOTH), `Sód w moczu` (BOTH)
- **Rdzeń `wskaznik filtracji kłebuszkowej`**: `Wskaźnik filtracji kłębuszkowej (wg CKD-EPI)` (BOTH), `Wskaźnik filtracji kłębuszkowej (wg MDRD)` (BOTH)

## 3A. Badania zlecane WYŁĄCZNIE w grupie NEURO (`ONLY_NEURO`) — 6 szt.

| Lp. | Nazwa w `examination_type` | KOR (liczba) | Lata KOR | NEURO (liczba) | Lata NEURO | Mierzone parametry (single) |
| :---: | :--- | :---: | :--- | :---: | :--- | :--- |
| 1 | `Kreatynina` | 0 | - | 1 114 | 2010–2013 (2010  2011  2012  2013) | `KREA  exam_list` |
| 2 | `Azot Mocznika` | 0 | - | 926 | 2010–2013 (2010  2011  2012  2013) | `-` |
| 3 | `Rozmaz mikroskopowy krwi obwodowej` | 0 | - | 41 | 2012–2016 (2012  2013  2014  2015  2016) | `-` |
| 4 | `Potas` | 0 | - | 4 | 2010–2012 (2010  2012) | `-` |
| 5 | `Sód` | 0 | - | 4 | 2010–2012 (2010  2012) | `-` |
| 6 | `Biochemia` | 0 | - | 1 | 2017 | `-` |

## 3B. Badania zlecane WYŁĄCZNIE w grupie KOR (`ONLY_KOR`) — 16 szt.

| Lp. | Nazwa w `examination_type` | KOR (liczba) | Lata KOR | NEURO (liczba) | Lata NEURO | Mierzone parametry (single) |
| :---: | :--- | :---: | :--- | :---: | :--- | :--- |
| 1 | `Beta-2-mikroglobulina we krwi` | 45 | 2020 | 0 | - | `-` |
| 2 | `Przeciwciała przeciwko cytrulinowanym peptydom (anty-CCP)` | 27 | 2020–2021 (2020  2021) | 0 | - | `-` |
| 3 | `Antystreptolizyna O (ASO)` | 16 | 2020–2021 (2020  2021) | 0 | - | `eGFR-MDRD  eGFRCKD  exam_list` |
| 4 | `Przeciwciała przeciwko cytrulinowanym peptydom` | 14 | 2020 | 0 | - | `-` |
| 5 | `Wydalanie dobowe białka z moczem` | 14 | 2020–2021 (2020  2021) | 0 | - | `-` |
| 6 | `Albumina w moczu` | 11 | 2020–2021 (2020  2021) | 0 | - | `-` |
| 7 | `Czynnik reumatoidalny w klasie IgM ( metoda turbidymetryczna)` | 11 | 2020 | 0 | - | `-` |
| 8 | `Antystreptolizyny` | 10 | 2020 | 0 | - | `-` |
| 9 | `Enzymy` | 4 | 2021 | 0 | - | `-` |
| 10 | `Amylaza w moczu` | 2 | 2020 | 0 | - | `-` |
| 11 | `Wydalanie dobowe sodu z moczem` | 2 | 2020 | 0 | - | `-` |
| 12 | `Wydalanie sodu.` | 2 | 2020 | 0 | - | `-` |
| 13 | `Erytroblasty` | 1 | 2021 | 0 | - | `-` |
| 14 | `Hormony 2` | 1 | 2021 | 0 | - | `-` |
| 15 | `Mocznik we krwi ( parametr wyliczany)` | 1 | 2021 | 0 | - | `-` |
| 16 | `Wydalanie kreatyniny/ na dobę` | 1 | 2020 | 0 | - | `-` |

## 3C. Badania zlecane w OBU grupach (`BOTH`) — 47 szt.

| Lp. | Nazwa w `examination_type` | KOR (liczba) | Lata KOR | NEURO (liczba) | Lata NEURO | Mierzone parametry (single) |
| :---: | :--- | :---: | :--- | :---: | :--- | :--- |
| 1 | `Morfologia` | 124 413 | 2019–2021 (2019  2020  2021) | 12 799 | 2010–2024 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `%BAZO  %EO  %IG  %LYMPH  %MONO  %NEUT  %NRBC  BAZO  EO  HCT  HGB  IG  LYMPH  MCH  MCHC  MCV  MONO  MPV  NEUT  NRBC  P-LCR  PLT  RBC  RDW  WBC  exam_list` |
| 2 | `Kreatynina we krwi` | 100 516 | 2019–2021 (2019  2020  2021) | 7 519 | 2013–2024 (2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `KREA  eGFR-MDRD  eGFRCKD  exam_list` |
| 3 | `Białko C-reaktywne` | 95 542 | 2019–2021 (2019  2020  2021) | 7 474 | 2010–2024 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `CRP  exam_list` |
| 4 | `Elektrolity (potas  sód)` | 70 334 | 2020–2021 (2020  2021) | 3 337 | 2020–2024 (2020  2021  2022  2023  2024) | `K  Na  exam_list` |
| 5 | `Elektrolity` | 42 666 | 2019–2020 (2019  2020) | 8 205 | 2010–2020 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020) | `K  Na  exam_list` |
| 6 | `Czas protrombinowy` | 44 453 | 2019–2021 (2019  2020  2021) | 5 064 | 2000–2024 (2000  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 7 | `Czas częściowej tromboplastyny po aktywacji` | 43 962 | 2019–2021 (2019  2020  2021) | 4 877 | 2000–2024 (2000  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `APTT  WAPTT  exam_list` |
| 8 | `Glukoza` | 43 069 | 2019–2021 (2019  2020  2021) | 4 205 | 2010–2024 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `GLU  exam_list` |
| 9 | `Transaminaza alaninowa` | 37 473 | 2019–2021 (2019  2020  2021) | 3 676 | 2010–2024 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `ALT  exam_list` |
| 10 | `Transaminaza asparaginianowa` | 29 879 | 2019–2021 (2019  2020  2021) | 3 259 | 2010–2024 (2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `AST  exam_list` |
| 11 | `Wskaźnik protrombinowy` | 27 205 | 2019–2021 (2019  2020  2021) | 4 455 | 2000–2024 (2000  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 12 | `Azot mocznika we krwi` | 27 009 | 2019–2021 (2019  2020  2021) | 4 169 | 2013–2024 (2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `BUN  exam_list` |
| 13 | `Wskaźnik filtracji kłębuszkowej (wg MDRD)` | 22 924 | 2019–2021 (2019  2020  2021) | 4 981 | 2014–2024 (2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 14 | `Wskaźnik filtracji kłębuszkowej (wg CKD-EPI)` | 22 925 | 2019–2021 (2019  2020  2021) | 4 965 | 2014–2024 (2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 15 | `INR` | 21 525 | 2019–2021 (2019  2020  2021) | 4 425 | 2000–2024 (2000  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 16 | `Współczynnik APTT` | 14 560 | 2019–2021 (2019  2020  2021) | 4 084 | 2000–2024 (2000  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `-` |
| 17 | `Bilirubina całkowita we krwi` | 12 957 | 2019–2021 (2019  2020  2021) | 193 | 2017–2021 (2017  2018  2019  2020  2021) | `eGFR-MDRD  eGFRCKD  exam_list` |
| 18 | `Koagulologia` | 4 918 | 2019–2021 (2019  2020  2021) | 135 | 2016–2021 (2016  2017  2018  2019  2020  2021) | `-` |
| 19 | `Fosforany nieorganiczne` | 4 010 | 2020–2021 (2020  2021) | 123 | 2020–2021 (2020  2021) | `-` |
| 20 | `Fosfor nieorganiczny we krwi` | 2 407 | 2019–2020 (2019  2020) | 89 | 2017–2020 (2017  2018  2019  2020) | `-` |
| 21 | `Morfologia z retikulocytami` | 2 215 | 2019–2021 (2019  2020  2021) | 92 | 2010–2024 (2010  2011  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `%BAZO  %EO  %IG  %LYMPH  %MONO  %NEUT  %NRBC  BAZO  EO  HCT  HGB  IG  LYMPH  MCH  MCHC  MCV  MONO  MPV  NEUT  NRBC  P-LCR  PLT  RBC  RDW  WBC  exam_list` |
| 22 | `Wapń całkowity we krwi` | 1 317 | 2019–2021 (2019  2020  2021) | 54 | 2017–2021 (2017  2018  2019  2020  2021) | `eGFR-MDRD  eGFRCKD  exam_list` |
| 23 | `Kwas moczowy` | 1 273 | 2020–2021 (2020  2021) | 11 | 2020–2021 (2020  2021) | `-` |
| 24 | `Albumina` | 934 | 2020–2021 (2020  2021) | 37 | 2020–2021 (2020  2021) | `-` |
| 25 | `Kwas moczowy we krwi` | 727 | 2019–2020 (2019  2020) | 45 | 2017–2020 (2017  2018  2019  2020) | `-` |
| 26 | `Wydalanie z moczem` | 653 | 2020–2021 (2020  2021) | 8 | 2017–2021 (2017  2021) | `KREA  exam_list` |
| 27 | `Magnez (Mg)` | 641 | 2020–2021 (2020  2021) | 7 | 2020–2021 (2020  2021) | `-` |
| 28 | `Kreatynina w moczu` | 406 | 2020–2021 (2020  2021) | 69 | 2013–2024 (2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024) | `KREA  exam_list` |
| 29 | `Magnez we krwi` | 386 | 2019–2020 (2019  2020) | 28 | 2017–2020 (2017  2019  2020) | `-` |
| 30 | `Albumina w surowicy` | 396 | 2019–2020 (2019  2020) | 13 | 2017–2020 (2017  2018  2019  2020) | `-` |
| 31 | `Azot mocznika w moczu` | 168 | 2020–2021 (2020  2021) | 6 | 2013–2023 (2013  2014  2016  2019  2022  2023) | `-` |
| 32 | `Sód (Na) w moczu` | 158 | 2020–2021 (2020  2021) | 14 | 2020–2024 (2020  2021  2023  2024) | `-` |
| 33 | `Sód w moczu` | 123 | 2019–2020 (2019  2020) | 18 | 2013–2020 (2013  2015  2016  2018  2019  2020) | `-` |
| 34 | `Bilirubina bezpośrednia` | 130 | 2020–2021 (2020  2021) | 2 | 2020 | `-` |
| 35 | `Białko całkowite we krwi` | 115 | 2020–2021 (2020  2021) | 7 | 2017–2021 (2017  2019  2020  2021) | `-` |
| 36 | `Chlorki (Cl)` | 108 | 2020–2021 (2020  2021) | 6 | 2020–2021 (2020  2021) | `-` |
| 37 | `Potas w moczu` | 89 | 2020 | 14 | 2013–2020 (2013  2015  2016  2018  2019  2020) | `-` |
| 38 | `Potas (K) w moczu` | 97 | 2020–2021 (2020  2021) | 3 | 2023 | `-` |
| 39 | `Beta-2-mikroglobulina` | 95 | 2020–2021 (2020  2021) | 2 | 2021 | `-` |
| 40 | `Chlorki we krwi` | 69 | 2020 | 5 | 2018–2020 (2018  2019  2020) | `-` |
| 41 | `Astrowirusy - antygen [kał]` | 65 | 2020–2021 (2020  2021) | 2 | 2019–2023 (2019  2023) | `-` |
| 42 | `KTV- surowica` | 14 | 2020–2021 (2020  2021) | 8 | 2022–2024 (2022  2023  2024) | `-` |
| 43 | `Czynnik reumatoidalny (RF)` | 18 | 2020–2021 (2020  2021) | 1 | 2021 | `-` |
| 44 | `Kwasy żółciowe` | 15 | 2020–2021 (2020  2021) | 2 | 2020 | `-` |
| 45 | `HPC Morfologia (produkt cytaferezy)` | 11 | 2020–2021 (2020  2021) | 1 | 2021 | `-` |
| 46 | `Koagulologia- przeciwciała` | 8 | 2020–2021 (2020  2021) | 1 | 2021 | `-` |
| 47 | `KTV- mocz` | 5 | 2020–2021 (2020  2021) | 1 | 2024 | `-` |
