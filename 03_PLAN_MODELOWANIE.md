# Plan: modelowanie Positive-Unlabeled

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zakres:** punkty 6–9 — trening, porównanie modeli i wnioski
**Poprzedza go:** `02_PLAN_PRZED_MODELOWANIEM.md` (etap 0 i punkty 1–5)
**Data:** 11.09.2026

To jest właściwy etap trenowania modeli predykcyjnych. Wcześniej dopasowywane są już scaler i imputer, ale pierwszy model przypisujący risk score powstaje w punkcie 6.

---

## Czego ten etap wymaga na wejściu

Nic z poniższego nie może ruszyć, zanim nie będzie gotowe i **zamrożone**:

| Wejście | Skąd |
|---|---|
| Foldy zewnętrzne + tabela `patient_id → fold` | punkt 1 |
| Pipeline foldów obsługujący walidację zagnieżdżoną | punkt 1 |
| Kolumny `true_label`, `observed_label`, `is_hidden` | punkt 2 |
| Maski ukrywania: udziały 20/40/60% × 5–10 ziaren, identyczne dla wszystkich modeli | punkt 2 |
| Metryki pacjentowe + reguła agregacji rekordów do pacjenta | punkty 0.3 i 4 |
| Funkcja celu `RecallHidden@q` z ustalonymi `q` i `α` | punkt 5 |
| Uzgodniony wariant kontroli czasu/źródła danych oraz analiza wrażliwości | punkt 0.2 |
| Z góry wskazany scenariusz główny: zestaw cech, udział ukrywania i seedy | punkty 0.4, 2 i 5 |

Do środowiska trzeba dodatkowo doinstalować bibliotekę boostingową — obecnie nie ma ani `xgboost`, ani `lightgbm`. Wybieramy **jedną**, nie obie.

---

## Plan prac

### 6. Strojenie hiperparametrów — z walidacją zagnieżdżoną

Strojenie na tych samych foldach, na których potem raportujemy wynik, zawyża jakość. Dlatego układ musi być zagnieżdżony:

```
dla każdego foldu zewnętrznego (ocena):
    inner Group CV na części treningowej tego foldu
    ├─ preprocessing (scaler + MICE) dopasowywany od nowa w każdym foldzie wewnętrznym
    ├─ Optuna szuka hiperparametrów maksymalizujących RecallHidden@q
    └─ najlepsza konfiguracja trenowana na całym trainie foldu zewnętrznego
    ocena na foldzie zewnętrznym — raz, bez zaglądania wcześniej
```

Wartość `q` nie jest hiperparametrem Optuny: zostaje zamrożona przed treningiem na podstawie przepustowości diagnostyki. Jeśli raportujemy dodatkowo klasyfikację progową, próg można dobrać wyłącznie w inner CV. Funkcją celu jest `RecallHidden@q` (albo wcześniej zamrożony wariant `S(q)`), a nie accuracy i nie ROC-AUC.

Preprocessing jest niezależny od hiperparametrów klasyfikatora, więc wyniki scaler + MICE można cache'ować dla każdego konkretnego inner splitu. Nie wolno natomiast raz zaimputować całego outer-train i dopiero potem dzielić go na inner foldy. Dla kontroli kosztu obliczeń Optunę uruchamiamy na scenariuszu głównym; pozostałe udziały ukrywania i seedy służą do oceny wrażliwości zamrożonej konfiguracji, a nie do wielokrotnego wybierania najlepszego modelu.

### 7. Porównanie modeli — z co najmniej jedną prawdziwą metodą PU

Regresja logistyczna, Random Forest i boosting uczone z KOR jako klasą `0` to **naiwne baseline'y PN**, a nie PU learning. Jeśli poprzestaniemy na nich, raport będzie porównaniem klasyfikatorów na zaszumionych etykietach — a nie tym, co deklarujemy w tytule projektu.

Zestaw do porównania:

| Model | Rola |
|---|---|
| Regresja logistyczna P-vs-U | baseline liniowy, punkt odniesienia |
| Random Forest | baseline drzewiasty |
| XGBoost **albo** LightGBM | jeden boosting, nie oba |
| PU Bagging | właściwa metoda PU |
| Elkan–Noto lub inne oszacowanie class prior | metoda/korekta PU; założenia i estymacja prioru wykonywane wyłącznie na train |

Wszystkie modele pracują na identycznych foldach, identycznych maskach ukrywania i z tym samym zestawem metryk pacjentowych. Założenie SCAR wymagane przez część metod musi być ocenione osobno; wyniku korekty Elkan–Noto nie wolno automatycznie nazywać skalibrowanym prawdopodobieństwem choroby.

### 8. Analiza pacjentów wysokiego ryzyka — wyłącznie na predykcjach OOF

Analiza opiera się **tylko na out-of-fold predictions**. Predykcja pacjenta pochodzi zawsze z modelu, który tego pacjenta nie widział w treningu.

Kwestia nazewnictwa, ważna dla raportu: wyniku modelu **nie nazywamy „prawdopodobieństwem tętniaka"**, dopóki nie jest skalibrowany z wiarygodnym class prior. Mówimy o **risk score** albo o pozycji w rankingu. Dla pacjentów z górnej części rankingu raportujemy:

- wynik i percentyl,
- stabilność wyniku między ziarnami ukrywania i modelami; pojedynczy pacjent występuje tylko w jednym outer foldzie, więc stabilność „między foldami” wymaga osobnego repeated outer CV,
- liczbę pomiarów danego pacjenta (przy rozrzucie 1–38 to istotny kontekst),
- najważniejsze cechy / SHAP,
- anomalie i braki danych w profilu,
- czy pacjent należy do kontrolowanych ukrytych pozytywnych.

Interpretacja kliniczna: to jest lista kandydatów do dalszej diagnostyki obrazowej, a nie rozpoznanie. Model wspiera selekcję do badań, nie stawia diagnozy.

### 9. Artefakty wyjściowe etapu

Oprócz samych modeli i tabel z metrykami z tego etapu muszą wyjść:

1. tabela `patient_id → fold`,
2. tabela `patient_id → true_label, observed_label, is_hidden, seed`,
3. predykcje OOF dla każdego modelu,
4. wyniki per ziarno wraz z przedziałami ufności,
5. analiza stabilności rankingów między ziarnami i modelami; między podziałami tylko wtedy, gdy wykonamy repeated outer CV,
6. **model card** — założenia, ograniczenia, przeznaczenie i czego model nie robi,
7. pełna konfiguracja eksperymentu i wersje bibliotek.

Przedziały ufności należy liczyć na poziomie pacjenta, np. bootstrapem grupowym. Foldów CV nie należy traktować jako niezależnych obserwacji do prostego testu t ani przedstawiać odchylenia między pięcioma foldami jako pełnej niepewności populacyjnej.

---

## Kolejność i zależności

```
wejście z 02_PLAN_PRZED_MODELOWANIEM.md (zamrożone)
└─> 6 (Optuna w walidacji zagnieżdżonej; q zamrożone, ewentualny próg ustalany wewnątrz)
     └─> 7 (porównanie modeli, w tym metody PU, na identycznych foldach i maskach)
          └─> 8 (analiza wysokiego ryzyka na predykcjach OOF)
               └─> 9 (artefakty, model card, konfiguracja)
                    └─> raport końcowy
```

---

## Ograniczenia, które muszą trafić do raportu

Niezależnie od wyników, sześć rzeczy trzeba powiedzieć wprost:

1. **Nie mierzymy rzeczywistego false-positive rate.** Ukrywanie pozytywnych testuje odzyskiwanie znanych przypadków. Nie mówi, ilu pacjentów KOR jest naprawdę zdrowych.
2. **Założenie SCAR jest wątpliwe.** NEURO to kohorta neurologiczna, prawdopodobnie bardziej objawowa niż przeciętny nierozpoznany chory, więc znani pozytywni nie są losową próbką wszystkich pozytywnych.
3. **ROC-AUC i PR-AUC liczone z KOR jako klasą negatywną to metryki P-vs-U**, a nie skuteczność wykrywania choroby.
4. **Kohorty są silnie rozdzielone czasowo i źródłowo.** Wynik może odzwierciedlać epokę, laboratorium, hospitalizację lub sposób pozyskania danych; analiza w oknie wspólnym jest obowiązkową analizą wrażliwości.
5. **Brak daty diagnozy ogranicza interpretację predykcyjną.** Przejście pacjenta z rekordu KOR do NEURO nie dowodzi, że pierwszy rekord był sprzed rozpoznania.
6. **Wyniki NEURO pochodzą najprawdopodobniej z hospitalizacji, w trakcie której rozpoznano i leczono tętniaka.** Przy tym założeniu model rozpoznaje profil pacjenta hospitalizowanego, a nie profil ryzyka przed rozpoznaniem. Część sygnału może pochodzić z zabiegu, kontrastu i reakcji okołooperacyjnej, dlatego wyniku nie wolno opisywać jako modelu przesiewowego.
