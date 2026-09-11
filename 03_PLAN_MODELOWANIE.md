# Plan: modelowanie Positive-Unlabeled

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Zakres:** punkty 6–9 — trening, porównanie modeli i wnioski
**Poprzedza go:** `PLAN_PRZYGOTOWANIE_PU.md` (etap 0 i punkty 1–5)
**Data:** 11.09.2026

To jest właściwy etap modelowania. Pierwszy moment w całym projekcie, w którym cokolwiek się uczy, to punkt 6.

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

Dwie rzeczy, o których łatwo zapomnieć: **próg klasyfikacji** i **wartość top-k** też są parametrami i też muszą być ustalane wewnątrz, bez dotykania foldu zewnętrznego. Funkcją celu jest `RecallHidden@q` (albo wariant `S(q)`), a nie accuracy i nie ROC-AUC.

### 7. Porównanie modeli — z co najmniej jedną prawdziwą metodą PU

Regresja logistyczna, Random Forest i boosting uczone z KOR jako klasą `0` to **naiwne baseline'y PN**, a nie PU learning. Jeśli poprzestaniemy na nich, raport będzie porównaniem klasyfikatorów na zaszumionych etykietach — a nie tym, co deklarujemy w tytule projektu.

Zestaw do porównania:

| Model | Rola |
|---|---|
| Regresja logistyczna P-vs-U | baseline liniowy, punkt odniesienia |
| Random Forest | baseline drzewiasty |
| XGBoost **albo** LightGBM | jeden boosting, nie oba |
| PU Bagging | właściwa metoda PU |
| Elkan–Noto lub inne oszacowanie class prior | korekta prawdopodobieństwa pod PU |

Wszystkie modele na identycznych foldach, identycznych maskach ukrywania i z tym samym zestawem metryk pacjentowych.

### 8. Analiza pacjentów wysokiego ryzyka — wyłącznie na predykcjach OOF

Analiza opiera się **tylko na out-of-fold predictions**. Predykcja pacjenta pochodzi zawsze z modelu, który tego pacjenta nie widział w treningu.

Kwestia nazewnictwa, ważna dla raportu: wyniku modelu **nie nazywamy „prawdopodobieństwem tętniaka"**, dopóki nie jest skalibrowany z wiarygodnym class prior. Mówimy o **risk score** albo o pozycji w rankingu. Dla pacjentów z górnej części rankingu raportujemy:

- wynik i percentyl,
- stabilność wyniku między foldami, ziarnami i modelami,
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
5. analiza stabilności rankingów (między foldami, ziarnami i modelami),
6. **model card** — założenia, ograniczenia, przeznaczenie i czego model nie robi,
7. pełna konfiguracja eksperymentu i wersje bibliotek.

---

## Kolejność i zależności

```
wejście z PLAN_PRZYGOTOWANIE_PU.md (zamrożone)
└─> 6 (Optuna w walidacji zagnieżdżonej, próg i top-k ustalane wewnątrz)
     └─> 7 (porównanie modeli, w tym metody PU, na identycznych foldach i maskach)
          └─> 8 (analiza wysokiego ryzyka na predykcjach OOF)
               └─> 9 (artefakty, model card, konfiguracja)
                    └─> raport końcowy
```

---

## Ograniczenia, które muszą trafić do raportu

Niezależnie od wyników, trzy rzeczy trzeba powiedzieć wprost:

1. **Nie mierzymy rzeczywistego false-positive rate.** Ukrywanie pozytywnych testuje odzyskiwanie znanych przypadków. Nie mówi, ilu pacjentów KOR jest naprawdę zdrowych.
2. **Założenie SCAR jest wątpliwe.** NEURO to kohorta neurologiczna, prawdopodobnie bardziej objawowa niż przeciętny nierozpoznany chory, więc znani pozytywni nie są losową próbką wszystkich pozytywnych.
3. **ROC-AUC i PR-AUC liczone z KOR jako klasą negatywną to metryki P-vs-U**, a nie skuteczność wykrywania choroby.
