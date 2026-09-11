# Raport z imputacji MissForest (Test na pełnych wierszach)

Model uczony jest na całym zbiorze danych, ale metryki ewaluacyjne liczone są **wyłącznie** na losowo zamaskowanych wartościach pacjentów, którzy oryginalnie posiadali komplet badań.

## KOR
* Liczba wszystkich pacjentów: 71546
* Pacjenci testowi (pełne wiersze): 50393
* Zamaskowanych komórek do testu: 161163

| Drzewa | RMSE | MAE | KL Divergence | Uwagi |
|:---:|:---:|:---:|:---:|:---|
| **10** | 0.0876 | 0.0339 | 0.0122 |  |
| **30** | 0.0846 | 0.0325 | 0.0130 |  |
| **70** | 0.0850 | 0.0324 | 0.0130 | Zatrzymano (poprawa < 1%) |

> **Wniosek:** Najlepszy wynik uzyskano dla algorytmu złożonego z **30 drzew** (RMSE = 0.0846).
---

## NEURO
* Liczba wszystkich pacjentów: 6616
* Pacjenci testowi (pełne wiersze): 3404
* Zamaskowanych komórek do testu: 8480

| Drzewa | RMSE | MAE | KL Divergence | Uwagi |
|:---:|:---:|:---:|:---:|:---|
| **10** | 0.1121 | 0.0556 | 0.0515 |  |
| **30** | 0.1084 | 0.0531 | 0.0777 |  |
| **70** | 0.1053 | 0.0517 | 0.0969 |  |
| **150** | 0.1057 | 0.0514 | 0.0966 | Zatrzymano (poprawa < 1%) |

> **Wniosek:** Najlepszy wynik uzyskano dla algorytmu złożonego z **70 drzew** (RMSE = 0.1053).
---
