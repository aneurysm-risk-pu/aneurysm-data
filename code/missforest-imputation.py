import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy


# --- Funkcja obliczająca KL Divergence ---
def calculate_kl_divergence(actual, predicted, bins=30):
    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))

    hist_actual, bin_edges = np.histogram(actual, bins=bins, range=(min_val, max_val), density=True)
    hist_predicted, _ = np.histogram(predicted, bins=bin_edges, density=True)

    epsilon = 1e-10
    p = hist_actual + epsilon
    q = hist_predicted + epsilon

    return entropy(p / np.sum(p), q / np.sum(q))


# -----------------------------------------

pliki_medyczne = {
    'KOR': '../datasets/cleaned/kor_shortend.csv',
    'NEURO': '../datasets/cleaned/neuro_shortend.csv'
}

plik_raportu = 'raport_missforest.md'

with open(plik_raportu, 'w', encoding='utf-8') as f:
    f.write("# Raport z imputacji MissForest (Test na pełnych wierszach)\n\n")
    f.write(
        "Model uczony jest na całym zbiorze danych, ale metryki ewaluacyjne liczone są **wyłącznie** na losowo zamaskowanych wartościach pacjentów, którzy oryginalnie posiadali komplet badań.\n")

for nazwa_oddzialu, sciezka_do_pliku in pliki_medyczne.items():
    print(f"\nURUCHAMIAM ANALIZĘ DLA: {nazwa_oddzialu.upper()}")

    try:
        df_raw = pd.read_csv(sciezka_do_pliku)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku: {sciezka_do_pliku}")
        continue

    # Usuwamy patient_id do analizy
    df_numeric = df_raw.select_dtypes(include=[np.number])
    if 'patient_id' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['patient_id'])

    # Zostawiamy tylko kolumny z minimum 60% wypełnienia
    threshold = 0.6 * len(df_numeric)
    best_cols = df_numeric.columns[df_numeric.count() > threshold]
    df = df_numeric[best_cols]

    # Skalujemy cały zbiór (potrzebne do poprawnego wyliczenia RMSE w jednej skali 0-1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # 1. IDENTYFIKACJA PEŁNYCH WIERSZY (COMPLETE CASES)
    maska_pelnych_wierszy = df_scaled.notna().all(axis=1)
    df_pelne_wiersze = df_scaled[maska_pelnych_wierszy]

    if len(df_pelne_wiersze) == 0:
        print(f"Brak pełnych wierszy w {nazwa_oddzialu}. Pomijam test.")
        continue

    print(f"Znaleziono {len(df_pelne_wiersze)} pacjentów z kompletem badań.")

    # 2. MASKOWANIE 10% DANYCH TYLKO W PEŁNYCH WIERSZACH
    np.random.seed(42)
    random_mask = np.random.rand(*df_pelne_wiersze.shape) < 0.1

    # Kopia pełnych wierszy, w której wstawiamy NaN tam, gdzie wylosowała się maska (True)
    df_pelne_wiersze_z_brakami = df_pelne_wiersze.mask(random_mask)

    # Kopia CAŁEGO zbioru, do której podmieniamy pełne wiersze na te z dziurami
    df_do_imputacji = df_scaled.copy()
    df_do_imputacji.loc[maska_pelnych_wierszy, :] = df_pelne_wiersze_z_brakami

    results = []
    trees_options = [10, 30, 70, 150]

    with open(plik_raportu, 'a', encoding='utf-8') as f:
        f.write(f"\n## {nazwa_oddzialu}\n")
        f.write(f"* Liczba wszystkich pacjentów: {len(df_scaled)}\n")
        f.write(f"* Pacjenci testowi (pełne wiersze): {len(df_pelne_wiersze)}\n")
        f.write(f"* Zamaskowanych komórek do testu: {random_mask.sum()}\n\n")
        f.write("| Drzewa | RMSE | MAE | KL Divergence | Uwagi |\n")
        f.write("|:---:|:---:|:---:|:---:|:---|\n")

    print("Rozpoczynam testowanie. Model liczy na całym zbiorze")

    for n in trees_options:
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=n, max_depth=15, max_features='sqrt', random_state=42, n_jobs=1),
            max_iter=4,
            random_state=42
        )

        # 3. IMPUTACJA NA CAŁYM ZBIORZE
        df_imputed_array = imputer.fit_transform(df_do_imputacji)
        df_imputed = pd.DataFrame(df_imputed_array, columns=df_scaled.columns, index=df_scaled.index)

        # 4. WYCIĄGNIĘCIE METRYK (TYLKO DLA KOMÓREK Z MASKI)
        # Bierzemy oryginalne wartości z pełnych wierszy i te same miejsca po imputacji
        actual = df_pelne_wiersze.values[random_mask]
        predicted = df_imputed.loc[maska_pelnych_wierszy, :].values[random_mask]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        kl_div = calculate_kl_divergence(actual, predicted)

        results.append({'Trees': n, 'RMSE': rmse, 'MAE': mae, 'KL_Div': kl_div})

        uwagi = ""
        if len(results) > 1:
            poprzednie_rmse = results[-2]['RMSE']
            procent_poprawy = (poprzednie_rmse - rmse) / poprzednie_rmse
            if procent_poprawy < 0.01:
                uwagi = f"Zatrzymano (poprawa < 1%)"

        with open(plik_raportu, 'a', encoding='utf-8') as f:
            f.write(f"| **{n}** | {rmse:.4f} | {mae:.4f} | {kl_div:.4f} | {uwagi} |\n")

        print(f"  Drzewa={n:3d} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | KL: {kl_div:.4f} {uwagi}")
        if uwagi: break

    # Podsumowanie
    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['RMSE'].idxmin()]

    with open(plik_raportu, 'a', encoding='utf-8') as f:
        f.write("\n> **Wniosek:** Najlepszy wynik uzyskano dla algorytmu złożonego z ")
        f.write(f"**{int(best_row['Trees'])} drzew** (RMSE = {best_row['RMSE']:.4f}).\n---\n")

print(f"\nZakończono. Raport dostępny pod: {os.path.abspath(plik_raportu)}")