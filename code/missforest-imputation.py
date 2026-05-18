import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

pliki_medyczne = {
    'KOR': '../datasets/cleaned/kor_cleaning.csv',
    'NEURO': '../datasets/cleaned/neuro_cleaning.csv'
}

for nazwa_oddzialu, sciezka_do_pliku in pliki_medyczne.items():
    print("\n" + "=" * 50)
    print(f"URUCHAMIAM ANALIZĘ DLA: {nazwa_oddzialu.upper()}")
    print("=" * 50)

    try:
        kor = pd.read_csv(sciezka_do_pliku)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku: {sciezka_do_pliku}")
        continue

    df_numeric = kor.select_dtypes(include=[np.number])
    if 'patient_id' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['patient_id'])

    threshold = 0.6 * len(df_numeric)
    best_cols = df_numeric.columns[df_numeric.count() > threshold]
    df_clean = df_numeric[best_cols].dropna()

    if len(df_clean) > 1000:
        df_clean = df_clean.iloc[:1000]

    scaler = MinMaxScaler()
    df_clean_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

    print(f"Kolumny ({len(best_cols)}). Rozmiar bazy: {df_clean_scaled.shape}")

    np.random.seed(42)
    df_masked = df_clean_scaled.copy()
    mask = np.random.rand(*df_masked.shape) < 0.1
    df_masked[mask] = np.nan

    results = []
    trees_options = [10, 30, 70, 150]

    print(f"\n Testowanie MissForest dla {nazwa_oddzialu}...")

    for n in trees_options:
        # SZYBKOŚĆ
        # max_features='sqrt' drastycznie przyspiesza budowę drzew
        # max_iter=4 zamiast 10 (lasy szybko zbiegają do optimum)
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=n, max_features='sqrt', random_state=42, n_jobs=-1),
            max_iter=4,
            random_state=42
        )

        df_imputed_array = imputer.fit_transform(df_masked)
        df_imputed = pd.DataFrame(df_imputed_array, columns=df_clean_scaled.columns)

        actual = df_clean_scaled.values[mask]
        predicted = df_imputed.values[mask]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        results.append({'Trees': n, 'RMSE': rmse, 'MAE': mae})
        print(f"  Drzewa={n:3d} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        # Early Stopping: Jeśli zysk z dołożenia drzew jest mniejszy niż 1%, przerywamy
        if len(results) > 1:
            poprzednie_rmse = results[-2]['RMSE']
            procent_poprawy = (poprzednie_rmse - rmse) / poprzednie_rmse

            if procent_poprawy < 0.01:  # 1% progu opłacalności
                print(
                    f"  [Early Stopping] Kolejne drzewa poprawiły wynik tylko o {procent_poprawy * 100:.2f}%. Zatrzymuję poszukiwania.")
                break

    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['RMSE'].idxmin()]

    print("-" * 40)
    print(f"NAJLEPSZA KONFIGURACJA DLA {nazwa_oddzialu.upper()}:")
    print(f"Liczba drzew: {int(best_row['Trees'])} | Najniższe RMSE: {best_row['RMSE']:.4f}")