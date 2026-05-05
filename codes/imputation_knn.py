import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Wczytanie danych
kor = pd.read_csv('datasets/kor_cleaning.csv')

# Wybór kolumn numerycznych i usuwanie ID 
df_numeric = kor.select_dtypes(include=[np.number])
if 'patient_id' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=['patient_id'])

# Dynamiczny wybór najlepszych kolumn
threshold = 0.6 * len(df_numeric)
best_cols = df_numeric.columns[df_numeric.count() > threshold]
df_clean = df_numeric[best_cols].dropna()

if len(df_clean) > 1000:
    df_clean = df_clean.iloc[:1000]

scaler = MinMaxScaler()
# Tworzymy czysty, przeskalowany dataset
df_clean_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

print(f"Wybrane kolumny ({len(best_cols)}): {list(best_cols)}")
print(f"Rozmiar czystego datasetu: {df_clean_scaled.shape}")

# Losowe zaburzanie brakami 
np.random.seed(42)
df_masked = df_clean_scaled.copy()
mask = np.random.rand(*df_masked.shape) < 0.1
df_masked[mask] = np.nan

results = []
# Testujemy różne wartości K, żeby znaleźć optimum
k_options = [1, 3, 5, 7, 11, 15, 21]

for k in k_options:
    imputer = KNNImputer(n_neighbors=k)
    df_imputed_array = imputer.fit_transform(df_masked)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df_clean_scaled.columns)
    
    actual = df_clean_scaled.values[mask]
    predicted = df_imputed.values[mask]
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    results.append({'K': k, 'RMSE': rmse, 'MAE': mae})
    print(f"K={k:2d} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

df_res = pd.DataFrame(results)
best_row = df_res.loc[df_res['RMSE'].idxmin()]

print("-" * 30)
print(f"NAJLEPSZY IMPUTER: KNN z K={int(best_row['K'])}")
print(f"Ostateczne RMSE: {best_row['RMSE']:.4f}")