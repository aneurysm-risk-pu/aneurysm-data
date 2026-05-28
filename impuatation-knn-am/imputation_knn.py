import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

NAZWA_BAZY = 'neuro_shortend.csv' 
kor = pd.read_csv(NAZWA_BAZY)

df_numeric = kor.select_dtypes(include=[np.number])
if 'patient_id' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=['patient_id'])

col_threshold = 0.40 * len(df_numeric)
best_cols = df_numeric.columns[df_numeric.count() >= col_threshold]
df_filtered = df_numeric[best_cols]

df_clean = df_filtered.dropna()

print(f"Pierwotna liczba kolumn: {df_numeric.shape[1]}")
print(f"Liczba kolumn po odrzuceniu najpustszych: {df_filtered.shape[1]}")
print(f"Liczba kompletnych wierszy do testu RMSE: {df_clean.shape[0]}")

scaler = MinMaxScaler()
df_clean_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

np.random.seed(42)
df_masked = df_clean_scaled.copy()
mask = np.random.rand(*df_masked.shape) < 0.1
df_masked[mask] = np.nan

results = []
k_options = [1, 3, 5, 7, 11, 15, 21]
weight_options = ['uniform', 'distance'] 

for weight in weight_options:
    for k in k_options:
        imputer = KNNImputer(n_neighbors=k, weights=weight)
        df_imputed_array = imputer.fit_transform(df_masked)
        df_imputed = pd.DataFrame(df_imputed_array, columns=df_clean_scaled.columns)
        
        actual = df_clean_scaled.values[mask]
        predicted = df_imputed.values[mask]
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        results.append({
            'K': k, 
            'Wagi': weight, 
            'RMSE': rmse, 
            'MAE': mae
        })

df_res = pd.DataFrame(results).sort_values(by='RMSE')

print("\n" + "="*50)
print("RANKING KONFIGURACJI KNN (od najniższego RMSE):")
print("="*50)
print(df_res.to_string(index=False))

best_row = df_res.iloc[0]
print("\n" + "-" * 50)
print(f"Najlepszy imputer KNN: K={int(best_row['K'])}, Wagi='{best_row['Wagi']}'")
print(f"Ostateczne RMSE: {best_row['RMSE']:.4f}")
print(f"Ostateczne MAE: {best_row['MAE']:.4f}")
print("-" * 50)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6), dpi=150)

sns.lineplot(
    data=df_res, 
    x='K', 
    y='RMSE', 
    hue='Wagi', 
    marker='o', 
    linewidth=2.5, 
    markersize=8,
    palette=['#1f77b4', '#ff7f0e']
)

plt.title(f'Optymalizacja hiperparametrów KNNImputer dla zbioru: {NAZWA_BAZY}', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Liczba sąsiadów (K)', fontsize=12)
plt.ylabel('Błąd RMSE', fontsize=12)
plt.xticks(k_options)
plt.legend(title='Strategia ważenia (weights)', fontsize=10, title_fontsize=11)
plt.tight_layout()
nazwa_wykresu = f"wykres_knn_{NAZWA_BAZY.replace('.csv', '')}.png"
plt.savefig(nazwa_wykresu, bbox_inches='tight')
plt.show()