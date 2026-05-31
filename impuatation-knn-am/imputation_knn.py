import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# =============================
# FUNKCJA KL DIVERGENCE
# =============================
def kl_divergence(p, q, bins=30):
    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    p_hist += 1e-10
    q_hist += 1e-10

    return entropy(p_hist, q_hist)

# =============================
# WCZYTANIE DANYCH
# =============================
NAZWA_BAZY = 'kor_shortend.csv'
kor = pd.read_csv(NAZWA_BAZY)

df_numeric = kor.select_dtypes(include=[np.number])

if 'patient_id' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=['patient_id'])

# =============================
# FILTRACJA KOLUMN
# =============================
col_threshold = 0.40 * len(df_numeric)
best_cols = df_numeric.columns[df_numeric.count() >= col_threshold]
df_filtered = df_numeric[best_cols]

df_clean = df_filtered.dropna()

print(f"Pierwotna liczba kolumn: {df_numeric.shape[1]}")
print(f"Liczba kolumn po odrzuceniu najpustszych: {df_filtered.shape[1]}")
print(f"Liczba kompletnych wierszy: {df_clean.shape[0]}")

# =============================
# SCALING
# =============================
scaler = MinMaxScaler()
df_clean_scaled = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns
)

# =============================
# MASKOWANIE DANYCH
# =============================
np.random.seed(42)
df_masked = df_clean_scaled.copy()
mask = np.random.rand(*df_masked.shape) < 0.1
df_masked[mask] = np.nan

# =============================
# TEST KNN
# =============================
results = []
k_options = [1, 3, 5, 7, 11, 15, 21]
weight_options = ['uniform', 'distance']

for weight in weight_options:
    for k in k_options:
        imputer = KNNImputer(n_neighbors=k, weights=weight)
        df_imputed_array = imputer.fit_transform(df_masked)
        df_imputed = pd.DataFrame(df_imputed_array, columns=df_clean_scaled.columns)

        # METRYKI
        actual = df_clean_scaled.values[mask]
        predicted = df_imputed.values[mask]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        # =============================
        # KL DIVERGENCE
        # =============================
        kl_values = []
        for col in df_clean_scaled.columns:
            orig = df_clean_scaled[col].values
            imp = df_imputed[col].values

            kl = kl_divergence(orig, imp)
            kl_values.append(kl)

        kl_avg = np.mean(kl_values)

        # =============================
        # WALIDACJA WARTOŚCI
        # =============================
        min_val = df_imputed.min().min()
        max_val = df_imputed.max().max()
        neg_count = (df_imputed < 0).sum().sum()

        # zapis wyników
        results.append({
            'K': k,
            'Wagi': weight,
            'RMSE': rmse,
            'MAE': mae,
            'KL': kl_avg,
            'Min': min_val,
            'Max': max_val,
            'Liczba_ujemnych': int(neg_count)
        })

# =============================
# WYNIKI
# =============================
df_res = pd.DataFrame(results).sort_values(by='RMSE')

print("\n" + "="*60)
print("RANKING KONFIGURACJI KNN (wg RMSE):")
print("="*60)
print(df_res.to_string(index=False))

best_row = df_res.iloc[0]

print("\n" + "-" * 60)
print(f"Najlepszy model:")
print(f"K = {int(best_row['K'])}")
print(f"Wagi = {best_row['Wagi']}")
print(f"RMSE = {best_row['RMSE']:.4f}")
print(f"MAE = {best_row['MAE']:.4f}")
print(f"KL = {best_row['KL']:.4f}")
print("-" * 60)

# =============================
# WYKRES
# =============================
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6), dpi=150)

sns.lineplot(
    data=df_res,
    x='K',
    y='RMSE',
    hue='Wagi',
    marker='o'
)

plt.title(f'Optymalizacja KNNImputer ({NAZWA_BAZY})')
plt.xlabel('Liczba sąsiadów (K)')
plt.ylabel('RMSE')

plt.xticks(k_options)

plt.tight_layout()
plt.savefig(f"wykres_knn_{NAZWA_BAZY}.png")
plt.show()

md_file = f"wyniki_knn_{NAZWA_BAZY.replace('.csv','')}.md"

with open(md_file, "w", encoding="utf-8") as f:
    f.write("# Wyniki KNN Imputer\n\n")

    f.write("## Ranking modeli\n\n")
    f.write(df_res.to_markdown(index=False))

    f.write("\n\n## Najlepszy model\n\n")
    f.write(f"- K: {int(best_row['K'])}\n")
    f.write(f"- Wagi: {best_row['Wagi']}\n")
    f.write(f"- RMSE: {best_row['RMSE']:.4f}\n")
    f.write(f"- MAE: {best_row['MAE']:.4f}\n")
    f.write(f"- KL divergence: {best_row['KL']:.4f}\n")

    f.write("\n\n## Walidacja wartości\n\n")
    f.write(f"- Min: {best_row['Min']:.4f}\n")
    f.write(f"- Max: {best_row['Max']:.4f}\n")
    f.write(f"- Liczba wartości ujemnych: {best_row['Liczba_ujemnych']}\n")
    
print(f"\nWyniki zapisane do: {md_file}")
