import pandas as pd

kor   = pd.read_csv(r"c:\Users\kjub2\Desktop\PU DATASET\kor_shortend.csv")
neuro = pd.read_csv(r"c:\Users\kjub2\Desktop\PU DATASET\neuro_shortend.csv")

# --- patient_id duplicates between datasets ---
common_ids = set(kor["patient_id"]) & set(neuro["patient_id"])
print(f"Unikalne patient_id w kor   : {kor['patient_id'].nunique()}")
print(f"Unikalne patient_id w neuro : {neuro['patient_id'].nunique()}")
print(f"Wspólne patient_id          : {len(common_ids)}")
print()

if common_ids:
    kor_counts   = kor[kor["patient_id"].isin(common_ids)].groupby("patient_id").size().rename("wizyty_kor")
    neuro_counts = neuro[neuro["patient_id"].isin(common_ids)].groupby("patient_id").size().rename("wizyty_neuro")
    summary = pd.concat([kor_counts, neuro_counts], axis=1).sort_values("wizyty_kor", ascending=False)
    print("Pacjenci obecni w obu zbiorach (posortowani po liczbie wizyt w KOR):")
    print(summary.to_string())
    print()
    print(f"Łączna liczba wierszy KOR   dla tych pacjentów: {kor_counts.sum()}")
    print(f"Łączna liczba wierszy NEURO dla tych pacjentów: {neuro_counts.sum()}")

