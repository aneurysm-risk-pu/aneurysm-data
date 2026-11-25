import ast
from pathlib import Path
import numpy as np
import pandas as pd


# 1. KONFIGURACJA ŚCIEŻEK
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_DIR = REPO_ROOT / "data"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KOR_INPUT = RAW_DIR / "kor_merged_aggregated_1W_mean.csv"
NEURO_INPUT = RAW_DIR / "neuro_merged_aggregated_1W_mean.csv"

# KOR_INPUT = REPO_ROOT / "data/interim/kor_cleaning.csv"
# NEURO_INPUT = REPO_ROOT / "data/interim/neuro_cleaning.csv"

KOR_OUTPUT = PROCESSED_DIR / "kor_preprocessed_final.csv"
NEURO_OUTPUT = PROCESSED_DIR / "neuro_preprocessed_final.csv"
CONCATTED_OUTPUT = PROCESSED_DIR / "concatted_preprocessed.csv"


def log_step_summary(name: str, step_label: str, prev_cols: list[str], df: pd.DataFrame) -> None:
    """Wypisuje rozmiar tabeli oraz listę kolumn odrzuconych w danym kroku."""
    dropped = [c for c in prev_cols if c not in df.columns]
    dropped_str = str(dropped) if dropped else "brak"
    print(f"[{name}] {step_label} -> rozmiar: {df.shape} | odrzucone kolumny ({len(dropped)}): {dropped_str}")


# 2. FUNKCJE POMOCNICZE (PARSOWANIE SŁOWNIKÓW, JEDNOSTKI, WALIDACJA)
def unpack_dict_cell(val):
    """
    Rozpakowuje struktury typu {'values': ['15.0']} lub {'values': ['A', 'B']}.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, dict):
        vals = val.get("values", [])
        if not vals:
            return np.nan
        return vals[0] if len(vals) == 1 else ", ".join(str(v) for v in vals)
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and ("'values'" in s or '"values"' in s):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict) and "values" in parsed:
                    vals = parsed["values"]
                    if not vals:
                        return np.nan
                    return vals[0] if len(vals) == 1 else ", ".join(str(v) for v in vals)
            except (ValueError, SyntaxError):
                pass
    return val


def unpack_all_dict_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Wykrywa i rozpakowuje wszystkie kolumny tekstowe/obiektowe zawierające słowniki {'values': ...}."""
    prev_cols = df.columns.tolist()
    df = df.copy()
    unpacked_cols = []
    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        sample = df[col].dropna().astype(str)
        if sample.str.contains(r"\{\s*['\"]values['\"]\s*:", regex=True).any():
            df[col] = df[col].apply(unpack_dict_cell)
            unpacked_cols.append(col)

    print(f"[{name}] Rozpakowano słowniki w {len(unpacked_cols)} kolumnach: {unpacked_cols}")
    log_step_summary(name, "Po unpack_all_dict_columns", prev_cols, df)
    return df


def convert_numeric_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Konwertuje rozpakowane kolumny laboratoryjne (np. ALT, BUN) na wartości numeryczne (float).
    """
    prev_cols = df.columns.tolist()
    df = df.copy()
    meta_cols = {
        "patient_id", "custom_id", "patient_age", "patient_sex",
        "diagnosis_id", "examination_date", "examination_type", "descriptive_result"
    }
    converted = []
    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        if col not in meta_cols:
            def _to_float(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, (int, float)):
                    return float(x)
                parts = [p.strip() for p in str(x).split(",") if p.strip()]
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        return np.nan
                return float(np.mean(nums)) if nums else np.nan

            df[col] = df[col].apply(_to_float)
            converted.append(col)

    if converted:
        print(f"[{name}] Przekonwertowano rozpakowane kolumny na float: {converted}")
    log_step_summary(name, "Po convert_numeric_features", prev_cols, df)
    return df


def process_and_drop_units_and_norms(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Weryfikuje spójność jednostek w kolumnach '-unit', uzupełnia braki dominantą
    (dla spójności audytu), a następnie usuwa wszystkie kolumny '-unit', '-norm'
    oraz opcjonalnie 'examination_type' (jeśli występuje w DataFrame).
    """
    prev_cols = df.columns.tolist()
    df = df.copy()
    unit_cols = [c for c in df.columns if c.endswith("-unit")]
    norm_cols = [c for c in df.columns if c.endswith("-norm")]

    inconsistent = []
    for col in unit_cols:
        non_null = df[col].dropna()
        if not non_null.empty:
            mode_val = non_null.mode().iloc[0]
            if non_null.nunique() > 1:
                inconsistent.append((col, non_null.unique().tolist()))
            df[col] = df[col].fillna(mode_val)

    if inconsistent:
        print(f"[{name}] Uwaga — wykryto niespójne jednostki w kolumnach: {inconsistent}")
    else:
        print(f"[{name}] Walidacja jednostek (-unit): brak niespójności.")

    cols_to_drop = unit_cols + norm_cols
    has_exam_type = "examination_type" in df.columns
    if has_exam_type:
        cols_to_drop.append("examination_type")

    df = df.drop(columns=cols_to_drop, errors="ignore")

    exam_msg = " oraz kolumnę 'examination_type'" if has_exam_type else " (brak kolumny 'examination_type')"
    print(f"[{name}] Usunięto {len(unit_cols)} kolumn '-unit', {len(norm_cols)} kolumn '-norm'{exam_msg}.")
    log_step_summary(name, "Po process_and_drop_units_and_norms", prev_cols, df)
    return df


def deduplicate_within_cohort(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Z Notatnika 1: usuwa zduplikowane rekordy custom_id wewnątrz pojedynczej kohorty (keep='first')."""
    prev_cols = df.columns.tolist()
    if "custom_id" not in df.columns:
        log_step_summary(name, "Po deduplicate_within_cohort (brak custom_id)", prev_cols, df)
        return df

    dup_count = df.duplicated(subset=["custom_id"], keep="first").sum()
    if dup_count > 0:
        dups = df[df.duplicated(subset=["custom_id"], keep=False)]["custom_id"].unique().tolist()
        print(f"[{name}] Wykryto i usunięto {dup_count} duplikatów custom_id (przykłady: {dups[:5]}).")
        df = df.drop_duplicates(subset=["custom_id"], keep="first").copy()
    else:
        print(f"[{name}] Brak duplikatów custom_id wewnątrz kohorty (0).")

    log_step_summary(name, "Po deduplicate_within_cohort", prev_cols, df)
    return df


def clean_demographics(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Z Notatnika 2:
    - Filtruje wiek pacjentów do zakresu 18–100 lat.
    - Konwertuje patient_sex na Int64 i usuwa wiersze z brakiem płci (np. 1 rekord w KOR).
    """
    prev_cols = df.columns.tolist()
    df = df.copy()
    initial_len = len(df)

    if "patient_age" in df.columns:
        df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
        df = df[df["patient_age"].between(18, 100)].copy()
        print(f"[{name}] Filtrowanie wieku (18–100 lat): usunięto {initial_len - len(df)} wierszy (pozostało {len(df)}).")

    if "patient_sex" in df.columns:
        before_sex = len(df)
        df["patient_sex"] = pd.to_numeric(df["patient_sex"], errors="coerce").astype("Int64")
        missing_sex = df["patient_sex"].isna().sum()
        if missing_sex > 0:
            df = df.dropna(subset=["patient_sex"]).copy()
        print(f"[{name}] Konwersja patient_sex -> Int64: usunięto {before_sex - len(df)} wierszy z brakiem płci.")

    log_step_summary(name, "Po clean_demographics", prev_cols, df)
    return df


def validate_clinical_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Z Notatnika 2 — walidacja kliniczna (zamiana błędnych pomiarów na NaN):
    - Usuwa wartości ujemne (< 0) we wszystkich parametrach numerycznych.
    - Usuwa zera (== 0) dla parametrów ściśle dodatnich (niebędących parametrami odsetkowymi/różniczkowymi).
    - Usuwa wartości procentowe > 100% (np. 5 rekordów w %NRBC oraz pozostałych kolumnach '%').
    """
    prev_cols = df.columns.tolist()
    df = df.copy()
    meta_cols = {
        "patient_id", "custom_id", "patient_age", "patient_sex",
        "diagnosis_id", "examination_date", "examination_type", "descriptive_result"
    }
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]

    # Parametry, w których wartość 0.0 jest klinicznie dopuszczalna (np. frakcje leukocytów, erytroblasty)
    zero_allowed_cols = {
        "%BAZO", "%EO", "%IG", "%LYMPH", "%MONO", "%NEUT", "%NRBC",
        "BAZO", "EO", "IG", "LYMPH", "MONO", "NEUT", "NRBC"
    }

    neg_replaced = 0
    zero_replaced = 0
    pct_replaced = 0

    for col in num_cols:
        # 1. Wartości ujemne -> NaN
        neg_mask = df[col] < 0
        if neg_mask.any():
            neg_replaced += int(neg_mask.sum())
            df.loc[neg_mask, col] = np.nan

        # 2. Zera dla parametrów ściśle dodatnich -> NaN
        if col not in zero_allowed_cols:
            zero_mask = df[col] == 0
            if zero_mask.any():
                zero_replaced += int(zero_mask.sum())
                df.loc[zero_mask, col] = np.nan

        # 3. Wartości procentowe > 100% -> NaN (szczególnie %NRBC)
        if col.startswith("%"):
            over_100_mask = df[col] > 100
            if over_100_mask.any():
                cnt = int(over_100_mask.sum())
                pct_replaced += cnt
                print(f"[{name}] Kolumna '{col}': zamieniono {cnt} wartości > 100% na NaN.")
                df.loc[over_100_mask, col] = np.nan

    print(
        f"[{name}] Walidacja kliniczna zakończona: "
        f"ujemne -> NaN ({neg_replaced}), niedozwolone zera -> NaN ({zero_replaced}), >100% -> NaN ({pct_replaced})."
    )
    log_step_summary(name, "Po validate_clinical_values", prev_cols, df)
    return df


def analyze_outliers_iqr(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Z Notatnika 2: analizuje wartości odstające metodą rozstępu ćwiartkowego (1.5 * IQR)
    dla wszystkich numerycznych cech laboratoryjnych (raport diagnostyczny bez usuwania rekordów).
    """
    prev_cols = df.columns.tolist()
    meta_cols = {
        "patient_id", "custom_id", "patient_age", "patient_sex",
        "diagnosis_id", "examination_date", "examination_type", "descriptive_result"
    }
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]

    records = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        pct = (outliers / len(series)) * 100
        records.append({
            "feature": col,
            "count": len(series),
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "outliers_count": int(outliers),
            "outliers_pct": round(pct, 2),
        })

    iqr_df = pd.DataFrame(records).sort_values(by="outliers_pct", ascending=False)
    print(f"[{name}] Top 10 cech z najwyższym odsetkiem outlierów (metoda 1.5 * IQR):")
    print(iqr_df.head(10).to_string(index=False))
    log_step_summary(name, "Po analyze_outliers_iqr", prev_cols, df)
    return iqr_df


def drop_high_missingness(
    df: pd.DataFrame,
    threshold: float = 0.60,
    meta_cols: list[str] | None = None,
    drop_cols: bool = True,
    drop_rows: bool = True,
    cohort_name: str = "DATASET",
) -> pd.DataFrame:
    """
    8.5. Usuwa kolumny oraz/lub wiersze (rekordy) mające więcej niż `threshold` (domyślnie 60%) braków (NaN)
    w obrębie cech laboratoryjnych.
    """
    prev_cols = df.columns.tolist()
    if meta_cols is None:
        meta_cols = [
            "custom_id", "patient_id", "examination_date", "patient_age",
            "patient_sex", "diagnosis_id", "descriptive_result",
            "examination_type", "cohort", "year", "label"
        ]

    work_df = df.copy()
    feature_cols = [c for c in work_df.columns if c not in meta_cols]

    if not feature_cols:
        print(f"[{cohort_name}] Brak kolumn cech do oceny braków danych.")
        log_step_summary(cohort_name, "Po drop_high_missingness", prev_cols, work_df)
        return work_df

    # 1. Drop kolumn z > 60% braków
    if drop_cols:
        col_missing_ratio = work_df[feature_cols].isna().mean()
        cols_to_drop = col_missing_ratio[col_missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            work_df = work_df.drop(columns=cols_to_drop)
            feature_cols = [c for c in feature_cols if c not in cols_to_drop]
            print(f"[{cohort_name}] Usunięto kolumny (> {threshold:.0%} braków): {cols_to_drop}")
        else:
            print(f"[{cohort_name}] Brak kolumn z > {threshold:.0%} braków.")

    # 2. Drop wierszy (rekordów) z > 60% braków wśród cech laboratoryjnych
    if drop_rows and feature_cols:
        row_missing_ratio = work_df[feature_cols].isna().mean(axis=1)
        mask_keep = row_missing_ratio <= threshold
        dropped_rows_cnt = int((~mask_keep).sum())
        work_df = work_df[mask_keep].reset_index(drop=True)
        print(
            f"[{cohort_name}] Usunięto wiersze (> {threshold:.0%} braków): {dropped_rows_cnt} "
            f"(pozostało: {len(work_df)})"
        )

    log_step_summary(cohort_name, "Po drop_high_missingness", prev_cols, work_df)
    return work_df


# 3. GŁÓWNY POTOK PRZETWARZANIA (PIPELINE)
def run_preprocessing():
    print("=" * 80)
    print("START PREPROCESSINGU DANYCH (KOR + NEURO)")
    print("=" * 80)

    # 1. Wczytanie surowych danych
    if not KOR_INPUT.exists() or not NEURO_INPUT.exists():
        raise FileNotFoundError(f"Nie znaleziono plików wejściowych w: {RAW_DIR}")

    kor = pd.read_csv(KOR_INPUT, encoding="utf-8-sig", low_memory=False)
    neuro = pd.read_csv(NEURO_INPUT, encoding="utf-8-sig", low_memory=False)
    print(f"[KOR] Wczytano dane -> rozmiar: {kor.shape} | kolumny ({len(kor.columns)}): {kor.columns.tolist()}")
    print(f"[NEURO] Wczytano dane -> rozmiar: {neuro.shape} | kolumny ({len(neuro.columns)}): {neuro.columns.tolist()}")
    print("-" * 80)

    # 2. Wstępna diagnostyka kolumn z >60% braków
    for name, df in [("KOR", kor), ("NEURO", neuro)]:
        prev_cols = df.columns.tolist()
        missing_ratio = df.isna().mean()
        high_missing = missing_ratio[missing_ratio > 0.60].index.tolist()
        if high_missing:
            print(f"[{name}] Kolumny z >60% braków (diagnostyka): {high_missing}")
        log_step_summary(name, "Po diagnostyce >60% braków", prev_cols, df)
    print("-" * 80)

    # 3. Deduplikacja, rozpakowanie słowników, jednostki/normy, konwersja numeryczna
    kor = deduplicate_within_cohort(kor, "KOR")
    neuro = deduplicate_within_cohort(neuro, "NEURO")
    print("-" * 80)

    kor = unpack_all_dict_columns(kor, "KOR")
    neuro = unpack_all_dict_columns(neuro, "NEURO")
    print("-" * 80)

    kor = process_and_drop_units_and_norms(kor, "KOR")
    neuro = process_and_drop_units_and_norms(neuro, "NEURO")
    print("-" * 80)

    kor = convert_numeric_features(kor, "KOR")
    neuro = convert_numeric_features(neuro, "NEURO")
    print("-" * 80)

    # Usunięcie diagnosis_id oraz descriptive_result (jeśli jeszcze istnieją)
    drop_meta = ["diagnosis_id", "descriptive_result"]
    for name, df_ref in [("KOR", kor), ("NEURO", neuro)]:
        prev_cols = df_ref.columns.tolist()
        new_df = df_ref.drop(columns=[c for c in drop_meta if c in df_ref.columns])
        log_step_summary(name, "Po usunięciu metadanych (diagnosis_id, descriptive_result)", prev_cols, new_df)
        if name == "KOR":
            kor = new_df
        else:
            neuro = new_df
    print("-" * 80)

    # 4. Usunięcie wspólnych pacjentów / rekordów występujących w obu kohortach
    kor_prev_cols = kor.columns.tolist()
    neuro_prev_cols = neuro.columns.tolist()

    if "patient_id" in kor.columns and "patient_id" in neuro.columns:
        common_patients = set(kor["patient_id"].dropna()) & set(neuro["patient_id"].dropna())
        print(f"Wykryto {len(common_patients)} wspólnych pacjentów (patient_id) między KOR i NEURO — usuwanie z obu kohort.")
        kor = kor[~kor["patient_id"].isin(common_patients)].copy()
        neuro = neuro[~neuro["patient_id"].isin(common_patients)].copy()
    elif "custom_id" in kor.columns and "custom_id" in neuro.columns:
        common_custom = set(kor["custom_id"].dropna()) & set(neuro["custom_id"].dropna())
        print(f"Wykryto {len(common_custom)} wspólnych rekordów (custom_id) między KOR i NEURO — usuwanie z obu kohort.")
        kor = kor[~kor["custom_id"].isin(common_custom)].copy()
        neuro = neuro[~neuro["custom_id"].isin(common_custom)].copy()

    log_step_summary("KOR", "Po usunięciu wspólnych pacjentów", kor_prev_cols, kor)
    log_step_summary("NEURO", "Po usunięciu wspólnych pacjentów", neuro_prev_cols, neuro)
    print("-" * 80)

    # 5. Czyszczenie demograficzne (wiek 18–100 lat, płeć Int64 + usunięcie NaN w płci)
    kor = clean_demographics(kor, "KOR")
    neuro = clean_demographics(neuro, "NEURO")
    print("-" * 80)

    # 6. Walidacja kliniczna wartości (ujemne, zera dla ściśle dodatnich, >100% w %NRBC)
    kor = validate_clinical_values(kor, "KOR")
    neuro = validate_clinical_values(neuro, "NEURO")
    print("-" * 80)

    # # 7. Analiza outlierów metodą IQR
    # analyze_outliers_iqr(kor, "KOR")
    # analyze_outliers_iqr(neuro, "NEURO")

    # 7. Drop > 60%
    kor = drop_high_missingness(kor, threshold=0.60, cohort_name="KOR")
    neuro = drop_high_missingness(neuro, threshold=0.60, cohort_name="NEURO")
    print("-" * 80)

    # 8. Wyrównanie schematu kolumn do części wspólnej (zachowując oryginalną kolejność z KOR)
    kor_prev_cols = kor.columns.tolist()
    neuro_prev_cols = neuro.columns.tolist()

    common_cols = [c for c in kor.columns if c in set(neuro.columns)]
    kor = kor[common_cols].copy()
    neuro = neuro[common_cols].copy()

    log_step_summary("KOR", "Po wyrównaniu do części wspólnej kolumn", kor_prev_cols, kor)
    log_step_summary("NEURO", "Po wyrównaniu do części wspólnej kolumn", neuro_prev_cols, neuro)
    print("-" * 80)

    # 9. Zapis finalnych zbiorów oraz pliku połączonego (concatted)
    kor.to_csv(KOR_OUTPUT, index=False, encoding="utf-8-sig")
    neuro.to_csv(NEURO_OUTPUT, index=False, encoding="utf-8-sig")
    concatted = merge_and_save_concatted(kor, neuro, CONCATTED_OUTPUT)

    print("=" * 80)
    print("PREPROCESSING ZAKOŃCZONY POMYŚLNIE")
    print(f"Zapisano KOR ({kor.shape[0]} wierszy, {kor.shape[1]} kolumn) -> {KOR_OUTPUT}")
    print(f"Zapisano NEURO ({neuro.shape[0]} wierszy, {neuro.shape[1]} kolumn) -> {NEURO_OUTPUT}")
    print(f"Zapisano CONCATTED ({concatted.shape[0]} wierszy, {concatted.shape[1]} kolumn) -> {CONCATTED_OUTPUT}")
    print("=" * 80)



# merge kor & neuro
def merge_and_save_concatted(
    kor_df: pd.DataFrame,
    neuro_df: pd.DataFrame,
    output_path: Path,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Łączy zbiory KOR i NEURO w jedną ramkę danych (`concatted`), dodając kolumnę celu `label`
    (0 = KOR, 1 = NEURO), usuwa wskazane kolumny identyfikatorów/dat i zapisuje wynik do CSV.
    """
    if drop_cols is None:
        drop_cols = ["custom_id", "patient_id", "examination_date"]

    kor_labeled = kor_df.copy()
    kor_labeled["label"] = 0

    neuro_labeled = neuro_df.copy()
    neuro_labeled["label"] = 1

    concatted = pd.concat([kor_labeled, neuro_labeled], ignore_index=True)

    existing_to_drop = [c for c in drop_cols if c in concatted.columns]
    concatted = concatted.drop(columns=existing_to_drop)

    concatted.to_csv(output_path, index=False, encoding="utf-8-sig")

    label_counts = concatted["label"].value_counts().to_dict()
    print(
        f"[CONCATTED] Połączono i zapisano zbiór -> rozmiar: {concatted.shape} | "
        f"odrzucone kolumny ({len(existing_to_drop)}): {existing_to_drop} | "
        f"rozkład label: {label_counts} | plik: {output_path}"
    )
    return concatted


if __name__ == "__main__":
    run_preprocessing()
