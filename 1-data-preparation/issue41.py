"""
Analiza pojedynczych cech (kolumn pomiarowych) w wyczyszczonych danych (_cleaned1.csv):
1. Rozkład niepustych pomiarów dla każdej pojedynczej cechy w podziale na kohortę (KOR vs NEURO) i rok.
2. Wykrywanie potencjalnych synonimów w kolumnie examination_type.
3. Generowanie zestawienia pojedynczych cech (BOTH, ONLY_KOR, ONLY_NEURO) do raportu .md.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. KONFIGURACJA ŚCIEŻEK
USE_SYNTHETIC = False  # Zmień na False, aby wygenerować finalny wynik na data/

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / ("data_syntetyczne" if USE_SYNTHETIC else "data")
CLEANED_DIR = BASE_DIR / "interim"

FILES_MAP = {
    "kor": BASE_DIR / "raw/kor_merged_aggregated_1W_mean.csv",
    "neuro": BASE_DIR / "raw/neuro_merged_aggregated_1W_mean.csv",
}

# FILES_MAP = {
#     "kor": CLEANED_DIR / "kor_cleaned1.csv",
#     "neuro": CLEANED_DIR / "neuro_cleaned1.csv",
# }

OUTPUT_DIR = REPO_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kolumny metadanych (niebędące cechami laboratoryjnymi)
META_COLS = {
    "cohort", "year", "examination_type", "patient_id", "custom_id",
    "patient_age", "patient_sex", "diagnosis_id", "descriptive_result",
    "date", "week", "examination_date", "nan_count"
}

# Mapowanie pojedynczych kolumn na polskie nazwy badań (z notatnika)
COLUMN_TO_POLISH_NAME = {
    "MCH": "Morfologia",
    "MCHC": "Morfologia",
    "MCV": "Morfologia",
    "HGB": "Morfologia",
    "RBC": "Morfologia",
    "WBC": "Morfologia",
    "HCT": "Morfologia",
    "RDW": "Morfologia",
    "PLT": "Morfologia",
    "MPV": "Morfologia",
    "P-LCR": "Morfologia (płytki krwi)",
    "NEUT": "Morfologia / Rozmaz mikroskopowy",
    "LYMPH": "Morfologia / Rozmaz mikroskopowy",
    "MONO": "Morfologia / Rozmaz mikroskopowy",
    "EO": "Morfologia / Rozmaz mikroskopowy",
    "BAZO": "Morfologia / Rozmaz mikroskopowy",
    "IG": "Morfologia (niedojrzałe granulocyty)",
    "NRBC": "Morfologia / Erytroblasty",
    "%NEUT": "Morfologia (%)",
    "%LYMPH": "Morfologia (%)",
    "%MONO": "Morfologia (%)",
    "%EO": "Morfologia (%)",
    "%BAZO": "Morfologia (%)",
    "%IG": "Morfologia (%)",
    "%NRBC": "Morfologia (%)",
    "GLU": "Glukoza",
    "KREA": "Kreatynina",
    "eGFR-MDRD": "Wskaźnik filtracji kłębuszkowej (wg MDRD)",
    "eGFRCKD": "Wskaźnik filtracji kłębuszkowej (wg CKD-EPI)",
    "BUN": "Azot mocznika we krwi",
    "Na": "Sód / Elektrolity",
    "K": "Potas / Elektrolity",
    "CRP": "Białko C-reaktywne",
    "ALT": "Transaminaza alaninowa",
    "AST": "Transaminaza asparaginianowa",
    "INR": "INR",
    "PT": "Czas protrombinowy",
    "WPT": "Wskaźnik protrombinowy",
    "APTT": "Czas częściowej tromboplastyny po aktywacji (APTT)",
    "WAPTT": "Współczynnik APTT",
}


# 2. WCZYTYWANIE I PARSOWANIE DANYCH WYCZYSZCZONYCH
def parse_examination_types(cell_value) -> list[str]:
    """
    Parsuje kolumnę examination_type po rozpakowaniu w notatniku (np. 'Morfologia,Białko C-reaktywne')
    oraz zachowuje wsteczną kompatybilność z surowym formatem {'values': [...]}.
    """
    if pd.isna(cell_value):
        return []
    if isinstance(cell_value, dict):
        return cell_value.get("values", [])
    if isinstance(cell_value, str):
        text = cell_value.strip()
        if not text:
            return []
        if text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed.get("values", [])
            except (ValueError, SyntaxError):
                pass
        # Rozdzielenie po przecinku, ignorując przecinki wewnątrz nawiasów np. "Elektrolity (potas, sód)"
        parts = re.split(r",(?![^()]*\))", text)
        return [p.strip() for p in parts if p.strip()]
    return []


def extract_year(df: pd.DataFrame) -> pd.Series:
    """Wyciąga rok bezpośrednio z kolumny examination_date (lub awaryjnie z custom_id)."""
    if "examination_date" in df.columns:
        return pd.to_datetime(df["examination_date"], errors="coerce").dt.year.astype("Int64")

    if "custom_id" in df.columns:
        return (
            df["custom_id"]
            .astype(str)
            .str.extract(r"-(\d{4})-")[0]
            .astype("Int64")
        )

    raise KeyError(f"Brak kolumny examination_date i custom_id. Dostępne: {list(df.columns)}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Zwraca listę pojedynczych kolumn pomiarowych (bez metadanych, -unit i -norm)."""
    return [
        c for c in df.columns
        if c not in META_COLS and not c.endswith("-unit") and not c.endswith("-norm")
    ]


def load_cohorts() -> pd.DataFrame:
    """Wczytuje obie wyczyszczone kohorty (kor_cleaned1 i neuro_cleaned1)."""
    dfs = []
    for cohort_name, file_path in FILES_MAP.items():
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Brak pliku: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        df["cohort"] = cohort_name
        df["year"] = extract_year(df)
        dfs.append(df)

        print(f"Wczytano {cohort_name.upper()} ({file_path.name}): {len(df)} rekordów.")

    return pd.concat(dfs, ignore_index=True)


# 3. ANALIZA 1: POJEDYNCZE CECHY WEDŁUG KOHORTY I ROKU
def analyze_features_by_cohort_and_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zlicza niepuste wystąpienia (pomiary) dla każdej pojedynczej cechy (kolumny)
    w podziale na (cohort, year).
    """
    feature_cols = get_feature_columns(df)

    # Zliczanie niepustych wartości dla każdej cechy w grupach (cohort, year)
    grouped_counts = (
        df.groupby(["cohort", "year"], dropna=False)[feature_cols]
        .apply(lambda g: g.notna().sum())
    )

    # Przekształcenie do formatu długiego (summary)
    summary = (
        grouped_counts.stack()
        .reset_index()
        .rename(columns={"level_2": "feature_name", 0: "count"})
        .sort_values(["feature_name", "cohort", "year"])
    )

    pivot = summary.pivot_table(
        index="feature_name",
        columns=["cohort", "year"],
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    pivot.columns = [f"{cohort}_{year}" for cohort, year in pivot.columns]

    kor_cols = [c for c in pivot.columns if c.startswith("kor_")]
    neuro_cols = [c for c in pivot.columns if c.startswith("neuro_")]

    total_kor = pivot[kor_cols].sum(axis=1) if kor_cols else pd.Series(0, index=pivot.index)
    total_neuro = pivot[neuro_cols].sum(axis=1) if neuro_cols else pd.Series(0, index=pivot.index)

    pivot["TOTAL_KOR"] = total_kor
    pivot["TOTAL_NEURO"] = total_neuro
    pivot["ONLY_IN"] = np.where(
        (total_kor == 0) & (total_neuro > 0),
        "ONLY_NEURO",
        np.where((total_neuro == 0) & (total_kor > 0), "ONLY_KOR", "BOTH"),
    )

    return summary, pivot


# 4. ANALIZA 2: IDENTYFIKACJA SYNONIMÓW W EXAMINATION_TYPE
def detect_synonyms(df: pd.DataFrame) -> dict:
    """Szuka synonimów w kolumnie examination_type na podstawie podobieństwa tekstowego i sygnatur cech."""
    exam_series = df["examination_type"].apply(parse_examination_types)
    unique_names = sorted({name for sublist in exam_series for name in sublist})

    lexical_groups = defaultdict(list)
    for name in unique_names:
        core = name.split("(")[0].strip().lower()
        lexical_groups[core].append(name)

    lexical_candidates = {k: v for k, v in lexical_groups.items() if len(v) > 1}

    feature_cols = get_feature_columns(df)
    single_exam_df = df.copy()
    single_exam_df["exam_list"] = exam_series
    single_exam_df = single_exam_df[single_exam_df["exam_list"].str.len() == 1].copy()
    single_exam_df["single_name"] = single_exam_df["exam_list"].str[0]

    feature_signatures = {}
    if not single_exam_df.empty and feature_cols:
        presence = (
            single_exam_df.groupby("single_name")[feature_cols]
            .apply(lambda g: g.notna().mean())
        )
        sig_to_names = defaultdict(list)
        for exam_name, row in presence.iterrows():
            active_features = tuple(sorted(row[row > 0.8].index.tolist()))
            if active_features:
                sig_to_names[active_features].append(exam_name)

        feature_signatures = {
            ", ".join(feats): names
            for feats, names in sig_to_names.items()
            if len(names) > 1
        }

    return {
        "kandydaci_tekstowi": lexical_candidates,
        "kandydaci_po_kolumnach_wynikow": feature_signatures,
        "zatwierdzone_mapowanie_synonimow": {},
    }


# 5. GENEROWANIE RAPORTU .MD DLA POJEDYNCZYCH CECH
def infer_polish_names_from_data(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, str]:
    """
    Uzupełnia słownik polskich nazw na podstawie pojedynczych zleceń z examination_type,
    jeśli cechy nie ma na sztywno w COLUMN_TO_POLISH_NAME.
    """
    mapping = dict(COLUMN_TO_POLISH_NAME)
    work_df = df[["examination_type"] + feature_cols].copy()
    work_df["exam_list"] = work_df["examination_type"].apply(parse_examination_types)
    single_df = work_df[work_df["exam_list"].str.len() == 1].copy()

    if single_df.empty:
        return mapping

    single_df["exam_name"] = single_df["exam_list"].str[0]
    for col in feature_cols:
        if col in mapping:
            continue
        non_null_exams = single_df.loc[single_df[col].notna(), "exam_name"]
        if not non_null_exams.empty:
            top_exam = non_null_exams.mode().iloc[0]
            mapping[col] = top_exam

    return mapping


def format_single_feature_label(feature_name: str, polish_map: dict[str, str]) -> str:
    """Zwraca format: '`FEATURE` (nazwa polska)'."""
    polish_name = polish_map.get(feature_name)
    if polish_name:
        return f"`{feature_name}` ({polish_name})"
    return f"`{feature_name}`"


def generate_cohort_overlap_report(df: pd.DataFrame, pivot_df: pd.DataFrame, tag: str) -> Path:
    """
    Zapisuje zestawienie POJEDYNCZYCH cech (BOTH, ONLY_KOR, ONLY_NEURO)
    do pliku raportu Markdown (.md).
    """
    report_path = OUTPUT_DIR / f"{tag}_raport_pokrycia_cech.md"
    feature_cols = get_feature_columns(df)
    polish_map = infer_polish_names_from_data(df, feature_cols)

    groups_meta = [
        (
            "ONLY_NEURO",
            "Cechy występujące WYŁĄCZNIE w kohorcie NEURO (`ONLY_NEURO`)",
            "Pojedyncze zmienne mające niepuste pomiary wyłącznie w kohorcie NEURO (ryzyko wycieku danych / *data leakage*).",
            "TOTAL_NEURO",
        ),
        (
            "ONLY_KOR",
            "Cechy występujące WYŁĄCZNIE w kohorcie KOR (`ONLY_KOR`)",
            "Pojedyncze zmienne mające niepuste pomiary wyłącznie w kohorcie KOR.",
            "TOTAL_KOR",
        ),
        (
            "BOTH",
            "Cechy wspólne dla OBU kohort (`BOTH`)",
            "Pojedyncze zmienne obecne zarówno w zbiorze `kor_cleaned1.csv`, jak i `neuro_cleaned1.csv`.",
            "TOTAL_ALL",
        ),
    ]

    work_pivot = pivot_df.copy()
    work_pivot["TOTAL_ALL"] = work_pivot["TOTAL_KOR"] + work_pivot["TOTAL_NEURO"]

    md_lines = [
        f"# Analiza pojedynczych cech (kolumn pomiarowych) między kohortami [{tag}]",
        "",
        "## Podsumowanie liczbowe",
        "",
        "| Kategoria | Liczba pojedynczych cech | Udział (%) |",
        "| :--- | :---: | :---: |",
    ]

    total_unique = len(work_pivot)
    for group_key in ["BOTH", "ONLY_KOR", "ONLY_NEURO"]:
        cnt = int((work_pivot["ONLY_IN"] == group_key).sum())
        pct = (cnt / total_unique * 100) if total_unique > 0 else 0.0
        md_lines.append(f"| `{group_key}` | {cnt} | {pct:.1f}% |")
    md_lines.append(f"| **RAZEM** | **{total_unique}** | **100.0%** |")
    md_lines.append("")

    for group_key, section_title, description, sort_col in groups_meta:
        subset = work_pivot[work_pivot["ONLY_IN"] == group_key].sort_values(
            by=[sort_col, "feature_name"], ascending=[False, True]
        )

        md_lines.extend([
            f"## {section_title} — {len(subset)} szt.",
            "",
            description,
            "",
            "| Lp. | Cecha (`feature`) (nazwa polska) | Wystąpienia KOR | Wystąpienia NEURO | Łącznie |",
            "| :---: | :--- | :---: | :---: | :---: |",
        ])

        if subset.empty:
            md_lines.append("| - | *Brak cech w tej kategorii* | 0 | 0 | 0 |")
        else:
            for idx, (feature_name, row) in enumerate(subset.iterrows(), start=1):
                label = format_single_feature_label(feature_name, polish_map).replace("|", "\\|")
                kor_val = int(row["TOTAL_KOR"])
                neuro_val = int(row["TOTAL_NEURO"])
                all_val = int(row["TOTAL_ALL"])
                md_lines.append(
                    f"| {idx} | {label} | {kor_val:,} | {neuro_val:,} | {all_val:,} |".replace(",", " ")
                )

        md_lines.append("")

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_path


# 6. WIZUALIZACJE
def visualize_results(summary_df: pd.DataFrame, pivot_df: pd.DataFrame, tag: str):
    """Tworzy wykresy dla pojedynczych cech w folderze analysis_outputs/plots/."""
    sns.set_theme(style="whitegrid", context="notebook")
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # WYKRES 1: Podział pojedynczych cech
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = pivot_df["ONLY_IN"].value_counts()
    colors = {"ONLY_NEURO": "#d95f02", "ONLY_KOR": "#7570b3", "BOTH": "#1b9e77"}
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(k, "#666666") for k in counts.index],
        edgecolor="black",
    )
    ax.set_title(f"[{tag}] Podział pojedynczych cech między kohortami", fontsize=12, fontweight="bold")
    ax.set_ylabel("Liczba pojedynczych cech")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_1_podzial_kohort_cechy.png", dpi=200)
    plt.close()

    # WYKRES 2: Top 20 najczęstszych cech
    top20 = (
        pivot_df.assign(TOTAL=pivot_df["TOTAL_KOR"] + pivot_df["TOTAL_NEURO"])
        .sort_values("TOTAL", ascending=False)
        .head(20)
        .sort_values("TOTAL", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    y = range(len(top20))
    ax.barh([i - 0.2 for i in y], top20["TOTAL_KOR"], height=0.4, label="KOR", color="#7570b3")
    ax.barh([i + 0.2 for i in y], top20["TOTAL_NEURO"], height=0.4, label="NEURO", color="#d95f02")
    ax.set_yticks(list(y))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Liczba niepustych pomiarów (skala logarytmiczna)")
    ax.set_title(f"[{tag}] Top 20 najczęstszych cech (KOR vs NEURO)", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_2_top20_cech.png", dpi=200)
    plt.close()

    # WYKRES 3: Heatmapa cech ONLY_NEURO po latach
    neuro_year_cols = sorted([c for c in pivot_df.columns if c.startswith("neuro_")])
    only_neuro = pivot_df[pivot_df["ONLY_IN"] == "ONLY_NEURO"].copy()

    if not only_neuro.empty and neuro_year_cols:
        only_neuro_top = only_neuro.sort_values("TOTAL_NEURO", ascending=False).head(25)
        heat_data = only_neuro_top[neuro_year_cols].astype(int)
        heat_data.columns = [c.replace("neuro_", "") for c in heat_data.columns]

        fig, ax = plt.subplots(figsize=(11, max(5, len(heat_data) * 0.35)))
        sns.heatmap(
            heat_data,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "Liczba pomiarów w NEURO"},
            ax=ax,
        )
        ax.set_title(f"[{tag}] Cechy występujące WYŁĄCZNIE w NEURO (wg lat)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Cecha (kolumna)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{tag}_3_heatmap_only_neuro_cechy.png", dpi=200)
        plt.close()

    # WYKRES 4: Łączna liczba niepustych pomiarów w podziale na lata i kohorty
    yearly = summary_df.groupby(["year", "cohort"])["count"].sum().unstack(fill_value=0)
    if not yearly.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        yearly.plot(kind="bar", ax=ax, color={"kor": "#7570b3", "neuro": "#d95f02"}, edgecolor="black")
        ax.set_title(f"[{tag}] Łączna liczba niepustych pomiarów w podziale na lata", fontsize=12, fontweight="bold")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Liczba pomiarów")
        ax.legend(title="Kohorta")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{tag}_4_trend_roczny_cechy.png", dpi=200)
        plt.close()

    print(f"Wygenerowano wykresy w: {plots_dir}/")


# MAIN
def main():
    print(f"Tryb: {'DANE SYNTETYCZNE' if USE_SYNTHETIC else 'PRAWDZIWE DANE (_cleaned1.csv)'}")
    df = load_cohorts()

    summary_df, pivot_df = analyze_features_by_cohort_and_year(df)
    print(f"\nLiczba analizowanych pojedynczych cech: {len(pivot_df)}")
    print("Podział występowania cech między kohortami:")
    print(pivot_df["ONLY_IN"].value_counts().to_string())

    synonyms_report = detect_synonyms(df)

    tag = "SYNTHETIC" if USE_SYNTHETIC else "FINAL"
    summary_path = OUTPUT_DIR / f"{tag}_cechy_grupa_rok.csv"
    pivot_path = OUTPUT_DIR / f"{tag}_cechy_tabela_przestawna.csv"
    synonyms_path = OUTPUT_DIR / f"{tag}_synonimy.json"

    report_path = generate_cohort_overlap_report(df, pivot_df, tag)
    visualize_results(summary_df, pivot_df, tag)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pivot_df.to_csv(pivot_path, encoding="utf-8-sig")
    with open(synonyms_path, "w", encoding="utf-8") as f:
        json.dump(synonyms_report, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano raport .md w: {report_path}")


if __name__ == "__main__":
    main()