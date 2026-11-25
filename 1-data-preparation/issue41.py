"""
Analiza nazw zleconych badań (kolumna examination_type) w danych medycznych:
1. Rozkład zleceń dla każdej nazwy z examination_type w podziale na kohortę (KOR vs NEURO) i rok.
2. Wykrywanie synonimów i zduplikowanych nazw oznaczających to samo badanie
   (analiza leksykalna + sygnatury mierzonych parametrów).
3. Generowanie raportu .md oraz zestawień .csv/.json.
4. Eksport gotowej macierzy cech (X, y) dla modelu kontrolnego „tylko nazwy badań”.
"""

import ast
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. KONFIGURACJA ŚCIEŻEK
USE_SYNTHETIC = False

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / ("data_syntetyczne" if USE_SYNTHETIC else "data")
CLEANED_DIR = BASE_DIR / "interim"

FILES_MAP = {
    "kor": BASE_DIR / "raw/kor_merged_aggregated_1W_mean.csv",
    "neuro": BASE_DIR / "raw/neuro_merged_aggregated_1W_mean.csv",
}

OUTPUT_DIR = REPO_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {
    "cohort", "year", "examination_type", "patient_id", "custom_id",
    "patient_age", "patient_sex", "diagnosis_id", "descriptive_result",
    "date", "week", "examination_date", "nan_count",
}


# 2. PARSOWANIE I EKSTRAKCJA DANYCH
def parse_examination_types(cell_value) -> list[str]:
    """
    Parsuje kolumnę examination_type z formatu słownikowego {'values': [...]}
    lub listy rozdzielonej przecinkami (z zachowaniem przecinków wewnątrz nawiasów).
    """
    if pd.isna(cell_value):
        return []
    if isinstance(cell_value, dict):
        return [str(v).strip() for v in cell_value.get("values", []) if str(v).strip()]
    if isinstance(cell_value, str):
        text = cell_value.strip()
        if not text:
            return []
        if text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return [str(v).strip() for v in parsed.get("values", []) if str(v).strip()]
            except (ValueError, SyntaxError):
                pass
        parts = re.split(r",(?![^()]*\))", text)
        return [p.strip() for p in parts if p.strip()]
    return []


def extract_year(df: pd.DataFrame) -> pd.Series:
    """Wyciąga rok z kolumny examination_date, date, week lub custom_id."""
    for date_col in ["examination_date", "date", "week"]:
        if date_col in df.columns:
            years = pd.to_datetime(df[date_col], errors="coerce").dt.year
            if years.notna().any():
                return years.astype("Int64")

    if "custom_id" in df.columns:
        return (
            df["custom_id"]
            .astype(str)
            .str.extract(r"-(\d{4})-")[0]
            .astype("Int64")
        )

    raise KeyError(f"Brak kolumny z datą/rokiem. Dostępne: {list(df.columns)}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Zwraca listę kolumn z wynikami liczbowymi (bez metadanych, -unit i -norm)."""
    return [
        c for c in df.columns
        if c not in META_COLS and not c.endswith("-unit") and not c.endswith("-norm")
    ]


def load_cohorts() -> pd.DataFrame:
    """Wczytuje obie kohorty i parsuje listę zleceń w każdym wierszu."""
    dfs = []
    for cohort_name, file_path in FILES_MAP.items():
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Brak pliku: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        df["cohort"] = cohort_name
        df["year"] = extract_year(df)
        df["exam_list"] = df["examination_type"].apply(parse_examination_types)
        dfs.append(df)

        print(f"Wczytano {cohort_name.upper()} ({file_path.name}): {len(df)} rekordów.")

    return pd.concat(dfs, ignore_index=True)


# 3. ANALIZA 1: NAZWY ZLECEŃ WEDŁUG KOHORTY I ROKU
def analyze_exams_by_cohort_and_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rozbija listy z examination_type (explode) i zlicza wystąpienia każdej nazwy zlecenia
    w podziale na (cohort, year).
    """
    exploded = (
        df[["cohort", "year", "exam_list"]]
        .explode("exam_list")
        .dropna(subset=["exam_list"])
        .rename(columns={"exam_list": "exam_name"})
    )
    exploded = exploded[exploded["exam_name"] != ""]

    summary = (
        exploded.groupby(["exam_name", "cohort", "year"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["exam_name", "cohort", "year"])
    )

    pivot = summary.pivot_table(
        index="exam_name",
        columns=["cohort", "year"],
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = [f"{cohort}_{year}" for cohort, year in pivot.columns]

    kor_cols = sorted([c for c in pivot.columns if c.startswith("kor_")])
    neuro_cols = sorted([c for c in pivot.columns if c.startswith("neuro_")])

    total_kor = pivot[kor_cols].sum(axis=1) if kor_cols else pd.Series(0, index=pivot.index)
    total_neuro = pivot[neuro_cols].sum(axis=1) if neuro_cols else pd.Series(0, index=pivot.index)

    # Zakres lat aktywności dla każdej nazwy w danej kohorcie
    def get_active_years(row, cols):
        active = [c.split("_")[1] for c in cols if row[c] > 0]
        if not active:
            return "-"
        return f"{min(active)}–{max(active)} ({', '.join(active)})" if len(active) > 1 else active[0]

    pivot["LATA_KOR"] = pivot.apply(lambda r: get_active_years(r, kor_cols), axis=1)
    pivot["LATA_NEURO"] = pivot.apply(lambda r: get_active_years(r, neuro_cols), axis=1)
    pivot["TOTAL_KOR"] = total_kor
    pivot["TOTAL_NEURO"] = total_neuro
    pivot["TOTAL_ALL"] = total_kor + total_neuro
    pivot["ONLY_IN"] = np.where(
        (total_kor == 0) & (total_neuro > 0),
        "ONLY_NEURO",
        np.where((total_neuro == 0) & (total_kor > 0), "ONLY_KOR", "BOTH"),
    )

    return summary, pivot


# 4. ANALIZA 2: WYKRYWANIE SYNONIMÓW W EXAMINATION_TYPE
def normalize_exam_text(name: str) -> str:
    """Normalizuje nazwę badania do postaci bazowej (usuwa nawiasy, dopiski 'we krwi', polskie znaki)."""
    text = name.lower().strip()
    # Usunięcie dopisków w nawiasach np. "Kreatynina (surowica)" -> "kreatynina"
    text = re.sub(r"\([^)]*\)", "", text)
    # Usunięcie popularnych sufiksów lokalizacyjnych
    for suffix in ["we krwi", "w surowicy", "w osoczu", "w moczu", "metodą", "oznaczenie"]:
        text = text.replace(suffix, "")
    text = re.sub(r"[^a-z0-9ąćęłńóśźż\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Transliteracja polskich znaków do porównań
    text_ascii = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text_ascii


def detect_exam_synonyms(df: pd.DataFrame, pivot_df: pd.DataFrame) -> dict:
    """
    Identyfikuje różne nazwy oznaczające to samo badanie za pomocą 3 metod:
    1. Normalizacja leksykalna (np. 'Kreatynina' vs 'Kreatynina we krwi').
    2. Identyczna sygnatura mierzonych parametrów (dla rekordów z pojedynczym zleceniem).
    3. Mapowanie nazwy zlecenia na dominujące kolumny wynikowe (nawet w pakietach).
    """
    unique_names = sorted(pivot_df.index.tolist())
    feature_cols = get_feature_columns(df)

    # Metoda 1: Grupowanie leksykalne
    lexical_groups = defaultdict(list)
    for name in unique_names:
        norm = normalize_exam_text(name)
        lexical_groups[norm].append(name)

    lexical_synonyms = {
        norm_key: names for norm_key, names in lexical_groups.items() if len(names) > 1
    }

    # Przygotowanie danych dla pojedynczych zleceń w wierszu (najczystszy sygnał parametrów)
    single_df = df[df["exam_list"].str.len() == 1].copy()
    single_df["single_name"] = single_df["exam_list"].str[0]

    # Słownik: nazwa badania -> lista mierzonych kolumn (sygnatura)
    exam_to_signature = {}
    sig_to_names = defaultdict(list)

    if not single_df.empty and feature_cols:
        presence = (
            single_df.groupby("single_name")[feature_cols]
            .apply(lambda g: g.notna().mean())
        )
        for exam_name, row in presence.iterrows():
            # Cechy obecne w >80% wystąpień tego pojedynczego zlecenia
            active = tuple(sorted(row[row > 0.8].index.tolist()))
            if active:
                exam_to_signature[exam_name] = list(active)
                sig_to_names[active].append(exam_name)

    signature_synonyms = {
        ", ".join(feats): names
        for feats, names in sig_to_names.items()
        if len(names) > 1
    }

    # Automatyczne budowanie słownika kanonicznego (stara_nazwa -> nazwa_kanoniczna)
    canonical_map = {}

    # Najpierw łączymy po identycznej sygnaturze parametrów (najbardziej wiarygodne klinicznie)
    for _, names in signature_synonyms.items():
        # Wybieramy nazwę najczęstszą w całym zbiorze jako kanoniczną
        best_name = max(names, key=lambda n: pivot_df.loc[n, "TOTAL_ALL"])
        for n in names:
            if n != best_name:
                canonical_map[n] = best_name

    # Następnie uzupełniamy oczywistymi synonimami leksykalnymi
    for _, names in lexical_synonyms.items():
        best_name = max(names, key=lambda n: pivot_df.loc[n, "TOTAL_ALL"])
        for n in names:
            if n not in canonical_map and n != best_name:
                canonical_map[n] = best_name

    # Szczegółowy profil każdego badania (kohorty + sygnatura kolumn)
    exam_profiles = {}
    for name in unique_names:
        row = pivot_df.loc[name]
        exam_profiles[name] = {
            "wystepowanie": row["ONLY_IN"],
            "kor_count": int(row["TOTAL_KOR"]),
            "neuro_count": int(row["TOTAL_NEURO"]),
            "lata_kor": row["LATA_KOR"],
            "lata_neuro": row["LATA_NEURO"],
            "mierzone_parametry_single": exam_to_signature.get(name, []),
            "sugerowana_nazwa_kanoniczna": canonical_map.get(name, name),
        }

    return {
        "synonimy_po_sygnaturze_parametrow": signature_synonyms,
        "synonimy_leksykalne": lexical_synonyms,
        "slownik_mapowania_kanonicznego": canonical_map,
        "profil_kazdego_badania": exam_profiles,
    }


# 5. PRZYGOTOWANIE ZBIORU DO MODELU KONTROLNEGO („TYLKO NAZWY”)
def build_name_only_control_dataset(
    df: pd.DataFrame, canonical_map: dict[str, str], tag: str
) -> tuple[Path, Path]:
    """
    Tworzy gotowe zbiory cech dla modelu kontrolnego opartego WYŁĄCZNIE na nazwach zleconych badań:
    1. Wersja RAW (oryginalne nazwy z examination_type) — testuje surowy wyciek danych.
    2. Wersja CANONICAL (po scaleniu synonimów) — testuje, czy sam fakt zlecenia danego typu
       badania (niezależnie od nazewnictwa oddziałowego) odróżnia KOR od NEURO.
    """
    meta_keep = [c for c in ["patient_id", "custom_id", "cohort", "year", "week", "date"] if c in df.columns]

    def encode_exam_lists(exam_lists: pd.Series, prefix: str) -> pd.DataFrame:
        all_names = sorted({n for lst in exam_lists for n in lst})
        data = {}
        for name in all_names:
            safe_col = re.sub(r"[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ_]+", "_", name).strip("_")
            # Zliczamy ile razy w danym tygodniu/rekordzie zlecono badanie (0 = brak)
            data[f"{prefix}_cnt__{safe_col}"] = exam_lists.apply(lambda lst: lst.count(name))
            data[f"{prefix}_bin__{safe_col}"] = (data[f"{prefix}_cnt__{safe_col}"] > 0).astype(int)
        return pd.DataFrame(data, index=exam_lists.index)

    # 1. Macierz dla nazw surowych
    X_raw = encode_exam_lists(df["exam_list"], prefix="raw")
    ds_raw = pd.concat([df[meta_keep], X_raw], axis=1)
    raw_path = OUTPUT_DIR / f"{tag}_model_kontrolny_TYLKO_NAZWY_raw.csv"
    ds_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # 2. Macierz dla nazw po scaleniu synonimów
    canonical_lists = df["exam_list"].apply(
        lambda lst: [canonical_map.get(n, n) for n in lst]
    )
    X_canon = encode_exam_lists(canonical_lists, prefix="canon")
    ds_canon = pd.concat([df[meta_keep], X_canon], axis=1)
    canon_path = OUTPUT_DIR / f"{tag}_model_kontrolny_TYLKO_NAZWY_canonical.csv"
    ds_canon.to_csv(canon_path, index=False, encoding="utf-8-sig")

    return raw_path, canon_path


# 6. GENEROWANIE RAPORTU MARKDOWN (.MD)
def generate_exam_report(pivot_df: pd.DataFrame, synonyms_report: dict, tag: str) -> Path:
    """Zapisuje kompletny raport pokrycia nazw z examination_type, lat występowania i synonimów."""
    report_path = OUTPUT_DIR / f"{tag}_raport_examination_type.md"
    profiles = synonyms_report["profil_kazdego_badania"]

    md_lines = [
        f"# Analiza zleconych badań (`examination_type`) między grupami KOR i NEURO [{tag}]",
        "",
        "## 1. Podsumowanie pokrycia nazw zleceń",
        "",
        "| Kategoria | Liczba unikalnych nazw | Udział (%) |",
        "| :--- | :---: | :---: |",
    ]

    total = len(pivot_df)
    for key in ["BOTH", "ONLY_KOR", "ONLY_NEURO"]:
        cnt = int((pivot_df["ONLY_IN"] == key).sum())
        pct = (cnt / total * 100) if total > 0 else 0.0
        md_lines.append(f"| `{key}` | {cnt} | {pct:.1f}% |")
    md_lines.append(f"| **RAZEM** | **{total}** | **100.0%** |")
    md_lines.append("")

    # Sekcja 2: Wykryte synonimy
    md_lines.extend([
        "## 2. Wykryte synonimy (różne nazwy oznaczające to samo badanie)",
        "",
        "### A. Synonimy potwierdzone identyczną sygnaturą mierzonych parametrów (>80% zgodności)",
        "",
    ])
    sig_syn = synonyms_report["synonimy_po_sygnaturze_parametrow"]
    if not sig_syn:
        md_lines.append("*Brak grup o identycznej sygnaturze.*")
    else:
        for feats, names in sig_syn.items():
            names_fmt = ", ".join(f"`{n}` ({profiles[n]['wystepowanie']})" for n in names)
            md_lines.append(f"- **Parametry `[{feats}]`**: {names_fmt}")
    md_lines.append("")

    md_lines.extend([
        "### B. Kandydaci leksykalni (podobna nazwa po normalizacji)",
        "",
    ])
    lex_syn = synonyms_report["synonimy_leksykalne"]
    if not lex_syn:
        md_lines.append("*Brak dodatkowych kandydatów leksykalnych.*")
    else:
        for core, names in lex_syn.items():
            names_fmt = ", ".join(f"`{n}` ({profiles[n]['wystepowanie']})" for n in names)
            md_lines.append(f"- **Rdzeń `{core}`**: {names_fmt}")
    md_lines.append("")

    # Sekcja 3: Szczegółowe tabele dla BOTH, ONLY_KOR, ONLY_NEURO z latami
    groups_meta = [
        ("ONLY_NEURO", "3A. Badania zlecane WYŁĄCZNIE w grupie NEURO (`ONLY_NEURO`)", "TOTAL_NEURO"),
        ("ONLY_KOR", "3B. Badania zlecane WYŁĄCZNIE w grupie KOR (`ONLY_KOR`)", "TOTAL_KOR"),
        ("BOTH", "3C. Badania zlecane w OBU grupach (`BOTH`)", "TOTAL_ALL"),
    ]

    for group_key, title, sort_col in groups_meta:
        subset = pivot_df[pivot_df["ONLY_IN"] == group_key].sort_values(
            by=[sort_col, "exam_name"], ascending=[False, True]
        )
        md_lines.extend([
            f"## {title} — {len(subset)} szt.",
            "",
            "| Lp. | Nazwa w `examination_type` | KOR (liczba) | Lata KOR | NEURO (liczba) | Lata NEURO | Mierzone parametry (single) |",
            "| :---: | :--- | :---: | :--- | :---: | :--- | :--- |",
        ])

        if subset.empty:
            md_lines.append("| - | *Brak* | 0 | - | 0 | - | - |")
        else:
            for idx, (exam_name, row) in enumerate(subset.iterrows(), start=1):
                safe_name = exam_name.replace("|", "\\|")
                params = ", ".join(profiles[exam_name]["mierzone_parametry_single"]) or "-"
                md_lines.append(
                    f"| {idx} | `{safe_name}` | {int(row['TOTAL_KOR']):,} | {row['LATA_KOR']} | "
                    f"{int(row['TOTAL_NEURO']):,} | {row['LATA_NEURO']} | `{params}` |".replace(",", " ")
                )
        md_lines.append("")

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_path


# 7. WIZUALIZACJE
def visualize_exam_results(summary_df: pd.DataFrame, pivot_df: pd.DataFrame, tag: str):
    """Generuje wykresy dla nazw zleceń (examination_type)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Wykres 1: Podział nazw zleceń między kohortami
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = pivot_df["ONLY_IN"].value_counts()
    colors = {"ONLY_NEURO": "#d95f02", "ONLY_KOR": "#7570b3", "BOTH": "#1b9e77"}
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(k, "#666666") for k in counts.index],
        edgecolor="black",
    )
    ax.set_title(f"[{tag}] Unikalne nazwy w examination_type (KOR vs NEURO)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Liczba unikalnych nazw badań")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_1_podzial_examination_type.png", dpi=200)
    plt.close()

    # Wykres 2: Heatmapa obecności nazw badań w podziale na (kohorta_rok) dla Top 30
    year_cols = sorted([c for c in pivot_df.columns if c.startswith("kor_") or c.startswith("neuro_")])
    if year_cols:
        top30 = pivot_df.sort_values("TOTAL_ALL", ascending=False).head(30)
        heat_data = np.log1p(top30[year_cols].astype(float))

        fig, ax = plt.subplots(figsize=(13, max(6, len(top30) * 0.35)))
        sns.heatmap(
            heat_data,
            annot=top30[year_cols].astype(int),
            fmt="d",
            cmap="YlGnBu",
            cbar_kws={"label": "log(1 + liczba zleceń)"},
            ax=ax,
        )
        ax.set_title(f"[{tag}] Top 30 nazw z examination_type w podziale na kohortę i rok", fontsize=12, fontweight="bold")
        ax.set_xlabel("Kohorta i rok")
        ax.set_ylabel("Nazwa zlecenia (examination_type)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{tag}_2_heatmap_examination_type_lata.png", dpi=200)
        plt.close()


# MAIN
def main():
    print(f"Tryb: {'DANE SYNTETYCZNE' if USE_SYNTHETIC else 'PRAWDZIWE DANE (raw aggregated 1W)'}")
    df = load_cohorts()

    summary_df, pivot_df = analyze_exams_by_cohort_and_year(df)
    print(f"\nLiczba unikalnych nazw w examination_type: {len(pivot_df)}")
    print("Podział występowania nazw zleceń między kohortami:")
    print(pivot_df["ONLY_IN"].value_counts().to_string())

    synonyms_report = detect_exam_synonyms(df, pivot_df)

    tag = "SYNTHETIC" if USE_SYNTHETIC else "FINAL"
    summary_path = OUTPUT_DIR / f"{tag}_examination_type_grupa_rok.csv"
    pivot_path = OUTPUT_DIR / f"{tag}_examination_type_tabela_przestawna.csv"
    synonyms_path = OUTPUT_DIR / f"{tag}_examination_type_synonimy.json"

    report_path = generate_exam_report(pivot_df, synonyms_report, tag)
    visualize_exam_results(summary_df, pivot_df, tag)
    raw_ds_path, canon_ds_path = build_name_only_control_dataset(
        df, synonyms_report["slownik_mapowania_kanonicznego"], tag
    )

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pivot_df.to_csv(pivot_path, encoding="utf-8-sig")
    with open(synonyms_path, "w", encoding="utf-8") as f:
        json.dump(synonyms_report, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano raport .md: {report_path}")
    print(f"Zapisano macierze do modelu kontrolnego 'tylko nazwy':")
    print(f"  - surowe nazwy:      {raw_ds_path}")
    print(f"  - scalone synonimy:  {canon_ds_path}")


if __name__ == "__main__":
    main()