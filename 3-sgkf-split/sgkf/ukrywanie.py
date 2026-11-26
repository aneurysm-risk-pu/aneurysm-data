"""Budowa i weryfikacja zamrozonej listy ukrytych pacjentow NEURO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from podzial import (FOLD_TESTOWY, KOL_ETYKIETA, KOL_PACJENT, WYNIKI,  # noqa: E402
                     foldy_do_oceny, odcisk_pliku, wczytaj_podzial, wczytaj_rekordy)

UDZIAL_UKRYTYCH = 0.40
ZIARNA = tuple(range(5))
PLIK_MASK = WYNIKI / "ukryci_pacjenci.csv"
PLIK_META = WYNIKI / "ukryci_pacjenci_meta.json"
KOLUMNY_MASKI = ["seed", KOL_PACJENT, "fold"]


def zbuduj_maski_ukrytych(
    df: pd.DataFrame,
    przydzial: pd.Series,
    udzial: float = UDZIAL_UKRYTYCH,
    ziarna: tuple[int, ...] = ZIARNA,
    test_koncowy: bool = False,
) -> pd.DataFrame:
    """Losuje staly odsetek w foldach roboczych; fold 4 wymaga jawnego trybu koncowego."""
    if not 0 < udzial < 1:
        raise ValueError("udzial ukrytych musi byc z zakresu (0, 1)")
    if not ziarna or len(set(ziarna)) != len(ziarna):
        raise ValueError("ziarna musza byc niepusta i niepowtarzalna lista")
    foldy = foldy_do_oceny(przydzial, test_koncowy=test_koncowy)
    if test_koncowy:
        foldy = sorted(set(foldy_do_oceny(przydzial)) | {FOLD_TESTOWY})

    pacjenci_dostepni = przydzial.index[przydzial.isin(foldy)]
    dane_maski = df[df[KOL_PACJENT].isin(pacjenci_dostepni)]
    etykiety = dane_maski.groupby(KOL_PACJENT)[KOL_ETYKIETA].max()
    fold_pacjentow = przydzial.reindex(etykiety.index)
    if fold_pacjentow.isna().any():
        raise ValueError("przydzial nie obejmuje wszystkich pacjentow w danych")
    chorzy = etykiety[etykiety == 1]
    wynik = []
    for seed in ziarna:
        for fold in foldy:
            ids = np.sort(chorzy.index[fold_pacjentow.loc[chorzy.index].to_numpy() == fold].to_numpy())
            n_ukrytych = int(round(udzial * len(ids)))
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), fold]))
            wybrani = np.sort(rng.choice(ids, size=n_ukrytych, replace=False))
            wynik.extend((int(seed), patient_id, fold) for patient_id in wybrani)
    return pd.DataFrame(wynik, columns=KOLUMNY_MASKI).sort_values(
        ["seed", "fold", KOL_PACJENT], kind="stable").reset_index(drop=True)


def _oczekiwana_meta(meta_podzialu: dict) -> dict:
    return {
        "udzial_ukrytych": UDZIAL_UKRYTYCH,
        "ziarna": list(ZIARNA),
        "regula_zaokraglenia": "round(udzial * liczba chorych w foldzie)",
        "losowanie": "numpy.default_rng(SeedSequence([seed, fold])), bez zwracania",
        "odcisk_danych_sha256": meta_podzialu["odcisk_danych_sha256"],
        "odcisk_przydzialu_sha256": meta_podzialu["odcisk_przydzialu_sha256"],
    }


def zamroz_lub_wczytaj_maski(
    df: pd.DataFrame,
    przydzial: pd.Series,
    meta_podzialu: dict,
) -> pd.DataFrame:
    """Zapisuje maski tylko raz; kolejne uruchomienia sprawdzaja zgodnosc, nie nadpisuja."""
    oczekiwane = zbuduj_maski_ukrytych(df, przydzial)
    oczekiwana_meta = _oczekiwana_meta(meta_podzialu)

    if PLIK_MASK.exists() != PLIK_META.exists():
        raise RuntimeError("niepelny zapis maski ukrytych pacjentow; sprawdz oba pliki wynikowe")
    if PLIK_MASK.exists():
        meta_maski = json.loads(PLIK_META.read_text(encoding="utf-8"))
        if (
            "test_koncowy" not in meta_maski
            and ("foldy" not in meta_maski or FOLD_TESTOWY in meta_maski["foldy"])
        ):
            if (
                any(meta_maski.get(k) != v for k, v in oczekiwana_meta.items())
                or meta_maski.get("odcisk_listy_sha256") != odcisk_pliku(PLIK_MASK)
            ):
                raise RuntimeError("nie mozna bezpiecznie zmigrowac starej listy ukrytych pacjentow")
            oczekiwane.to_csv(PLIK_MASK, index=False)
            meta = {**oczekiwana_meta, "foldy": foldy_do_oceny(przydzial), "test_koncowy": False,
                    "odcisk_listy_sha256": odcisk_pliku(PLIK_MASK)}
            PLIK_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return oczekiwane
        return _wczytaj_zapisane_maski(
            oczekiwane,
            {**oczekiwana_meta, "foldy": foldy_do_oceny(przydzial), "test_koncowy": False},
        )

    PLIK_MASK.parent.mkdir(parents=True, exist_ok=True)
    oczekiwane.to_csv(PLIK_MASK, index=False)
    meta = {**oczekiwana_meta, "foldy": foldy_do_oceny(przydzial), "test_koncowy": False,
            "odcisk_listy_sha256": odcisk_pliku(PLIK_MASK)}
    PLIK_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return oczekiwane


def wczytaj_maski(
    df: pd.DataFrame,
    przydzial: pd.Series,
    meta_podzialu: dict,
    test_koncowy: bool = False,
) -> pd.DataFrame:
    if not PLIK_MASK.exists() or not PLIK_META.exists():
        raise RuntimeError("brak zamrozonych masek; uruchom najpierw 3-sgkf-split/sgkf/ukrywanie.py")
    oczekiwane = zbuduj_maski_ukrytych(df, przydzial, test_koncowy=test_koncowy)
    meta_oczekiwana = {
        **_oczekiwana_meta(meta_podzialu),
        "foldy": (sorted(set(foldy_do_oceny(przydzial)) | {FOLD_TESTOWY})
                  if test_koncowy else foldy_do_oceny(przydzial)),
        "test_koncowy": test_koncowy,
    }
    return _wczytaj_zapisane_maski(oczekiwane, meta_oczekiwana)


def _wczytaj_zapisane_maski(oczekiwane: pd.DataFrame, oczekiwana_meta: dict) -> pd.DataFrame:
    maski = pd.read_csv(PLIK_MASK)
    meta_maski = json.loads(PLIK_META.read_text(encoding="utf-8"))
    if meta_maski.get("odcisk_listy_sha256") != odcisk_pliku(PLIK_MASK):
        raise RuntimeError("lista ukrytych pacjentow zmienila sie od zamrozenia")
    klucze_stale = set(oczekiwana_meta) - {"foldy", "test_koncowy"}
    if any(meta_maski.get(k) != oczekiwana_meta[k] for k in klucze_stale):
        raise RuntimeError("parametry lub dane listy ukrytych pacjentow nie zgadzaja sie z zamrozeniem")
    foldy_oczekiwane = oczekiwana_meta["foldy"]
    if oczekiwana_meta["test_koncowy"]:
        if meta_maski.get("test_koncowy") is not True or meta_maski.get("foldy") != foldy_oczekiwane:
            raise RuntimeError("brak jawnie zamrozonej maski do testu koncowego")
    elif meta_maski.get("test_koncowy") not in (False, True) or not set(foldy_oczekiwane).issubset(
        set(meta_maski.get("foldy", []))
    ):
        raise RuntimeError("lista ukrytych pacjentow nie obejmuje wymaganych foldow roboczych")
    foldy_porownania = oczekiwana["fold"].unique().tolist()
    wybrane = maski[maski["fold"].isin(foldy_porownania)].reset_index(drop=True)
    if not wybrane.equals(oczekiwane):
        raise RuntimeError("lista ukrytych pacjentow rozni sie od odtwarzalnego losowania")
    return wybrane


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-koncowy", action="store_true",
                        help="jawnie utworz/wczytaj maski zawierajace fold 4")
    args = parser.parse_args(argv)
    przydzial, meta = wczytaj_podzial()
    df = wczytaj_rekordy()
    if args.test_koncowy:
        maski_robocze = zamroz_lub_wczytaj_maski(df, przydzial, meta)
        maski_testowe = zbuduj_maski_ukrytych(df, przydzial, test_koncowy=True)
        maski = pd.concat(
            [maski_robocze, maski_testowe[maski_testowe["fold"] == FOLD_TESTOWY]],
            ignore_index=True,
        ).sort_values(["seed", "fold", KOL_PACJENT], kind="stable").reset_index(drop=True)
        maski.to_csv(PLIK_MASK, index=False)
        meta_maski = {**_oczekiwana_meta(meta),
                      "foldy": sorted(set(foldy_do_oceny(przydzial)) | {FOLD_TESTOWY}),
                      "test_koncowy": True,
                      "odcisk_listy_sha256": odcisk_pliku(PLIK_MASK)}
        PLIK_META.write_text(json.dumps(meta_maski, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        maski = zamroz_lub_wczytaj_maski(df, przydzial, meta)
    print(f"Zweryfikowano zamrozone listy dla {len(ZIARNA)} ziaren "
          f"(40% chorych w kazdym foldzie); odcisk {odcisk_pliku(PLIK_MASK)}.")


if __name__ == "__main__":
    main()
