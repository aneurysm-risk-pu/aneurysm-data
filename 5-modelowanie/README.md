> **Skąd ten folder:** przeniesione z repo `aneurysm-pu-modeling` (commit `d4181ec`, 25.11.2026), żeby nie zginęło, gdy tamto repo staje się czystym pipeline'em.
> Zmiany względem tamtego repo: `modelowanie/` jest tu jako `5-modelowanie/`, `prototyp_wrzesien/` jako `5-modelowanie/prototyp_wrzesien/`, a opisy zadań z `docs/` są w `docs/sprints/sprint6/zadania/`.
> Kopie `3-sgkf-split/`, `4-pu-setup/` i `narzedzia/dane_syntetyczne/` były identyczne z `master` @ `d6d9e58`, więc ich tu nie ma.
> **Uwaga:** w tym repo `data/processed/aneurysm_sgkf_input.csv` to prawdziwe dane. Dane syntetyczne są w `data_syntetyczne/` (`--dane data_syntetyczne/processed/aneurysm_sgkf_input.csv`).

# Modelowanie PU (praca robocza)

Repo do zadań #29–#33 ze Sprintu 6 (zasady oceny modeli). Pracujemy tu **wyłącznie na danych syntetycznych**. Kod ma działać bez zmian na prawdziwych danych: na koniec trafia przez PR do repo zespołu (`aneurysm-risk-pu/aneurysm-data`, np. jako `5-modelowanie/`) i tam liczymy wyniki do raportu.

**Zasady:**
- tu nie ma prawdziwych danych ani wyników z nich,
- liczb stąd nie raportujemy, bo pokazują tylko, czy kod działa,
- fold 4 jest testowy (D1), a kod roboczy go nie dotyka.

## Start

```bash
python narzedzia/dane_syntetyczne/generuj.py --wyjscie data       # dane syntetyczne (ok. 1 min)
python 3-sgkf-split/sgkf/podzial.py                                 # podział na foldy dla tych danych
python modelowanie/tests/test_modelowanie.py                        # testy funkcji oceny
cd modelowanie && python uruchom_ocene.py                           # modele kontrolne + funkcja oceny (ok. 4 min)
```

Dane do sprawdzania kodu w innych trybach:

```bash
python narzedzia/dane_syntetyczne/generuj.py --tryb brak --wyjscie data_brak
cd modelowanie && python uruchom_ocene.py --dane data_brak/processed/aneurysm_sgkf_input.csv --nazwa brak
```

Wersje bibliotek: `requirements.txt`.

## Co jest w repo

| Folder | Co to jest | Skąd |
|---|---|---|
| `modelowanie/` | **nasz kod**: dane i foldy, listy ukrytych, profil pacjenta, funkcja oceny, modele kontrolne, testy | nowy |
| `docs/` | krótki opis każdego zadania: co zrobione, na co uważać, co sprawdzić po zadaniach zespołu | nowy |
| `3-sgkf-split/` | podział na foldy i imputacja z etapu 3 (używamy, nie zmieniamy) | `aneurysm-data`, `master` @ `d6d9e58` |
| `4-pu-setup/` | plan i opcje modelowania, analiza agregacji | jw. |
| `narzedzia/dane_syntetyczne/` | generator danych syntetycznych | jw. |
| `prototyp_wrzesien/` | prototyp PU z 25.09 (nie uruchamiamy, tylko źródło pomysłów) | commit `2cb6d1a` |

## Kroki i zadania

| Krok | Plik | Zadanie | Stan |
|---|---|---|---|
| 1. Dane i układy foldów | `modelowanie/dane.py` | wersja tymczasowa #38 (Adrian) | gotowe |
| 2. Listy ukrytych | `modelowanie/ukrywanie.py` | wersja tymczasowa #39 (Adrian) | gotowe |
| 3. Profil pacjenta | `modelowanie/profil.py`, `docs/30_profil_pacjenta.md` | **#30** | propozycja |
| 4. Funkcja oceny | `modelowanie/ocena.py`, `docs/33_funkcja_oceny.md` | **#33** | gotowe, z testami |
| 5. Imputacja | nie robimy | Dominika (#47, #51) | pierwsze modele obsługują braki bez imputacji |
| 6. Modele kontrolne | `modelowanie/kontrolne.py`, `uruchom_ocene.py`, `docs/31_modele_kontrolne.md` | **#31** | gotowe bez „nazw badań” |
| 7. Zasady oceny | `docs/29_zasady_oceny.md` | **#29** | szkic na spotkanie |
| 8. Przegląd modeli | `docs/32_przeglad_modeli.md` | **#32** | do zrobienia |

## Do sprawdzenia, gdy zespół skończy swoje zadania

| Nasz krok | Zadanie zespołu | Co wtedy robimy |
|---|---|---|
| 1. Dane i foldy | #38, #37 (Adrian) | zastępujemy `dane.uklady()` wersją zespołu albo ujednolicamy interfejs |
| 2. Listy ukrytych | #39 (Adrian) | podmieniamy `listy_ukrytych()` na listy zamrożone w repo zespołu |
| 5. Imputacja | #47, #51 (Dominika) | dla modeli, które nie znoszą braków (np. regresja logistyczna), bierzemy jej foldy i standaryzację |
| 6. Modele kontrolne | #41 (Liwia), #49, #50 (Dominika) | dodajemy kontrolę „nazwy badań” i ocenę w grupach o podobnych brakach |
| 3 i 7 | #40 (Adrian), #45 (Liwia), odpowiedzi z #34 | poprawiamy zasady, np. jeśli silny okazuje się wpływ okresu badań |
| wszystko | #36 (Adrian) | warunek przeniesienia kodu do repo zespołu i liczenia na prawdziwych danych |
