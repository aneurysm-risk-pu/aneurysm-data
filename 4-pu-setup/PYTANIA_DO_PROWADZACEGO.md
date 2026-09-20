# Pytania i założenia do potwierdzenia

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Adresat:** dr inż. Patryk Jasik, ewentualnie właściciel danych klinicznych
**Podstawa:** diagnostyka z `4-pu-setup/ETAP0_USTALENIA.md`
**Aktualizacja:** 20.09.2026 — po ustaleniach zespołu

Dokument zawiera jedno założenie przyjęte roboczo przez zespół oraz dwie kwestie do omówienia na najbliższym spotkaniu.

---

## Założenie robocze: wyniki NEURO pochodzą z hospitalizacji

**Co zakładamy.** Pacjenci NEURO to osoby z **już wykrytym** tętniakiem, a ich wyniki laboratoryjne pochodzą z hospitalizacji, w trakcie której tętniaka rozpoznano i leczono — nie z okresu przed rozpoznaniem.

**Co za tym przemawia w danych.** Spośród 1 823 pacjentów NEURO 1 215 ma więcej niż jeden rekord. U nich mediana rozpiętości badań wynosi 35 dni, przy czym 48,9% mieści się w 30 dniach, a 58,2% w 90. To wzorzec pojedynczego epizodu szpitalnego, a nie obserwacji ambulatoryjnej rozłożonej w czasie.

**Co z tego wynika dla wniosków.** Model uczony na takich danych rozpoznaje **profil pacjenta hospitalizowanego z rozpoznanym tętniakiem**, a nie profil ryzyka osoby jeszcze niezdiagnozowanej. Część sygnału może pochodzić z samej hospitalizacji: zabiegu, podania kontrastu, stresu okołooperacyjnego czy reakcji zapalnej. W raporcie nie wolno więc opisywać wyniku jako modelu przesiewowego bez tego zastrzeżenia.

**Czego to nie zmienia.** Protokół, pipeline, foldy i metryki pozostają bez zmian. Założenie wpływa wyłącznie na interpretację i na sposób opisania wyników.

**Pytanie do potwierdzenia.** Czy wyniki laboratoryjne pacjentów NEURO pochodzą z okresu przed rozpoznaniem tętniaka, czy z hospitalizacji, w trakcie której go rozpoznano i leczono?

---

## Do omówienia na spotkaniu

### 1. Rozjazd czasowy kohort

Kohorta KOR mieści się praktycznie w całości w latach 2020–2021 (99,5% rekordów), podczas gdy NEURO rozciąga się na lata 2000–2024. W oknie dat KOR mieści się tylko 680 z 6 651 rekordów NEURO, czyli 271 z 1 823 pacjentów. Porównanie median cech między NEURO z tego okna a NEURO spoza niego daje różnice tego samego rzędu co różnice między kohortami, a dla sodu, potasu i RDW większe.

Ryzyko: model może rozpoznawać okres, laboratorium albo tryb pozyskania danych zamiast choroby. Do rozstrzygnięcia, czy analiza główna zostaje na całości z jawną kontrolą czasu i źródła, czy przenosimy się do wspólnego okna dat, co zostawia 271 pacjentów pozytywnych zamiast 1 823.

### 2. Jednostki kreatyniny

Mediana KREA wynosi 0,87, co wskazuje na mg/dl, ale 767 wartości (1,19%) przekracza 15, a 210 wartości (0,33%) przekracza 50 — to zakres typowy dla µmol/l. Problem występuje w obu kohortach w podobnej proporcji.

Automatyczne przeliczenie wszystkiego powyżej wybranego progu skasowałoby prawdziwe przypadki ciężkiej niewydolności nerek, a pozostawienie danych bez zmian wprowadza szum do istotnej klinicznie cechy. Potrzebna informacja, czy w systemie źródłowym zapisana jest jednostka konkretnego oznaczenia.

### 3. Status 63 pacjentów obecnych w obu kohortach

63 pacjentów ma rekordy zarówno w KOR, jak i w NEURO. U 55 z nich wszystkie rekordy KOR poprzedzają pierwszy rekord NEURO, z medianą odstępu 546 dni. Ponieważ grupujemy dane po pacjencie, jedna osoba nie może być jednocześnie przypadkiem pozytywnym i nieoznaczonym.

Do rozstrzygnięcia, czy nadajemy im status pozytywny z flagą pochodzenia rekordu, czy wyłączamy ich jako niejednoznacznych. Przy przyjętym założeniu o hospitalizacji ich wcześniejsze rekordy KOR nie są materiałem sprzed rozpoznania, tylko wynikami sprzed tej konkretnej hospitalizacji.

---

## Czego te pytania nie blokują

Równolegle domykamy to, co nie zależy od odpowiedzi: konfigurację podziału danych (punkt 0.6, wyniki w `4-pu-setup/results/`), regułę agregacji rekordów do pacjenta (punkt 0.3) oraz metryki pacjentowe i funkcję celu (punkty 4–5).
