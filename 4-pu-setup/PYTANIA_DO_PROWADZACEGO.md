# Pytania blokujące etap modelowania

**Projekt:** Ocena ryzyka wystąpienia tętniaka mózgu z wykorzystaniem modelowania Positive-Unlabeled (ID-1650)
**Adresat:** dr inż. Patryk Jasik, ewentualnie właściciel danych klinicznych
**Podstawa:** diagnostyka z `4-pu-setup/ETAP0_USTALENIA.md`

Trzy pytania poniżej blokują zamrożenie protokołu. Każde z nich zmienia kształt modelowania, więc nie chcemy rozstrzygać ich samodzielnie ani odkładać na etap, w którym modele już powstaną.

---

## 1. Czym są rekordy NEURO względem momentu rozpoznania

**Co widzimy w danych.** Kohorta KOR mieści się praktycznie w całości w latach 2020–2021 (99,5% rekordów), podczas gdy NEURO rozciąga się na lata 2000–2024. W oknie dat KOR mieści się tylko 680 z 6 651 rekordów NEURO, czyli 271 z 1 823 pacjentów. Porównanie median cech między NEURO z tego okna a NEURO spoza niego daje różnice tego samego rzędu co różnice między kohortami, a dla sodu, potasu i RDW większe.

**Dlaczego to blokuje.** Model może rozpoznawać nie tętniaka, lecz okres, laboratorium albo tryb pozyskania danych. Nie mamy też daty rozpoznania, więc nie wiemy, czy wyniki NEURO pochodzą sprzed diagnozy, z okresu diagnostyki, czy już po leczeniu. W tym ostatnim przypadku model uczyłby się skutków hospitalizacji, a nie ryzyka.

**O co prosimy.** Potwierdzenie, w jakim momencie ścieżki pacjenta powstają wyniki NEURO, oraz informację, czy da się uzyskać datę rozpoznania lub choćby datę przyjęcia na oddział. Jeśli nie, prosimy o opinię, czy akceptowalne jest ograniczenie analizy głównej do wspólnego okna dat, co zostawia 271 pacjentów pozytywnych zamiast 1 823.

## 2. Status 63 pacjentów występujących w obu kohortach

**Co widzimy w danych.** 63 pacjentów ma rekordy zarówno w KOR, jak i w NEURO. U 55 z nich wszystkie rekordy KOR poprzedzają pierwszy rekord NEURO, z medianą odstępu 546 dni.

**Dlaczego to blokuje.** Grupujemy dane po pacjencie, więc jedna osoba nie może być jednocześnie przypadkiem pozytywnym i nieoznaczonym. Kusi nas, żeby potraktować ich wcześniejsze wyniki jako materiał sprzed rozpoznania, ale bez daty diagnozy jest to tylko kolejność źródeł, a nie dowód.

**O co prosimy.** Rozstrzygnięcie, czy obecność w NEURO oznacza potwierdzone rozpoznanie tętniaka, czy samo skierowanie na oddział. Od tego zależy, czy nadajemy tym pacjentom status pozytywny, czy wyłączamy ich jako niejednoznacznych.

## 3. Jednostki kreatyniny

**Co widzimy w danych.** Mediana KREA wynosi 0,87, co wskazuje na mg/dl, ale 767 wartości (1,19%) przekracza 15, a 210 wartości (0,33%) przekracza 50 — to zakres typowy dla µmol/l. Problem występuje w obu kohortach w podobnej proporcji.

**Dlaczego to blokuje.** Automatyczne przeliczenie wszystkiego powyżej wybranego progu skasowałoby prawdziwe przypadki ciężkiej niewydolności nerek, a pozostawienie danych bez zmian wprowadza szum do istotnej klinicznie cechy.

**O co prosimy.** Informację, czy w systemie źródłowym zapisana jest jednostka konkretnego oznaczenia, albo wskazanie laboratorium i okresu, dla których obowiązywała inna jednostka.

---

## Czego nie blokują te pytania

Równolegle domykamy to, co nie zależy od odpowiedzi: konfigurację podziału danych (punkt 0.6, wyniki w `4-pu-setup/results/`), regułę agregacji rekordów do pacjenta oraz przygotowanie metryk pacjentowych i funkcji celu.
