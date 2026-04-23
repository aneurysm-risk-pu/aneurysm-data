import csv, os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
path = 'datasets/cleaned/neuro_cleaning.csv'

# Checker 
# (parametr, min_ok,  max_ok,  opis_anomalii)
# Rekord jest anomalią jeśli v < min_ok LUB v > max_ok (None = brak progu)
CHECKS = [
    ('WBC',  0,    100,  'WBC > 100 G/l - bardzo wysoko'),
    ('PLT',  1,   None,  'PLT = 0 '),
    ('Na',  100,  None,  'Na < 100 mmol/l'),
    ('K',   None,   9,   'K > 9 mmol/l'),
]

rows = []
with open(path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f'Zaladowano {len(rows)} rekordow z: {path}\n')

for param, min_ok, max_ok, opis in CHECKS:
    hits = []
    for row in rows:
        val_str = row.get(param, '').strip()
        if not val_str:
            continue
        try:
            v = float(val_str)
        except ValueError:
            continue
        if (min_ok is not None and v < min_ok) or (max_ok is not None and v > max_ok):
            hits.append((row['patient_id'], row['examination_date'], v,
                         row.get(param + '-unit', '')))
    print(f'=== {param} — {opis} ===')
    if hits:
        for pid, dt, v, unit in hits:
            print(f'  pacjent={pid}  date={dt}  {param}={v}  [{unit}]')
    else:
        print('  Brak nieprawidlowych rekordow.')
    print()

