import csv, re, os
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = 'datasets/cleaned/neuro_cleaning.csv'
rows = []
with open(path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        rows.append(row)

# Duplikaty - wyczyszcone
ids = Counter(r['custom_id'] for r in rows)
dups = [(k,v) for k,v in ids.items() if v > 1]
print(f'Duplikatow custom_id: {len(dups)}')
if dups:
    print('Przyklady:', dups[:3])
print('Poprawione')

# Przykladowy format PLT vs Na - poprawione chodzi o values
print()
print('=== FORMAT: PLT (dict) vs Na (plain liczba) ===')
for r in rows[:2]:
    plt_val = r['PLT']
    na_val = r['Na']
    print(f'  PLT={plt_val[:50]}  |  Na={str(na_val)[:30]}')
print('Poprawione')

# KREA - podejrzane wartosci > 20 (moze umol/l zamiast mg/dl)
print()
print('=== KREA > 20 mg/dl (podejrzane - moze umol/l?) ===')
count = 0
for row in rows:
    val_str = row.get('KREA', '').strip()
    unit = row.get('KREA-unit', '').strip()
    if not val_str:
        continue
    nums = re.findall(r'[-+]?\d*\.?\d+', val_str)
    if not nums:
        continue
    v = float(nums[0])
    if v > 20:
        print(f'  pacjent {row["patient_id"]}, {row["examination_date"]}: KREA={v} [{unit}]')
        count += 1
        if count >= 15:
            print('  ...')
            break
print('Drugi skrypt do sprawdzenia KREA: check_krea.py')

# Zakresy norm flag - ile rekordow ma flage -1/0/1 vs puste
print()
print('=== FLAGI NORM (ile -1 / 0 / 1 / puste) dla kluczowych parametrow ===')
for col in ['HGB', 'WBC', 'PLT', 'KREA', 'Na', 'K', 'CRP']:
    norm_col = col + '-norm'
    vals = Counter(r.get(norm_col, '').strip() for r in rows)
    print(f'  {col:6s}: -1={vals.get("-1.0",0)+vals.get("-1",0):4d}  0={vals.get("0.0",0)+vals.get("0",0):4d}  1={vals.get("1.0",0)+vals.get("1",0):4d}  puste={vals.get("",0):4d}')

# Zakres dat
print()
dates = sorted(r['examination_date'] for r in rows if r['examination_date'])
print(f'=== ZAKRES DAT: {dates[0]} ... {dates[-1]} ===')

# Ile pacjentow z bardzo krotkim vs dlugim pobytem
pc = Counter(r['patient_id'] for r in rows)
print()
print('=== LICZBA REKORDOW NA PACJENTA ===')
long_stay = [(pid, cnt) for pid, cnt in pc.items() if cnt >= 20]
long_stay.sort(key=lambda x: -x[1])
print(f'  Pacjentow z >= 20 rekordami (>= 20 tyg.): {len(long_stay)}')
print(f'  TOP 5 najdluzszych:')
for pid, cnt in long_stay[:5]:
    print(f'    pacjent {pid}: {cnt} tygodni')
