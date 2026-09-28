import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
TOTAL = 698680199
NPARTS = 16
OUT = "NormanWeissman2019_filtered.h5ad"
CH = (TOTAL + NPARTS - 1) // NPARTS

def get_size(part):
    fn = f"{OUT}.part{part:02d}"
    return os.path.getsize(fn) if os.path.exists(fn) else 0

def want_size(part):
    return min(CH, TOTAL - part * CH)

def dl(part):
    lo = part * CH + get_size(part)
    hi = part * CH + want_size(part) - 1
    if get_size(part) >= want_size(part):
        return part, get_size(part)
    fn = f"{OUT}.part{part:02d}.new"
    r = subprocess.run(['curl', '-sL', '-r', f'{lo}-{hi}', '-o', fn, URL],
                       capture_output=True, timeout=280)
    if os.path.exists(fn):
        with open(fn, 'rb') as fnew, open(f"{OUT}.part{part:02d}", 'ab') as fold:
            fold.write(fnew.read())
        os.remove(fn)
    return part, get_size(part)

parts = list(range(NPARTS))
todo = [p for p in parts if get_size(p) < want_size(p)]
print(f'todo parts: {len(todo)}', flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    for part, sz in ex.map(dl, todo):
        print(f'part {part} -> {sz}/{want_size(part)}', flush=True)
done = all(get_size(p) >= want_size(p) for p in parts)
print('ALL_PARTS_DONE' if done else 'INCOMPLETE', flush=True)
if done:
    with open(OUT, 'wb') as out:
        for p in parts:
            with open(f"{OUT}.part{p:02d}", 'rb') as f:
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    out.write(b)
    print('MERGED', os.path.getsize(OUT))
    for p in parts:
        os.remove(f"{OUT}.part{p:02d}")
