# src/generate_synth.py
import argparse, random, re, json, csv
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "data_generation_lists"
OUTDIR_DEFAULT = ROOT / "data"

MESES = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]

def read_lines(p): 
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def load_lists():
    nombres   = read_lines(LISTS/"nombres.txt")
    apellidos = read_lines(LISTS/"apellidos.txt")
    distritos = read_lines(LISTS/"distritos.txt")
    plantillas= read_lines(LISTS/"plantillas.txt")
    return nombres, apellidos, distritos, plantillas

def rand_dni():        return f"{random.randint(10_000_000, 99_999_999)}"
def rand_telefono():
    base = f"9{random.randint(10_000_000, 99_999_999)}"
    r = random.random()
    if r < 0.2:  return f"+51 {base}"
    if r < 0.35: return f"{base[:3]} {base[3:6]} {base[6:]}"
    if r < 0.5:  return f"+51-{base}"
    return base
def rand_fecha_nac():
    start, end = date(1965,1,1), date(2007,12,31)
    d = start + timedelta(days=random.randint(0,(end-start).days))
    r = random.random()
    if r < 0.5:  return d.strftime("%d/%m/%Y")
    if r < 0.75: return d.strftime("%d-%m-%Y")
    return f"{d.day} de {MESES[d.month-1]} del {d.year}"
def rand_direccion():
    vias = ["Av.","Jr.","Calle","Psje.","Urb."]
    nombres = ["Arequipa","San Martín","Los Álamos","Progreso","Brasil","La Unión","Primavera",
               "Grau","Bolívar","La Paz","Bolognesi","Libertad","Los Olivos"]
    suf = ["", f"Depto {random.randint(101,504)}", f"Interior {random.randint(1,20)}",
           f"Mz {random.choice('ABCDEFG')}", f"Lt {random.randint(1,30)}", ""]
    return f"{random.choice(vias)} {random.choice(nombres)} {random.randint(100,9999)}" + (f" {random.choice(suf)}" if random.random()<0.5 else "")
def rand_apellidos(apells):
    a1 = random.choice(apells)
    if random.random() < 0.85:
        a2 = random.choice(apells)
        return f"{a1}-{a2}" if random.random()<0.10 else f"{a1} {a2}"
    return a1
def rand_nombres(noms):
    n = random.choice(noms)
    if " " in n or random.random()>0.75: return n
    return f"{n} {random.choice(noms)}" if random.random()<0.25 else n

_tok_re = re.compile(r"\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]", re.UNICODE)
def tokenize_with_spans(text):
    toks, spans = [], []
    for m in _tok_re.finditer(text):
        toks.append(m.group(0)); spans.append((m.start(), m.end()))
    return toks, spans
def find_span(text, value):
    i = text.find(value);  return None if i==-1 else (i, i+len(value))
def to_bio(text, values):
    tokens, spans = tokenize_with_spans(text)
    labels = ["O"] * len(tokens)

    for k, v in values.items():
        # Ubica el span de la entidad en el texto ya formateado
        i = text.find(v)
        if i == -1:
            continue
        a, b = i, i + len(v)

        started = False
        for ti, (s, e) in enumerate(spans):
            if e <= a or s >= b:
                continue
            # el primer token de este span -> B-; los siguientes -> I-
            labels[ti] = ("B-" + k) if not started else ("I-" + k)
            started = True

    return tokens, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--split", type=str, default="0.8,0.1,0.1")
    ap.add_argument("--out-dir", type=str, default=str(OUTDIR_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    nombres, apellidos, distritos, plantillas = load_lists()
    records = []
    for _ in range(args.n):
        vals = {
            "NOMBRES":   rand_nombres(nombres),
            "APELLIDOS": rand_apellidos(apellidos),
            "TIPO_DOC":  "DNI",
            "DNI":       rand_dni(),
            "FECHA_NAC": rand_fecha_nac(),
            "TELEFONO":  rand_telefono(),
            "DISTRITO":  random.choice(distritos),
            "DIRECCION": rand_direccion(),
        }
        text = random.choice(plantillas).format(**vals)
        toks, labs = to_bio(text, vals)
        records.append((toks,labs,text,vals))

    p_tr, p_va, p_te = [float(x) for x in args.split.split(",")]
    n = len(records); n_tr = int(n*p_tr); n_va = int(n*p_va)
    splits = [("train.conll", records[:n_tr]),
              ("val.conll",   records[n_tr:n_tr+n_va]),
              ("test.conll",  records[n_tr+n_va:])]

    for fname,items in splits:
        with open(outdir/fname, "w", encoding="utf-8") as f:
            for toks,labs,_,_ in items:
                for t,y in zip(toks,labs): f.write(f"{t}\t{y}\n")
                f.write("\n")

    # muestrita para debug
    with open(outdir/"sample_train.csv","w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["text","values_json"])
        for toks,labs,text,vals in records[:200]:
            w.writerow([text, json.dumps(vals, ensure_ascii=False)])

    print("[OK] train/val/test creados en", outdir)

if __name__ == "__main__":
    main()
