#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generador de dataset sintético (BIO) para llenado de formularios.
Campos: {NOMBRE}, {APELLIDO}, {DNI}, {TELEFONO}, {DIRECCION}, {FECHA_NAC}

Uso (desde la raíz del repo):
  python src/data_generator.py --n 3000 --splits 70,15,15 \
    --out_dir data --lists_dir data_generation_lists \
    --noise_pct 0 --seed 42 --also_csv
"""
import os, re, random, argparse, unicodedata, datetime, csv, json
from typing import List, Tuple, Dict

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

# ---------- util ----------
def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def gen_dni() -> str:
    return f"{random.randint(10_000_000, 99_999_999)}"

def gen_tel() -> str:
    return f"+51 9{random.randint(10_000_000, 99_999_999)}"

def gen_direccion(apellidos):
    """Devuelve algo como: 'Lima, distrito de Surco; Av. Quispe 1234'"""
    tipo_via = random.choice(["Av.", "Jr.", "Calle", "Psje."])
    via_nombre = random.choice(apellidos)       # usa tu lista de apellidos como nombre de calle
    numero = random.randint(100, 9999)

    # distritos comunes de Lima
    distritos = [
        "Lima", "Miraflores", "San Isidro", "San Borja", "Santiago de Surco", "La Molina",
        "Barranco", "Chorrillos", "Jesús María", "Lince", "Breña", "Magdalena del Mar",
        "Pueblo Libre", "San Miguel", "Rímac", "El Agustino", "Ate", "Santa Anita",
        "San Juan de Lurigancho", "San Juan de Miraflores", "Villa El Salvador",
        "Villa María del Triunfo", "Comas", "Independencia", "Los Olivos", "Carabayllo"
    ]
    distrito = random.choice(distritos)

    # No repitas el distrito dentro de la dirección; lo dejamos aparte para sonar como el ejemplo
    return f"Lima, distrito de {distrito}; {tipo_via} {via_nombre} {numero}"

def gen_fecha_nac():
    """Fecha de nacimiento en varios formatos realistas."""
    start = datetime.date(1970, 1, 1)
    end   = datetime.date(2007,12,31)
    d = start + datetime.timedelta(days=random.randint(0, (end-start).days))

    meses = ["enero","febrero","marzo","abril","mayo","junio",
             "julio","agosto","septiembre","octubre","noviembre","diciembre"]

    formatos = [
        f"{d.day:02d}/{d.month:02d}/{d.year}",               # 28/05/2004
        f"{d.day}/{d.month}/{d.year}",                       # 28/5/2004
        f"{d.day} de {meses[d.month-1]} de {d.year}",        # 28 de mayo de 2004
        f"{d.day} de {meses[d.month-1]} del {d.year}",       # 28 de mayo del 2004
        f"{d.year}-{d.month:02d}-{d.day:02d}",               # 2004-05-28
        f"{d.day:02d}-{d.month:02d}-{d.year}",               # 28-05-2004
    ]
    return random.choice(formatos)

def build_values(nombres: List[str], apellidos: List[str]) -> Dict[str, Tuple[str,str]]:
    nombre = random.choice(nombres)
    # ~60% con dos apellidos (paterno + materno) sin separarlos en archivos
    if random.random() < 0.60 and len(apellidos) >= 2:
        a1, a2 = random.sample(apellidos, 2)
        apellido = f"{a1} {a2}"
    else:
        apellido = random.choice(apellidos)
    return {
        "NOMBRE":    (nombre, "NOMBRE"),
        "APELLIDO":  (apellido, "APELLIDO"),
        "DNI":       (gen_dni(), "DNI"),
        "TELEFONO":  (gen_tel(), "TELEFONO"),
        "DIRECCION": (gen_direccion(apellidos), "DIRECCION"),
        "FECHA_NAC": (gen_fecha_nac(), "FECHA_NAC"),
    }

def apply_template(template: str, vals: Dict[str, Tuple[str,str]]):
    """Rellena {PLACEHOLDER} y devuelve texto + spans de entidades."""
    out, spans = [], []
    i, cur = 0, 0
    while i < len(template):
        if template[i] == "{":
            j = template.find("}", i+1)
            if j != -1:
                key = template[i+1:j]
                if key in vals:
                    val, etype = vals[key]
                    start = cur
                    out.append(val); cur += len(val)
                    spans.append({"type": etype, "start": start, "end": start+len(val)})
                    i = j + 1
                    continue
        out.append(template[i]); cur += 1; i += 1
    return "".join(out), spans

def tokenize_with_spans(text: str):
    toks, spans = [], []
    for m in TOKEN_PATTERN.finditer(text):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans

def bio_tags(tokens, token_spans, entity_spans):
    tags = ["O"] * len(tokens)
    for ent in entity_spans:
        etype, s, e = ent["type"], ent["start"], ent["end"]
        first = True
        for idx, (ts, te) in enumerate(token_spans):
            if te <= s or ts >= e:  # no solapa
                continue
            tags[idx] = f"B-{etype}" if first else f"I-{etype}"
            first = False
    return tags

def apply_noise(text: str, pct: float) -> str:
    """Ruido simple tipo STT (opcional): quitar tildes/puntuación."""
    if pct <= 0 or random.random() > pct/100.0:
        return text
    t = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    t = re.sub(r"[.,:;]", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t

def gen_examples(n, nombres, apellidos, plantillas, noise_pct=0.0):
    exs = []
    for _ in range(n):
        tpl = random.choice(plantillas)
        vals = build_values(nombres, apellidos)
        text, ents = apply_template(tpl, vals)
        noisy = apply_noise(text, noise_pct)
        toks, spans = tokenize_with_spans(noisy)
        tags = bio_tags(toks, spans, ents)
        exs.append({"text": noisy, "tokens": toks, "tags": tags})
    return exs

def split_dataset(exs, splits):
    a, b, c = splits
    n = len(exs)
    n_train = int(n * a/100); n_val = int(n * b/100)
    train = exs[:n_train]; val = exs[n_train:n_train+n_val]; test = exs[n_train+n_val:]
    return train, val, test

def write_conll(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            for tok, tag in zip(ex["tokens"], ex["tags"]):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")

def write_csv(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["text","tokens","tags"])
        for ex in examples:
            w.writerow([ex["text"], " ".join(ex["tokens"]), " ".join(ex["tags"])])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Total de ejemplos a generar")
    ap.add_argument("--splits", type=str, default="70,15,15", help="Porcentajes train,val,test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="data", help="Directorio de salida")
    ap.add_argument("--lists_dir", type=str, default="data_generation_lists", help="Carpeta con nombres/apellidos/plantillas")
    ap.add_argument("--noise_pct", type=float, default=0.0, help="% de ruido tipo STT (0,5,10)")
    ap.add_argument("--also_csv", action="store_true", help="Además de .conll, exportar .csv")
    args = ap.parse_args()

    random.seed(args.seed)
    splits = tuple(map(int, args.splits.split(",")))
    assert sum(splits) == 100, "splits deben sumar 100"

    nombres = read_list(os.path.join(args.lists_dir, "nombres.txt"))
    apells  = read_list(os.path.join(args.lists_dir, "apellidos.txt"))
    plants  = read_list(os.path.join(args.lists_dir, "plantillas.txt"))

    # validación simple de placeholders en plantillas
    valid_keys = {"NOMBRE","APELLIDO","DNI","TELEFONO","DIRECCION","FECHA_NAC"}
    for ln, p in enumerate(plants, 1):
        for key in re.findall(r"{(.*?)}", p):
            if key not in valid_keys:
                raise ValueError(f"Placeholder no soportado en plantillas (línea {ln}): {{{key}}}")

    exs = gen_examples(args.n, nombres, apells, plants, noise_pct=args.noise_pct)

    out = args.out_dir
    train, val, test = split_dataset(exs, splits)
    write_conll(train, os.path.join(out, "train.conll"))
    write_conll(val,   os.path.join(out, "val.conll"))
    write_conll(test,  os.path.join(out, "test.conll"))
    if args.also_csv:
        write_csv(train, os.path.join(out, "train.csv"))
        write_csv(val,   os.path.join(out, "val.csv"))
        write_csv(test,  os.path.join(out, "test.csv"))

    meta = {"n_total": len(exs),
            "splits": {"train": len(train), "val": len(val), "test": len(test)},
            "noise_pct": args.noise_pct, "seed": args.seed}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("OK:", json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
