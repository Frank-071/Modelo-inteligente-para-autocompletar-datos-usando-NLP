# src/generate_synth.py
import argparse, random, re, os, json, datetime
from pathlib import Path

# --- Tokenizador (solo para alinear BIO con offsets) ---
try:
    import spacy
    nlp = spacy.blank("es")
except Exception:
    raise SystemExit("Instala spaCy (requirements.txt) para tokenizar: pip install spacy")

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "data_generation_lists"

# ---------- util lectura ----------
def load_lines(p: Path):
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_default_departamentos(lines):
    if lines:
        return lines
    # 24 departamentos del Perú
    return [
        "Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Cusco","Huancavelica",
        "Huánuco","Ica","Junín","La Libertad","Lambayeque","Lima","Loreto","Madre de Dios",
        "Moquegua","Pasco","Piura","Puno","San Martín","Tacna","Tumbes","Ucayali"
    ]

# ---------- generadores de campos ----------
MESES = ["enero","febrero","marzo","abril","mayo","junio",
         "julio","agosto","septiembre","octubre","noviembre","diciembre"]

def gen_fecha():
    # Rango razonable de nacimiento
    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    f1 = f"{day:02d}/{month:02d}/{year}"
    f2 = f"{day:02d}-{month:02d}-{year}"
    f3 = f"{day} de {MESES[month-1]} de {year}"
    return random.choice([f1, f2, f3])

def gen_telefono():
    # celulares PE típicos con variantes
    core = "9" + "".join(random.choice("0123456789") for _ in range(8))
    variants = [
        core,
        f"+51 {core}",
        f"+51-{core}",
        f"+51 {core[:3]} {core[3:6]} {core[6:]}",
        f"{core[:3]}-{core[3:6]}-{core[6:]}"
    ]
    return random.choice(variants)

def gen_direccion():
    vias = ["Av.", "Avenida", "Jr.", "Jirón", "Calle", "Psje.", "Pasaje", "Mz", "Mz.", "Block"]
    nombres = ["Los Olivos","San Martín","Primavera","El Sol","Los Pinos","España","Bolívar","Arequipa",
               "Progreso","Libertad","Petroperú","Los Laureles","Industrial"]
    num = random.randint(10, 9999)
    extra = random.choice([
        "", f" dpto. {random.randint(101, 120)}", f" piso {random.randint(2, 15)}",
        f" Mz {random.randint(1, 25)} Lt {random.randint(1, 30)}"
    ])
    return f"{random.choice(vias)} {random.choice(nombres)} {num}{extra}".strip()

def gen_dni():
    """Genera un DNI en 8 dígitos con algunas variantes de formato."""
    d = "".join(random.choice("0123456789") for _ in range(8))
    r = random.random()
    if r < 0.50:  # 50% compacto
        return d                      # 12345678
    if r < 0.65:
        return f"{d[:4]}-{d[4:]}"     # 1234-5678
    if r < 0.80:
        return f"{d[:4]} {d[4:]}"     # 1234 5678
    if r < 0.90:
        return f"{d[:2]}-{d[2:4]}-{d[4:6]}-{d[6:]}"  # 12-34-56-78
    return f"{d[:2]} {d[2:5]} {d[5:]}"              # 12 345 678

# ---------- render + BIO ----------
PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")

def render_with_spans(template: str, values: dict):
    """
    Devuelve texto y spans exactos {label: [(start, end)]} para cada placeholder,
    resolviendo posiciones durante el render (sin búsquedas ambiguas).
    """
    out = []
    spans = {k: [] for k in values.keys()}
    i = 0
    pos = 0
    while i < len(template):
        m = PLACEHOLDER_RE.search(template, i)
        if not m:
            lit = template[i:]
            out.append(lit); pos += len(lit)
            break
        # literal previo
        lit = template[i:m.start()]
        out.append(lit); pos += len(lit)
        key = m.group(1)
        val_str = str(values.get(key, f"{{{key}}}"))
        start = pos
        out.append(val_str)
        pos += len(val_str)
        spans.setdefault(key, []).append((start, pos))
        i = m.end()
    text = "".join(out)
    return text, spans

def to_bio(text, spans_by_label):
    doc = nlp(text)
    labels = ["O"] * len(doc)
    # a cada span asigna B-/I- con el mismo nombre del placeholder
    for label, spans in spans_by_label.items():
        for (s, e) in spans:
            for i, tok in enumerate(doc):
                if tok.idx >= s and tok.idx + len(tok.text) <= e:
                    prefix = "B" if tok.idx == s else "I"
                    labels[i] = f"{prefix}-{label}"
    return doc, labels

# ---------- pipeline ----------
def synth_one(nombres, apellidos, distritos, departamentos, plantilla):
    v = {
        "NOMBRES":      random.choice(nombres),
        "APELLIDOS":    random.choice(apellidos),
        "DNI":          gen_dni(),
        "FECHA_NAC":    gen_fecha(),
        "TELEFONO":     gen_telefono(),
        "DEPARTAMENTO": random.choice(departamentos),
        "DISTRITO":     random.choice(distritos),
        "DIRECCION":    gen_direccion()
    }
    text, spans = render_with_spans(plantilla, v)
    doc, y = to_bio(text, spans)
    tokens = [t.text for t in doc]
    return tokens, y

def write_conll(path: Path, examples):
    with path.open("w", encoding="utf-8") as f:
        for toks, tags in examples:
            for t, y in zip(toks, tags):
                f.write(f"{t}\t{y}\n")
            f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-train", type=float, default=0.8)
    ap.add_argument("--split-val", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)

    nombres       = load_lines(LISTS / "nombres.txt")
    apellidos     = load_lines(LISTS / "apellidos.txt")
    distritos     = load_lines(LISTS / "distritos.txt")
    departamentos = ensure_default_departamentos(load_lines(LISTS / "departamentos.txt"))
    plantillas    = load_lines(LISTS / "plantillas.txt")

    if not (nombres and apellidos and distritos and plantillas):
        raise SystemExit("Faltan listas (nombres/apellidos/distritos/plantillas). Revisa data_generation_lists/")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for _ in range(args.n):
        tpl = random.choice(plantillas)
        examples.append(synth_one(nombres, apellidos, distritos, departamentos, tpl))

    random.shuffle(examples)
    n = len(examples)
    n_train = int(n * args.split_train)
    n_val = int(n * args.split_val)
    train = examples[:n_train]
    val   = examples[n_train:n_train + n_val]
    test  = examples[n_train + n_val:]

    write_conll(out_dir / "train.conll", train)
    write_conll(out_dir / "val.conll",   val)
    write_conll(out_dir / "test.conll",  test)

    meta = {
        "n_total": n, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "seed": args.seed, "timestamp": datetime.datetime.now().isoformat()
    }
    with (out_dir / "synth_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Generado: {out_dir}/train.conll ({len(train)}), val ({len(val)}), test ({len(test)})")

if __name__ == "__main__":
    main()

