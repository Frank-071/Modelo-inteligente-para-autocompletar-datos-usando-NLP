# src/generate_synth.py
import argparse, random, re, os, json, datetime
from pathlib import Path

# --- Tokenizador (para alinear BIO con offsets) ---
try:
    import spacy
    nlp = spacy.blank("es")
except Exception:
    raise SystemExit(
        "Instala spaCy (requirements.txt) para tokenizar: pip install spacy"
    )

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "data_generation_lists"

# ---------- util lectura ----------
def load_lines(p: Path):
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_default_departamentos(lines):
    # si ya hay un departamentos.txt úsalo; si no, mete los 24 dptos
    if lines:
        return lines
    return [
        "Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Cusco","Huancavelica",
        "Huánuco","Ica","Junín","La Libertad","Lambayeque","Lima","Loreto","Madre de Dios",
        "Moquegua","Pasco","Piura","Puno","San Martín","Tacna","Tumbes","Ucayali"
    ]

# ---------- generadores de campos ----------
MESES = [
    "enero","febrero","marzo","abril","mayo","junio",
    "julio","agosto","septiembre","octubre","noviembre","diciembre"
]

def gen_fecha():
    # rango humano razonable
    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    f1 = f"{day:02d}/{month:02d}/{year}"
    f2 = f"{day:02d}-{month:02d}-{year}"
    f3 = f"{day} de {MESES[month-1]} del {year}"
    # OJO: acá puse "del" en vez de "de" para parecer más real (tú hablas así)
    # eso ayuda a que el modelo reconozca tu manera de hablar.
    return random.choice([f1, f2, f3])

def gen_telefono():
    # celulares PE típicos (9 dígitos que empiezan en 9) con variantes
    core = "9" + "".join(random.choice("0123456789") for _ in range(8))
    variants = [
        core,
        f"+51 {core}",
        f"+51-{core}",
        f"+51 {core[:3]} {core[3:6]} {core[6:]}",
        f"{core[:3]}-{core[3:6]}-{core[6:]}"
    ]
    return random.choice(variants)

def _gen_direccion_calle():
    # estilo "Av. Los Olivos 1234 dpto. 203"
    vias = ["Av.", "Avenida", "Jr.", "Jirón", "Calle", "Psje.", "Pasaje"]
    nombres = [
        "Los Olivos","San Martín","Primavera","El Sol","Los Pinos","España",
        "Bolívar","Arequipa","Progreso","Libertad","Petroperú",
        "Los Laureles","Industrial","Los Nogales","Los Cedros","Los Sauces"
    ]
    num = random.randint(10, 9999)

    extra = random.choice([
        "",
        f" dpto. {random.randint(101, 120)}",
        f" piso {random.randint(2, 15)}",
        f" Mz {random.randint(1, 25)} Lt {random.randint(1, 30)}"
    ])
    return f"{random.choice(vias)} {random.choice(nombres)} {num}{extra}".strip()

def _gen_direccion_manzana_lote():
    # estilo condominio/asentamiento que tú dices al hablar
    # ejemplos que queremos que el modelo aprenda como DIRECCION, NO DISTRITO:
    # "Manzana G Lote 19"
    # "Mz G Lt 19"
    # "Manzana J Lote 24 Los Nogales"
    manzana_label = random.choice(["Manzana", "Mz", "Mz."])
    letra = random.choice(list("ABCDEFGHJKLMNPRSTUVWX"))  # sin O para no confundir con 0
    lote_label = random.choice(["Lote", "Lt", "Lt."])
    lote_num = random.randint(1, 40)

    base = f"{manzana_label} {letra} {lote_label} {lote_num}"

    # a veces agregamos algo tipo "de Los Nogales" / "Los Nogales"
    sufijo_opcional = random.choice([
        "",
        f" de Los Nogales",
        f" Los Nogales",
        f" Urbanización Los Nogales",
        f" Asoc. Los Olivos",
    ])

    return (base + sufijo_opcional).strip()

def gen_direccion():
    """
    Mezcla dos estilos:
    - dirección urbana clásica (Av., Jr., etc.)
    - dirección tipo manzana/lote (Manzana G Lote 19 ...)
    Eso ayuda a que el modelo no confunda 'Manzana G Lote 19' con DISTRITO.
    """
    if random.random() < 0.5:
        return _gen_direccion_calle()
    else:
        return _gen_direccion_manzana_lote()

def gen_dni():
    """
    Genera un DNI en 8 dígitos con variantes de formato:
    12345678
    1234-5678
    12-34-56-78
    etc.
    Esto es importante porque tú hablas el DNI con pausas/guiones.
    """
    base = "".join(random.choice("0123456789") for _ in range(8))
    r = random.random()
    if r < 0.50:  # 50% compacto
        return base                      # 12345678
    if r < 0.65:
        return f"{base[:4]}-{base[4:]}"   # 1234-5678
    if r < 0.80:
        return f"{base[:4]} {base[4:]}"   # 1234 5678
    if r < 0.90:
        return f"{base[:2]}-{base[2:4]}-{base[4:6]}-{base[6:]}"  # 12-34-56-78
    return f"{base[:2]} {base[2:5]} {base[5:]}"                  # 12 345 678

# ---------- render + BIO ----------
PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")

def render_with_spans(template: str, values: dict):
    """
    Devuelve:
    - text final con todos los placeholders reemplazados
    - spans[label] = [(start,end), ...] ubicaciones de cada campo
    Esto permite etiquetar BIO sin tener que buscar cadenas luego.
    """
    out_chunks = []
    spans = {k: [] for k in values.keys()}
    i = 0
    pos = 0

    while i < len(template):
        m = PLACEHOLDER_RE.search(template, i)
        if not m:
            lit = template[i:]
            out_chunks.append(lit)
            pos += len(lit)
            break

        # literal antes del placeholder
        lit = template[i:m.start()]
        out_chunks.append(lit)
        pos += len(lit)

        key = m.group(1)
        val_str = str(values.get(key, f"{{{key}}}"))
        start = pos
        out_chunks.append(val_str)
        pos += len(val_str)

        spans.setdefault(key, []).append((start, pos))
        i = m.end()

    text = "".join(out_chunks)
    return text, spans

def to_bio(text, spans_by_label):
    """
    Toma el texto final y los spans exactos,
    tokeniza con spaCy y asigna B-<LABEL>/I-<LABEL>/O token por token.
    """
    doc = nlp(text)
    labels = ["O"] * len(doc)

    for label, lst in spans_by_label.items():
        for (s, e) in lst:
            started = False
            for i, tok in enumerate(doc):
                tok_start = tok.idx
                tok_end   = tok.idx + len(tok.text)
                if tok_start >= s and tok_end <= e:
                    labels[i] = ("B-" + label) if not started else ("I-" + label)
                    started = True

    return doc, labels

# ---------- pipeline ----------
def synth_one(nombres, apellidos, distritos, departamentos, plantilla):
    # Elegimos un departamento y un distrito aleatoriamente,
    # luego generamos una dirección que puede ser "Av ..." o "Manzana G Lote 19"
    dep = random.choice(departamentos)
    dist = random.choice(distritos)
    direccion = gen_direccion()

    valores = {
        "NOMBRES":      random.choice(nombres),
        "APELLIDOS":    random.choice(apellidos),
        "DNI":          gen_dni(),
        "FECHA_NAC":    gen_fecha(),
        "TELEFONO":     gen_telefono(),
        "DEPARTAMENTO": dep,
        "DISTRITO":     dist,
        "DIRECCION":    direccion
    }

    text, spans = render_with_spans(plantilla, valores)
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
    ap.add_argument("--n", type=int, required=True,
                    help="cuántas oraciones sintéticas generar en total")
    ap.add_argument("--out-dir", type=str, default="data",
                    help="carpeta destino (train/val/test.conll)")
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
        raise SystemExit(
            "Faltan listas (nombres/apellidos/distritos/plantillas). "
            "Revisa data_generation_lists/"
        )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generar ejemplos sintéticos
    examples = []
    for _ in range(args.n):
        tpl = random.choice(plantillas)
        examples.append(synth_one(nombres, apellidos, distritos, departamentos, tpl))

    # Shuffle + splits
    random.shuffle(examples)
    n_total = len(examples)
    n_train = int(n_total * args.split_train)
    n_val   = int(n_total * args.split_val)

    train = examples[:n_train]
    val   = examples[n_train:n_train + n_val]
    test  = examples[n_train + n_val:]

    # Guardar
    write_conll(out_dir / "train.conll", train)
    write_conll(out_dir / "val.conll",   val)
    write_conll(out_dir / "test.conll",  test)

    meta = {
        "n_total": n_total,
        "n_train": len(train),
        "n_val":   len(val),
        "n_test":  len(test),
        "seed":    args.seed,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with (out_dir / "synth_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Generado en {out_dir} -> "
        f"train({len(train)}), val({len(val)}), test({len(test)})"
    )

if __name__ == "__main__":
    main()

