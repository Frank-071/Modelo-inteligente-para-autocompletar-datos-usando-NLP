# src/main.py
# Demo: audio -> Whisper (STT) -> NER (SVM) -> JSON

import sys, os, re, time, json, warnings
from pathlib import Path
from datetime import datetime as dt
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import simple as feat_simple
from features import pos as feat_pos
from features import pos_emb as feat_pos_emb
warnings.filterwarnings("ignore", r"pkg_resources is deprecated", category=UserWarning)
from faster_whisper import WhisperModel

EXPERIMENTS = {
    "exp1_svm_simple": feat_simple.featurize,
    "exp2_svm_pos": feat_pos.featurize,
    "exp3_svm_pos_emb_pro": feat_pos_emb.featurize,  # usa svm_baseline por dentro
    "exp4_mlp_pos_emb_pro": feat_pos_emb.featurize,
}


# ---------- tokenización ----------
TOK_RE = re.compile(r"\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]", re.UNICODE)
def tokenize(text: str):
    return [m.group(0) for m in TOK_RE.finditer(text)]

def featurize_tokens(tokens, feature_fn):
    sents = [[(t, "O") for t in tokens]]
    X_flat, _ = feature_fn(sents)
    return X_flat

# ---------- util numéricas ----------
def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

# fechas comunes: 28/07/2004, 28-07-2004, "28 de julio del 2004"
def _parse_fecha(s: str) -> bool:
    s = (s or "").strip().lower()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt.strptime(s, fmt); return True
        except Exception:
            pass
    m = re.match(r"^(\d{1,2})\s+de\s+([a-záéíóú]+)\s+del\s+(\d{4})$", s)
    if m:
        d, mes, y = int(m.group(1)), m.group(2), int(m.group(3))
        meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
                 "septiembre","octubre","noviembre","diciembre"]
        return mes in meses and 1 <= d <= 31 and 1900 <= y <= 2100
    return False

# ---------- diccionarios ligeros ----------
LIMA_DISTRICTS = {
    "San Martín de Porres","San Juan de Lurigancho","San Juan de Miraflores","San Borja","San Isidro",
    "San Luis","San Miguel","Los Olivos","Comas","Independencia","Rímac","Cercado de Lima","Breña","Pueblo Libre",
    "Jesús María","La Victoria","Magdalena del Mar","Miraflores","Barranco","Surco","Santiago de Surco","Surquillo",
    "Chorrillos","La Molina","Lince","Ate","Santa Anita","El Agustino","Villa El Salvador","Villa María del Triunfo",
    "Carabayllo","Puente Piedra","Ancón","Santa Rosa","Pachacámac","Pucusana","Punta Hermosa","Punta Negra",
    "San Bartolo","Santa María del Mar","Lurín","Lurigancho","Cieneguilla"
}
# normaliza espacios/acentos ligeros
def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.strip()).lower()

# ---------- patrones ----------
# DNI indicado por etiqueta cercana ("dni", "documento")
DNI_FROM_LABEL_RE = re.compile(
    r"(?:dni|documento(?:\s+nacional\s+de\s+identidad)?)\s*(?:n(?:º|°|o)?|num\.?|número|:|#)?\s*([0-9][\s-]?\d(?:[\s-]?\d){6,7})",
    re.IGNORECASE,
)

# Teléfono (8 o 9 dígitos) cuando aparece tras la palabra 'cel/celular/teléfono'
PHONE_FROM_LABEL_RE = re.compile(
    r"(?:cel(?:ular)?|tel(?:éfono)?)[:\s-]*((?:\+?51[\s-]?)?\d(?:[\s-]?\d){7,8})",
    re.IGNORECASE,
)

# fallback de DNI (8 dígitos) si no hay etiqueta y no parece teléfono
GENERIC_8DIG_RE = re.compile(r"\b\d(?:[\s-]?\d){7}\b")  # 8 dígitos con separadores

# Dirección: Av./Jr./Calle … 123 [dpto/piso], o Manzana/Mz … Lote/Lt …
DIRECCION_RE = re.compile(
    r"(?:"
    r"(?:Av\.|Avenida|Jr\.|Jirón|Calle|Psje\.|Pasaje)\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]+?\s+\d{1,5}(?:\s+(?:dpto\.?|piso)\s*\w+)?"
    r"|"
    r"(?:Manzana|Mz\.?|Mz)\s*[A-Za-z]\s*(?:Lote|Lt\.?|Lt)\s*\d+"
    r")",
    re.IGNORECASE,
)

def _extract_name_block(text: str):
    """
    Si hay 'mi nombre es|me llamo|soy ... ,' usa ese bloque y separa apellidos como últimos 1-2 tokens.
    """
    m = re.search(r"(?:mi\s+nombre\s+es|me\s+llamo|soy)\s+(.+?)(?:,|\.|$)", text, re.IGNORECASE)
    if not m:
        return None, None
    block = _norm(m.group(1))
    toks = block.split()
    if len(toks) >= 3:
        # heurística simple: últimos dos como apellidos
        apes = " ".join(toks[-2:])
        noms = " ".join(toks[:-2])
    elif len(toks) == 2:
        noms, apes = toks[0], toks[1]
    else:
        noms, apes = toks[0], None
    # devuelve con capitalización “título” light
    def cap(s): 
        if not s: return s
        keep = {"de","del","la","las","los","y","san","santa"}
        return " ".join(w.capitalize() if w.lower() not in keep else w.lower() for w in s.split())
    return cap(noms), cap(apes) if apes else None

def _guess_distrito_departamento(text: str, ent: dict) -> dict:
    low = text.lower()

    # Si ya hay DEPARTAMENTO vacío y aparece "lima"
    if "lima" in low and not ent.get("DEPARTAMENTO"):
        ent["DEPARTAMENTO"] = "Lima"

    # Buscar un distrito válido en el texto
    found_valid = None
    for d in LIMA_DISTRICTS:
        if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
            found_valid = d
            break

    # Si no hay DISTRITO o el que hay no es válido, usa el encontrado
    if found_valid:
        if ent.get("DISTRITO") not in LIMA_DISTRICTS:
            ent["DISTRITO"] = found_valid
        if not ent.get("DEPARTAMENTO"):
            ent["DEPARTAMENTO"] = "Lima"

    return ent


def bio_to_spans(tokens, labels):
    spans = []
    i = 0
    while i < len(tokens):
        tag = labels[i]
        if tag.startswith("B-"):
            typ = tag.split("-", 1)[1]
            j = i + 1
            while j < len(tokens) and labels[j] == f"I-{typ}":
                j += 1
            spans.append({"type": typ, "start": i, "end": j, "text": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return spans

def fix_entities_with_rules(text: str, ent_map: dict) -> dict:
    """
    Reglas de respaldo y normalización:
    - Prioriza DNI tras la palabra 'DNI'; si no hay, evita confundirlo con teléfono.
    - Teléfono desde label (8 o 9 dígitos).
    - Acepta DNI de 7–8 dígitos (tu ejemplo muestra 7).
    - Extrae Dirección si aparece un patrón de vía o 'Manzana/Mz ... Lote/Lt ...'.
    - Reconoce distritos de Lima (p.ej., San Martín de Porres) y fija DEPARTAMENTO.
    - Refina nombre si hay 'mi nombre es|me llamo|soy ...'.
    """
    ent = dict(ent_map)

    # 0) Nombre por bloque, si falta o es corto
    n_nom, n_ape = _extract_name_block(text)
    if n_nom:
        if not ent.get("NOMBRES") or len(ent["NOMBRES"].split()) < len(n_nom.split()):
            ent["NOMBRES"] = n_nom
        if n_ape and (not ent.get("APELLIDOS") or len(ent["APELLIDOS"].split()) < len(n_ape.split())):
            ent["APELLIDOS"] = n_ape

    # 1) Teléfono por label
    if not ent.get("TELEFONO"):
        m = PHONE_FROM_LABEL_RE.search(text)
        if m:
            tel = _digits(m.group(1))
            if tel.startswith("51") and len(tel) > 9:
                tel = tel[-9:]
            # admite 8 o 9 dígitos según entrada
            if 8 <= len(tel) <= 9:
                ent["TELEFONO"] = tel

    # 2) DNI explícito por label
    dni_val = None
    m_dni = DNI_FROM_LABEL_RE.search(text)
    if m_dni:
        d = _digits(m_dni.group(1))
        if 7 <= len(d) <= 8:
            dni_val = d

    # 3) Si el modelo ya dio DNI/NUM_DOC, conserva el más razonable (prefiere 7–8 dígitos)
    model_dni = ent.get("NUM_DOC") or ent.get("DNI")
    if model_dni:
        md = _digits(model_dni)
        if 7 <= len(md) <= 8:
            dni_val = md

    # 4) Fallback: busca un 8-dígitos aislado que no parezca teléfono
    if not dni_val:
        for m in GENERIC_8DIG_RE.finditer(text):
            cand = _digits(m.group(0))
            # si cerca de "cel"/"tel", sáltalo
            s = max(0, m.start() - 12); e = min(len(text), m.end() + 12)
            ctx = text[s:e].lower()
            if re.search(r"cel|tel", ctx):
                continue
            dni_val = cand
            break

    if dni_val:
        ent["NUM_DOC"] = dni_val
        ent.pop("DNI", None)  # unifica

    # 5) Si FECHA_NAC es 8 dígitos y no parsea -> probablemente era DNI
    fn = ent.get("FECHA_NAC")
    if fn:
        d8 = _digits(fn)
        if len(d8) in (7, 8) and not _parse_fecha(fn):
            ent.setdefault("NUM_DOC", d8)

    # 6) Dirección si falta
    if not ent.get("DIRECCION"):
        mdir = DIRECCION_RE.search(text)
        if mdir:
            ent["DIRECCION"] = mdir.group(0).strip()

    # 7) Ubicación (distrito/departamento)
    ent = _guess_distrito_departamento(text, ent)

    # 8) Normalizaciones finales
    if ent.get("NUM_DOC"):
        ent["NUM_DOC"] = _digits(ent["NUM_DOC"])
    if ent.get("TELEFONO"):
        tel = _digits(ent["TELEFONO"])
        if tel.startswith("51") and len(tel) > 9:
            tel = tel[-9:]
        ent["TELEFONO"] = tel

    return ent

# ---------- audio & STT ----------
def record_until_silence(
    samplerate=16000, channels=1, blocksize=1024,
    silence_thr=0.015, silence_ms=1200, max_seconds=30, warmup_ms=200
) -> np.ndarray:
    print("🎤 Di algo… (se detiene al detectar silencio)")
    frames, started = [], False
    silence_run_ms, start_time, warm_ms_left = 0, time.time(), warmup_ms

    def callback(indata, frames_count, time_info, status):
        nonlocal started, silence_run_ms, warm_ms_left
        mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-9)
        voiced = rms >= silence_thr
        if warm_ms_left > 0:
            warm_ms_left -= int(1000 * (frames_count / samplerate))
            voiced = False
        if not started:
            if voiced:
                started = True
                frames.append(mono)
        else:
            frames.append(mono)
            silence_run_ms = 0 if voiced else (silence_run_ms + int(1000 * (frames_count / samplerate)))

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32",
                        blocksize=blocksize, callback=callback):
        while True:
            sd.sleep(50)
            if started and silence_run_ms >= silence_ms:
                break
            if (time.time() - start_time) > max_seconds:
                print("⏱️ Tiempo máximo alcanzado, deteniendo…")
                break

    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(frames).astype("float32")

def save_wav(path: Path, audio: np.ndarray, sr=16000):
    if audio.size > 0:
        sf.write(str(path), audio, sr, subtype="PCM_16")

def transcribe(audio_path: Path, whisper_size="small", device="cpu"):
    model = WhisperModel(whisper_size, device=device, compute_type="int8")
    segments, info = model.transcribe(str(audio_path), language="es")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return re.sub(r"\s+", " ", text)

# ---------- CLI principal ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta del .pkl entrenado (Pipeline)")
    ap.add_argument("--whisper_size", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu o cuda")
    ap.add_argument("--silence_ms", type=int, default=1200)
    ap.add_argument("--silence_thr", type=float, default=0.015)
    ap.add_argument("--max_seconds", type=int, default=30)
    ap.add_argument("--out_wav", default="audio/capture.wav")
    ap.add_argument(
        "--exp",
        required=True,
        choices=["exp1_svm_simple", "exp2_svm_pos", "exp3_svm_pos_emb_pro", "exp4_mlp_pos_emb_pro"],
        help="Experimento usado para entrenar el modelo (define el featurizer).",
    )
    ap.add_argument(
        "--text",
        default=None,
        help="Texto manual para probar el modelo (salta la parte de audio/Whisper).",
    )
    args = ap.parse_args()
    # elegir el featurizer según el experimento
    feature_fn = EXPERIMENTS[args.exp]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] No existe el modelo: {model_path}")
        sys.exit(1)

    if args.text:
        text = args.text
        print("=== Texto manual ===")
        print(text)

    else:
        out_wav = Path(args.out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        audio = record_until_silence(
            silence_ms=args.silence_ms, silence_thr=args.silence_thr, max_seconds=args.max_seconds
        )
        if audio.size == 0:
            print("[WARN] No se capturó audio.")
            sys.exit(0)
        save_wav(out_wav, audio, 16000)
        print(f"[OK] Audio guardado en: {out_wav}")

        print("[STT] Transcribiendo…")
        text = transcribe(out_wav, whisper_size=args.whisper_size, device=args.device)
        print("=== Transcripción ===")
        print(text if text else "(vacío)")

    pipe = joblib.load(model_path)
    tokens = tokenize(text)
    X = featurize_tokens(tokens, feature_fn)
    y_pred = pipe.predict(X)

    spans = bio_to_spans(tokens, y_pred)
    ent_map = {}
    for s in spans:
        prev = ent_map.get(s["type"])
        if not prev or len(s["text"]) > len(prev):
            ent_map[s["type"]] = s["text"]

    ent_map = fix_entities_with_rules(text, ent_map)

    # Salida (sin TIPO_DOC por simplicidad)
    order = ["NOMBRES","APELLIDOS","NUM_DOC","FECHA_NAC","TELEFONO","DEPARTAMENTO","DISTRITO","DIRECCION"]
    print("\n=== Entidades extraídas ===")
    for k in order:
        print(f"{k.title().replace('_',' ')}: {ent_map.get(k, '—')}")

    print("\nJSON:")
    print(json.dumps(ent_map, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()


