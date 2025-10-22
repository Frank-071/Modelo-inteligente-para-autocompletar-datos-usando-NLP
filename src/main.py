# --- rutas para importar desde src/ner ---
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../TA-IA-pipeline
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# importa las features del baseline (ubicado en src/ner/svm_baseline.py)
from ner.svm_baseline import token_features

# imports normales del script
import argparse, os, time, json, re
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

TOK_RE = re.compile(r"\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]", re.UNICODE)

DNI_RE = re.compile(r"\b(?:\d{8}|\d{4}[-\s]\d{4}|\d{2}(?:[-\s]\d{2}){3})\b")

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _parse_fecha(s: str) -> bool:
    s = (s or "").strip().lower()
    # dd/mm/aaaa | dd-mm-aaaa
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            datetime.strptime(s, fmt); return True
        except Exception:
            pass
    # "28 de mayo del 2004" (simple)
    m = re.match(r"^(\d{1,2})\s+de\s+([a-záéíóú]+)\s+del\s+(\d{4})$", s)
    if m:
        d, mes, y = int(m.group(1)), m.group(2), int(m.group(3))
        meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
                 "septiembre","octubre","noviembre","diciembre"]
        return mes in meses and 1 <= d <= 31 and 1900 <= y <= 2100
    return False

def fix_entities_with_rules(text: str, ent_map: dict) -> dict:
    """
    - Si falta DNI, lo busca en el texto (admite 73-88-43-59, 73884359, 7388 4359).
    - Si 'FECHA_NAC' parece 8 dígitos y NO es fecha válida, lo trata como DNI.
    - Normaliza el DNI a 8 dígitos para mostrar bonito.
    """
    ent_map = dict(ent_map)  # copia

    # 1) si no hay DNI, intenta sacarlo del texto
    if not ent_map.get("DNI"):
        m = DNI_RE.search(text)
        if m:
            dig = _digits(m.group(0))
            if len(dig) == 8:
                ent_map["DNI"] = dig

    # 2) si FECHA_NAC parece DNI y no parsea como fecha -> úsalo como DNI
    fn = ent_map.get("FECHA_NAC")
    if fn:
        dig = _digits(fn)
        if len(dig) == 8 and not _parse_fecha(fn):
            ent_map.setdefault("DNI", dig)
            # opcional: borrar la "fecha" inválida
            # ent_map.pop("FECHA_NAC", None)

    # 3) normaliza el formato del DNI a 8 dígitos
    if ent_map.get("DNI"):
        d = _digits(ent_map["DNI"])
        if len(d) == 8:
            ent_map["DNI"] = d

    return ent_map

def tokenize(text: str):
    return [m.group(0) for m in TOK_RE.finditer(text)]

def featurize_tokens(tokens):
    sent = [(t, "O") for t in tokens]
    return [token_features(sent, i) for i in range(len(tokens))]

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

def record_until_silence(
    samplerate=16000,
    channels=1,
    blocksize=1024,
    silence_thr=0.015,
    silence_ms=1200,
    max_seconds=30,
    warmup_ms=200
) -> np.ndarray:
    print("🎤 Di algo… (se detiene al detectar silencio)")
    frames = []
    started = False
    silence_run_ms = 0
    start_time = time.time()
    warm_ms_left = warmup_ms

    def callback(indata, frames_count, time_info, status):
        nonlocal started, silence_run_ms, warm_ms_left
        if status:
            pass
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
            if voiced:
                silence_run_ms = 0
            else:
                silence_run_ms += int(1000 * (frames_count / samplerate))

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
    audio = np.concatenate(frames).astype("float32")
    return audio

def save_wav(path: Path, audio: np.ndarray, sr=16000):
    if audio.size == 0:
        return
    sf.write(str(path), audio, sr, subtype="PCM_16")

def transcribe(audio_path: Path, whisper_size="small", device="cpu"):
    model = WhisperModel(whisper_size, device=device, compute_type="int8")
    segments, info = model.transcribe(str(audio_path), language="es")
    text = "".join(seg.text for seg in segments).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta del .pkl entrenado (Pipeline)")
    ap.add_argument("--whisper_size", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu o cuda")
    ap.add_argument("--silence_ms", type=int, default=1200, help="ms de silencio para cortar")
    ap.add_argument("--silence_thr", type=float, default=0.015, help="umbral de RMS para voz")
    ap.add_argument("--max_seconds", type=int, default=30)
    ap.add_argument("--out_wav", default="audio/capture.wav")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] No existe el modelo: {model_path}")
        sys.exit(1)

    out_wav = Path(args.out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    audio = record_until_silence(silence_ms=args.silence_ms, silence_thr=args.silence_thr, max_seconds=args.max_seconds)
    if audio.size == 0:
        print("[WARN] No se capturó audio. Intenta hablar más cerca del micrófono.")
        sys.exit(0)

    save_wav(out_wav, audio, 16000)
    print(f"[OK] Audio guardado en: {out_wav}")

    print("[STT] Transcribiendo…")
    text = transcribe(out_wav, whisper_size=args.whisper_size, device=args.device)
    print("=== Transcripción ===")
    print(text if text else "(vacío)")

    pipe = joblib.load(model_path)
    tokens = tokenize(text)
    X = featurize_tokens(tokens)
    y_pred = pipe.predict(X)

    spans = bio_to_spans(tokens, y_pred)
    ent_map = {}
    for s in spans:
        ent_map.setdefault(s["type"], s["text"])
    
    ent_map = fix_entities_with_rules(text, ent_map)

    order = ["NOMBRES","APELLIDOS","DNI","TIPO_DOC","FECHA_NAC","TELEFONO","DISTRITO","DIRECCION"]
    print("\n=== Entidades extraídas ===")
    for k in order:
        v = ent_map.get(k, "—")
        print(f"{k.title().replace('_',' ')}: {v}")

    print("\nJSON:")
    print(json.dumps(ent_map, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
