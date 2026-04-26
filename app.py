from flask import Flask, request, jsonify, render_template, send_file
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
import io
import google.generativeai as genai
import os
import json
import datetime
import sqlite3
import re
from dotenv import load_dotenv

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_03d40c10c97ae26b8ba135ec221828be9c219958b43bf86a")
set_api_key(ELEVENLABS_API_KEY)

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAMhON-34Hps54Wcu-zmH8zvoJIn6gqSZc")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

if OCR_AVAILABLE:
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DB_PATH = "techmorph.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    NOT NULL,
            mode        TEXT    NOT NULL,
            input_text  TEXT    NOT NULL,
            target_lang TEXT,
            output      TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()


init_db()


def save_history(mode, input_text, output, target_lang=None):
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO history (created_at, mode, input_text, target_lang, output) VALUES (?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), mode,
             input_text[:2000], target_lang, output[:5000])
        )
        con.commit()
        con.close()
    except Exception:
        pass


LANG_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "te": "Telugu",
    "ta": "Tamil",  "kn": "Kannada", "gu": "Gujarati",
    "bn": "Bengali", "pa": "Punjabi", "ur": "Urdu",
    "ml": "Malayalam"
}


def get_keywords(text):
    if SPACY_AVAILABLE:
        doc = nlp(text)
        return list({t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 2})
    words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b|\b[a-zA-Z]{5,}\b', text)
    seen, result = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            result.append(w)
    return result[:20]


def clean_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    if not OCR_AVAILABLE:
        return jsonify({"error": "Tesseract / Pillow not installed."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    try:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        return jsonify({"text": text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    lang = data.get("target_lang", "hi")
    lang_name = LANG_NAMES.get(lang, lang)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    terms = get_keywords(text)
    if not terms:
        return jsonify({"error": "No technical keywords found"}), 400

    prompt = f"""You are a technical translator for Indian regional languages.
From the text below I have extracted these keywords:
{terms}

For each keyword provide:
1. The English word
2. Its {lang_name} translation/equivalent
3. A one-line simple explanation in {lang_name}

Return ONLY a valid JSON array (no markdown fences, no preamble):
[
  {{"english": "word", "translated": "{lang_name} translation", "meaning": "simple explanation in {lang_name}"}},
  ...
]

Source text: {text}"""

    res = None
    try:
        res = model.generate_content(prompt)
        parsed = json.loads(clean_json(res.text))
        save_history("translate", text, json.dumps(parsed), lang)
        return jsonify({"terms": parsed, "lang": lang_name})
    except Exception as e:
        return jsonify({"error": str(e), "raw": res.text[:300] if res else ""}), 500


@app.route('/ocr_translate', methods=['POST'])
def ocr_translate():
    lang = request.form.get("target_lang", "hi")
    lang_name = LANG_NAMES.get(lang, lang)

    if 'image' in request.files and request.files['image'].filename:
        if not OCR_AVAILABLE:
            return jsonify({"error": "Tesseract not installed."}), 500
        file = request.files['image']
        try:
            img = Image.open(file)
            text = pytesseract.image_to_string(img).strip()
        except Exception as e:
            return jsonify({"error": f"OCR failed: {e}"}), 500
        if not text:
            return jsonify({"error": "No text found in image. Try a clearer photo."}), 400
    elif request.form.get("text"):
        text = request.form.get("text").strip()
    else:
        return jsonify({"error": "No image or text provided"}), 400

    terms = get_keywords(text)
    if not terms:
        return jsonify({"error": "No keywords found in the text"}), 400

    prompt = f"""You are a technical translator for Indian regional languages.
From the text below I have identified these technical keywords:
{terms}

For each keyword provide:
1. The English word
2. Its {lang_name} translation/equivalent
3. A one-line simple explanation in {lang_name}

Return ONLY a valid JSON array (no markdown fences):
[
  {{"english": "word", "translated": "{lang_name} translation", "meaning": "simple explanation in {lang_name}"}},
  ...
]

Text: {text}"""

    res = None
    try:
        res = model.generate_content(prompt)
        parsed = json.loads(clean_json(res.text))
        save_history("ocr_translate", text, json.dumps(parsed), lang)
        return jsonify({"ocr_text": text, "terms": parsed, "lang": lang_name})
    except Exception as e:
        return jsonify({"error": str(e), "raw": res.text[:300] if res else ""}), 500


@app.route('/exam', methods=['POST'])
def exam():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prompt = f"""You are a helpful study assistant for students. Analyze the following text and return ONLY a valid JSON object (no markdown fences, no preamble):

{{
  "summary": "A clear 3-4 sentence summary a student can easily understand",
  "steps": [
    {{"step": 1, "title": "Short title", "detail": "One sentence explanation"}},
    {{"step": 2, "title": "Short title", "detail": "One sentence explanation"}}
  ],
  "keywords": ["key", "terms", "from", "the", "text"],
  "difficulty": "Easy"
}}

Rules:
- "difficulty" must be exactly one of: Easy, Medium, Hard
- Include 4-8 steps
- Include 5-10 keywords
- Keep language simple and student-friendly

Text to analyze:
{text}"""

    res = None
    try:
        res = model.generate_content(prompt)
        parsed = json.loads(clean_json(res.text))
        save_history("exam", text, json.dumps(parsed))
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e), "raw": res.text[:300] if res else ""}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    context = (data.get("context") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    system = "You are TechMorph AI, a helpful assistant for students learning technical subjects. Be concise, clear, and encouraging."
    ctx_part = f"\n\nContext from their text:\n{context[:800]}" if context else ""
    prompt = f"{system}{ctx_part}\n\nStudent question: {question}\n\nAnswer in 2-4 sentences, simply and clearly:"

    res = None
    try:
        res = model.generate_content(prompt)
        return jsonify({"answer": res.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM history ORDER BY id DESC LIMIT 30")
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return jsonify(rows)
    except Exception:
        return jsonify([])


@app.route('/history/<int:entry_id>', methods=['DELETE'])
def delete_history(entry_id):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM history WHERE id=?", (entry_id,))
        con.commit()
        con.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── TTS via ElevenLabs ────────────────────────────────────────────────────────
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 4000:
        text = text[:4000]

    # Current free-tier models (2025): eleven_turbo_v2_5, eleven_turbo_v2
    # eleven_monolingual_v1 and eleven_multilingual_v2 are DEPRECATED & removed from free tier
    MODELS_TO_TRY = [
        ("pNInz6obpgDQGcFmaJgB", "eleven_turbo_v2_5"),   # Adam  + turbo v2.5
        ("21m00Tcm4TlvDq8ikWAM", "eleven_turbo_v2_5"),   # Rachel + turbo v2.5
        ("pNInz6obpgDQGcFmaJgB", "eleven_turbo_v2"),     # Adam  + turbo v2
        ("21m00Tcm4TlvDq8ikWAM", "eleven_turbo_v2"),     # Rachel + turbo v2
    ]

    last_error = "Unknown error"
    for voice_id, model_id in MODELS_TO_TRY:
        try:
           audio = client.text_to_speech.convert(
    text=text,
    voice_id=voice_id,
    model_id=model_id
)

audio_bytes = b"".join(audio)

return send_file(
    io.BytesIO(audio_bytes),
    mimetype="audio/mpeg",
    as_attachment=False,
    download_name="speech.mp3"
)
        except Exception as e:
            last_error = str(e)
            continue

    return jsonify({"error": f"All TTS attempts failed. Last error: {last_error}"}), 500


if __name__ == '__main__':
    print("\n TechMorph starting at http://localhost:5000\n")
    app.run(debug=True, port=5000)