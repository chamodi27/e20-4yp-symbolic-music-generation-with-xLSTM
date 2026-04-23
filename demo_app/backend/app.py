"""
app.py
======
Flask API server for xLSTM Music Generation.

Endpoints:
  GET  /health    — liveness check
  POST /generate  — generate a MIDI file from a prompt

Usage:
  export MODEL_PATH=/path/to/checkpoint-46000-last
  python app.py

Configuration via environment variables:
  MODEL_PATH      — (required) path to checkpoint folder
  PORT            — server port (default: 5000)
  HOST            — server host (default: 0.0.0.0)
  CONTEXT_LENGTH  — inference context length override (default: 16384)
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from converter import MIDIConverter
from generator import xLSTMGenerator
from token_analysis import analyse_tokens

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xlstm-api")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
MODEL_PATH     = os.environ.get("MODEL_PATH", "")
PORT           = int(os.environ.get("PORT", 5000))
HOST           = os.environ.get("HOST", "0.0.0.0")  # NOTE: 'HOST' is overridden by conda to the toolchain
API_HOST       = os.environ.get("API_HOST", "0.0.0.0")  # Use API_HOST to avoid conda collision
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", 16384))

# Generation parameter limits
MIN_LENGTH    = 100
MAX_LENGTH    = 12288
MIN_TEMP      = 0.1
MAX_TEMP      = 2.0
DEFAULT_PROMPT = "s-9 o-0 t-38"

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow all origins — restrict in production if needed

# ---------------------------------------------------------------------------
# Load model + converter at startup (once)
# ---------------------------------------------------------------------------
generator: xLSTMGenerator = None
converter: MIDIConverter = None

def load_model():
    global generator, converter
    if not MODEL_PATH:
        raise RuntimeError(
            "MODEL_PATH environment variable is not set. "
            "Set it to the path of your checkpoint folder.\n"
            "Example: export MODEL_PATH=/path/to/checkpoint-46000-last"
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at MODEL_PATH={MODEL_PATH!r}")

    logger.info("Loading xLSTMGenerator from %s ...", MODEL_PATH)
    generator = xLSTMGenerator(
        model_path_or_repo=MODEL_PATH,
        config_overrides={"context_length": CONTEXT_LENGTH},
        device="auto",
    )
    logger.info("Loading MIDIConverter ...")
    converter = MIDIConverter()
    logger.info("All components ready. API server starting.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness check."""
    return jsonify({
        "status": "ok",
        "model_loaded": generator is not None,
        "model_path": MODEL_PATH,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate a MIDI file from a REMIGEN2 prompt.

    Request JSON:
      prompt      : str   (default: "s-9 o-0 t-38")
      length      : int   (default: 2048, range: 100–12288)
      temperature : float (default: 0.8,  range: 0.1–2.0)
      seed        : int   (optional, for reproducibility)

    Response:
      Binary MIDI file (audio/midi) with custom X- headers for metadata.
    """
    if generator is None:
        return jsonify({"error": "Model not loaded"}), 503

    # ------------------------------------------------------------------ parse
    data = request.get_json(silent=True) or {}

    prompt      = str(data.get("prompt", DEFAULT_PROMPT)).strip()
    length      = data.get("length", 2048)
    temperature = data.get("temperature", 0.8)
    seed        = data.get("seed", None)

    # ------------------------------------------------------------------ validate
    errors = []
    if not prompt:
        errors.append("'prompt' must be a non-empty string.")
    if not isinstance(length, int) or not (MIN_LENGTH <= length <= MAX_LENGTH):
        errors.append(f"'length' must be an integer between {MIN_LENGTH} and {MAX_LENGTH}.")
    if not isinstance(temperature, (int, float)) or not (MIN_TEMP <= float(temperature) <= MAX_TEMP):
        errors.append(f"'temperature' must be a number between {MIN_TEMP} and {MAX_TEMP}.")
    if seed is not None and not isinstance(seed, int):
        errors.append("'seed' must be an integer or null.")
    if errors:
        return jsonify({"error": "Invalid request", "details": errors}), 400

    temperature = float(temperature)

    # ------------------------------------------------------------------ generate
    logger.info(
        "Generating: length=%d temp=%.2f seed=%s prompt=%r",
        length, temperature, seed, prompt[:40],
    )
    try:
        result = generator.generate(
            prompt=prompt,
            max_length=length,
            temperature=temperature,
            seed=seed,
        )
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        return jsonify({"error": "Generation failed", "details": str(exc)}), 500

    tokens = result["tokens"]

    # ------------------------------------------------------------------ analyse
    analysis = analyse_tokens(tokens)

    # ------------------------------------------------------------------ convert to MIDI
    tmp_dir  = tempfile.mkdtemp()
    midi_path = os.path.join(tmp_dir, f"output_{uuid.uuid4().hex}.mid")

    success = converter.tokens_to_midi(tokens, midi_path, use_clean_fallback=True)
    if not success:
        return jsonify({
            "error": "MIDI conversion failed",
            "grammar_error_rate": round(analysis.grammar_error_rate, 6),
            "actual_tokens": result["actual_tokens"],
        }), 422

    logger.info(
        "Done: tokens=%d bars=%d errors=%.4f time=%.2fs tps=%.1f",
        result["actual_tokens"],
        analysis.num_bars,
        analysis.grammar_error_rate,
        result["generation_time_s"],
        result["tokens_per_second"],
    )

    # ------------------------------------------------------------------ respond
    response = send_file(
        midi_path,
        mimetype="audio/midi",
        as_attachment=True,
        download_name="generated.mid",
    )
    # Attach generation metadata as custom headers (useful for client-side display)
    response.headers["X-Generation-Time"]    = round(result["generation_time_s"], 2)
    response.headers["X-Tokens-Per-Second"]  = round(result["tokens_per_second"], 2)
    response.headers["X-Actual-Tokens"]      = result["actual_tokens"]
    response.headers["X-Num-Bars"]           = analysis.num_bars
    response.headers["X-Grammar-Error-Rate"] = round(analysis.grammar_error_rate, 6)
    response.headers["X-Target-Reached"]     = str(result["target_reached"]).lower()
    # Expose headers to browsers (CORS)
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Generation-Time, X-Tokens-Per-Second, X-Actual-Tokens, "
        "X-Num-Bars, X-Grammar-Error-Rate, X-Target-Reached"
    )
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_model()
    logger.info("Starting Flask server on %s:%d", API_HOST, PORT)
    # threaded=False — model is NOT thread-safe; requests queue naturally
    app.run(host=API_HOST, port=PORT, debug=False, threaded=False)
