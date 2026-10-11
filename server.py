"""Local realtime TTS: sherpa-onnx Kokoro on Apple Silicon. POST /say {"text","voice"} -> wav."""
import io, json, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import sherpa_onnx

D = Path('/tmp/kokoro-en-v0_19')
SID = {'af_heart': 0, 'af_bella': 1, 'af_sarah': 3, 'af_sky': 4,
       'am_adam': 5, 'am_michael': 6, 'bf_emma': 7, 'bm_george': 9}

cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(D / 'model.onnx'), voices=str(D / 'voices.bin'),
            tokens=str(D / 'tokens.txt'), data_dir=str(D / 'espeak-ng-data')),
        num_threads=4))
assert cfg.validate(), 'bad model dir'
tts = sherpa_onnx.OfflineTts(cfg)

def synth(text, voice):
    a = tts.generate(text=text, sid=SID.get(voice, 0), speed=1.0)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(a.sample_rate)
        import numpy as np
        w.writeframes((np.array(a.samples) * 32767).astype('<i2').tobytes())
    return buf.getvalue()

print('warming up...', flush=True)
synth('Ready.', 'af_heart')
print('ready', flush=True)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        self.send_response(200); self._cors(); self.send_header('Content-Type', 'text/plain'); self.end_headers()
        self.wfile.write(b'voice server ok')
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try: body = json.loads(self.rfile.read(n) or b'{}')
        except Exception: body = {}
        text = str(body.get('text', '')).strip()[:500]
        voice = str(body.get('voice') or 'af_heart')
        if not text:
            self.send_response(400); self._cors(); self.end_headers(); return
        try: wav = synth(text, voice)
        except Exception as e:
            self.send_response(500); self._cors(); self.end_headers()
            self.wfile.write(('error: ' + str(e)[:200]).encode()); return
        self.send_response(200); self._cors()
        self.send_header('Content-Type', 'audio/wav'); self.send_header('Content-Length', str(len(wav)))
        self.end_headers(); self.wfile.write(wav)

HTTPServer(('127.0.0.1', 8765), H).serve_forever()
