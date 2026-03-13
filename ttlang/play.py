"""
Interactive Pong world model server (HTTP polling, no WebSocket).

Runs on the TT server inside Docker. Generates frames on TT hardware
in response to player actions received via HTTP POST.

Usage:
  1. Run in Docker: cd /tmp && python3 play.py
  2. SSH tunnel: ssh -L 8765:localhost:8765 zcarver@bh-qbae-15
     (+ socat if needed for Docker bridge networking)
  3. Open http://localhost:8765 in your browser
  4. Use arrow keys to play!
"""

import json
import io
import base64
import time
import torch
import torch.nn.functional as F
import ttnn
import ttl
import http.server
import threading

from sample_v2 import (
    sample_frame, trim_kv_cache, preload_weights, prealloc_scratch,
    extend_rope_tables, to_tt, zeros_tt,
    D_MODEL, D_HEAD, TILE, N_BLOCKS, TOKS_PER_FRAME, SEQ_PADDED, HEIGHT, WIDTH,
)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Pong World Model - TT Hardware</title>
<style>
  body { background: #111; color: #eee; font-family: monospace; text-align: center; margin-top: 40px; }
  canvas { border: 2px solid #555; image-rendering: pixelated; }
  .info { margin: 16px; font-size: 14px; color: #aaa; }
  .status { font-size: 18px; margin: 12px; }
  .controls { margin: 16px; font-size: 14px; color: #888; }
  kbd { background: #333; padding: 2px 8px; border-radius: 3px; border: 1px solid #555; }
</style>
</head>
<body>
<h1>Pong World Model on Tenstorrent</h1>
<div class="status" id="status">Loading model...</div>
<canvas id="game" width="480" height="480"></canvas>
<div class="controls">
  <kbd>&uarr;</kbd> Up &nbsp;&nbsp;
  <kbd>&darr;</kbd> Down &nbsp;&nbsp;
  <kbd>Space</kbd> Stay &nbsp;&nbsp;
  <kbd>1</kbd>-<kbd>8</kbd> Steps &nbsp;&nbsp;
</div>
<div class="info">
  <span id="fps">--</span> &nbsp;|&nbsp;
  Frame <span id="frame_num">0</span> &nbsp;|&nbsp;
  Steps: <span id="steps_info">6</span> &nbsp;|&nbsp;
  Cache: <span id="cache_info">0</span> frames
</div>

<script>
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
let currentAction = 1; // 1=stay
let nSteps = 6;
let running = false;

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowUp') { currentAction = 2; e.preventDefault(); }
  else if (e.key === 'ArrowDown') { currentAction = 3; e.preventDefault(); }
  else if (e.key === ' ') { currentAction = 1; e.preventDefault(); }
  else if (e.key >= '1' && e.key <= '8') {
    nSteps = parseInt(e.key);
    document.getElementById('steps_info').textContent = nSteps;
    e.preventDefault();
  }
  if (!running) { running = true; generateLoop(); }
});

async function generateLoop() {
  status.textContent = 'Generating...';
  status.style.color = '#ff4';
  while (true) {
    const action = currentAction;
    status.textContent = 'Generating frame (action=' + ['?','stay','up','down'][action] + ')...';
    try {
      const resp = await fetch('/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: action, n_steps: nSteps})
      });
      const data = await resp.json();
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, 480, 480); };
      img.src = 'data:image/png;base64,' + data.image;
      document.getElementById('frame_num').textContent = data.frame_idx;
      document.getElementById('cache_info').textContent = data.cached_frames;
      document.getElementById('fps').textContent = (1.0 / data.elapsed).toFixed(1) + ' fps';
      status.textContent = 'Playing! Press arrows to steer.';
      status.style.color = '#4f4';
    } catch (err) {
      status.textContent = 'Error: ' + err;
      status.style.color = '#f44';
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

// Draw initial screen
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 480, 480);
ctx.fillStyle = '#555';
ctx.font = '16px monospace';
ctx.textAlign = 'center';
ctx.fillText('Press an arrow key to start', 240, 240);

// Check if model is ready
fetch('/status').then(r => r.json()).then(d => {
  status.textContent = 'Ready! Press an arrow key to start.';
  status.style.color = '#4f4';
});
</script>
</body>
</html>"""


class GameState:
    def __init__(self):
        self.tt_device = None
        self.state = None
        self.dev = None
        self.scr = None
        self.scaler_tt = None
        self.mean_scale_tt = None
        self.mean_scale_16_tt = None
        self.device_kv_cache = None
        self.frame_idx = 0
        self.n_steps = 6
        self.cfg = 1.0
        self.n_window = 30
        self.lock = threading.Lock()

    def init_model(self):
        print("Opening TT device...")
        self.tt_device = ttnn.open_device(device_id=0)
        torch.manual_seed(42)
        print("Loading weights...")
        ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
        self.state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}
        self.scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), self.tt_device)
        self.mean_scale_tt = to_tt(
            torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), self.tt_device)
        self.mean_scale_16_tt = to_tt(
            torch.full((TILE, TILE), 1.0 / D_HEAD, dtype=torch.bfloat16), self.tt_device)
        print("Pre-loading weights to device...")
        self.dev = preload_weights(self.state, self.tt_device)
        print("Pre-allocating scratch buffers...")
        self.scr = prealloc_scratch(self.tt_device)
        print("Extending RoPE tables...")
        extend_rope_tables(self.state)
        print("Model ready!")

    def generate_frame(self, action, n_steps=None):
        with self.lock:
            steps = n_steps if n_steps is not None else self.n_steps
            noise = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
            t0 = time.time()
            frame, new_device_kv = sample_frame(
                noise, action, steps, self.cfg,
                self.state, self.dev, self.scr, self.tt_device,
                self.scaler_tt, self.mean_scale_tt, self.mean_scale_16_tt,
                device_kv_cache=self.device_kv_cache, frame_idx=self.frame_idx,
            )
            self.device_kv_cache = trim_kv_cache(new_device_kv, self.n_window)
            elapsed = time.time() - t0
            cached_frames = min(self.frame_idx, self.n_window - 1)
            fidx = self.frame_idx
            self.frame_idx += 1

            img = ((frame[0].float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            img_np = img.permute(1, 2, 0).numpy()

            from PIL import Image
            pil_img = Image.fromarray(img_np).resize((240, 240), Image.NEAREST)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            print(f"Frame {fidx} (action={action}, cache={cached_frames}) in {elapsed:.1f}s")
            return {
                'image': img_b64,
                'frame_idx': fidx,
                'cached_frames': cached_frames,
                'elapsed': elapsed,
            }


game = GameState()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ready'}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            action = max(0, min(3, int(data.get('action', 1))))
            n_steps = max(1, min(8, int(data.get('n_steps', 6))))
            result = game.generate_frame(action, n_steps=n_steps)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default logging


if __name__ == "__main__":
    game.init_model()
    server = http.server.HTTPServer(('0.0.0.0', 8765), Handler)
    print(f"Server listening on 0.0.0.0:8765")
    print(f"Open http://localhost:8765 in your browser")
    server.serve_forever()
