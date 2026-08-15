"""
Local web interface for the bone-scaffold framework.

    .venv/bin/python webapp/server.py          # then open http://127.0.0.1:8000

Standard library only - no Flask, no build step, no network install. That is deliberate:
this exists to be demonstrated live, and the failure mode of a demo is a missing
dependency five minutes beforehand.

It is a thin skin over the same code paths the CLI uses. /api/predict calls
tools.predict.predict, /api/pipeline runs run_pipeline.py as a subprocess. Nothing is
reimplemented here, so the browser cannot disagree with the terminal.

Bound to 127.0.0.1 by design. The pipeline endpoint starts a fixed command with no
user-supplied arguments, and figures are served from a whitelist rather than a path the
caller controls - a local tool is still worth not making into a file-read primitive.
"""

import json
import re
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.predict import predict, list_configs        # noqa: E402
from pipeline import config as C                       # noqa: E402

sys.path.insert(0, str(ROOT / "sim"))
import physics as phys                                 # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE_NAME = re.compile(r"^fig[0-9a-zA-Z_]+\.png$")


# --------------------------------------------------------------------------
# Pipeline runner - one at a time, output streamed to a buffer the UI polls
# --------------------------------------------------------------------------

class PipelineRun:
    def __init__(self):
        self.lines, self.proc, self.lock = [], None, threading.Lock()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.running:
            return False
        with self.lock:
            self.lines = []
        # -u so stage headings reach the browser as they happen rather than in one burst
        # when the process exits.
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "run_pipeline.py"], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._drain, daemon=True).start()
        return True

    def _drain(self):
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))
        self.proc.wait()
        with self.lock:
            self.lines.append(f"__EXIT__{self.proc.returncode}")

    def since(self, offset):
        with self.lock:
            return self.lines[offset:], len(self.lines)


RUN = PipelineRun()


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def api_meta():
    """Everything the form needs to populate itself, straight from the source tables."""
    cfg = list_configs().reset_index()
    return dict(
        polymers=sorted(phys.POLYMERS),
        ceramics=sorted(phys.CERAMICS),
        sites={k: v for k, v in phys.BONE_SITES.items()},
        configs=cfg.to_dict("records"),
        windows=dict(pore_um=list(C.PORE_WINDOW_UM), min_strut_um=C.MIN_STRUT_UM,
                     porosity=list(C.POROSITY_WINDOW)),
        figures=sorted(p.name for p in (C.FIGURES).glob("fig*.png"))
        if C.FIGURES.exists() else [])


def api_results():
    """Headline numbers from the last pipeline run, for the dashboard."""
    path = C.RESULTS / "metrics.json"
    if not path.exists():
        return dict(available=False)
    m = json.loads(path.read_text())
    ladder = m.get("model_ladder", [])
    best = max(ladder, key=lambda r: r.get("r2", -9)) if ladder else None
    ga = next((r for r in ladder if r["model"] == "gibson_ashby_global"), None)
    return dict(available=True, dataset=m.get("dataset", {}), ladder=ladder,
                best=best, gibson_ashby=ga, leakage=m.get("leakage", {}),
                convergence=m.get("convergence", {}), fusion=m.get("fusion", {}),
                inverse=m.get("inverse", {}),
                figures=sorted(p.name for p in C.FIGURES.glob("fig*.png"))
                if C.FIGURES.exists() else [])


def api_predict(body):
    """One specification -> the same dict the CLI prints, plus arrays for the chart."""
    def num(key, default, cast=float):
        v = body.get(key, default)
        return default if v in (None, "") else cast(v)

    return predict(
        topology=body.get("topology", "gyroid"),
        mode=body.get("mode", "network"),
        porosity=num("porosity", 0.65),
        polymer=body.get("polymer", "PLGA_85_15"),
        ceramic=body.get("ceramic", "none"),
        ceramic_wt=num("ceramic_wt", 0.0),
        site=body.get("site", "trabecular_mid"),
        n_cells=num("n_cells", 1, int),
        n_struts=num("n_struts", 3, int),
        n_layers=num("n_layers", 12, int),
        solve=bool(body.get("solve", False)),
        include_trajectory=True)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "ScaffoldLab"

    def log_message(self, fmt, *args):            # keep the console readable
        if "/api/pipeline/log" not in (args[0] if args else ""):
            sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    # -- helpers ------------------------------------------------------------
    def _send(self, code, payload, ctype="application/json", raw=False):
        data = payload if raw else json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass                                   # user navigated away mid-response

    def _file(self, path, ctype):
        if not path.exists():
            return self._send(404, dict(error=f"{path.name} not found"))
        self._send(200, path.read_bytes(), ctype, raw=True)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(HERE / "index.html", "text/html; charset=utf-8")
        if u.path == "/api/meta":
            return self._send(200, api_meta())
        if u.path == "/api/results":
            return self._send(200, api_results())
        if u.path == "/api/pipeline/log":
            offset = int(q.get("offset", ["0"])[0])
            lines, total = RUN.since(offset)
            return self._send(200, dict(lines=lines, total=total, running=RUN.running))
        if u.path == "/api/figure":
            name = q.get("name", [""])[0]
            # Whitelist-matched and re-resolved: never join user input onto a path and
            # trust the result.
            if not FIGURE_NAME.match(name):
                return self._send(400, dict(error="bad figure name"))
            target = (C.FIGURES / name).resolve()
            if target.parent != C.FIGURES.resolve():
                return self._send(400, dict(error="bad figure path"))
            return self._file(target, "image/png")
        return self._send(404, dict(error="no such route"))

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, dict(error="body must be JSON"))

        if u.path == "/api/predict":
            try:
                return self._send(200, api_predict(body))
            except (KeyError, ValueError, RuntimeError) as exc:
                return self._send(400, dict(error=str(exc).strip('"')))
            except Exception as exc:                                  # noqa: BLE001
                return self._send(500, dict(error=f"{type(exc).__name__}: {exc}"))
        if u.path == "/api/pipeline/start":
            started = RUN.start()
            return self._send(200, dict(started=started, running=RUN.running))
        return self._send(404, dict(error="no such route"))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  Bone-scaffold lab  ->  http://127.0.0.1:{port}\n"
          f"  serving from {ROOT}\n  Ctrl-C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
