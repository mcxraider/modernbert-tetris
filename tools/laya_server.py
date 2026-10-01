#!/usr/bin/env python3
"""Local HTTP server that lets the battle page play against Laya.

Laya is an open-weight typed-decision model (Convai Innovations). Two runtimes
are supported, picked automatically:

  * laya-mlx  (Apple Silicon, MLX)       pip install laya-mlx
  * laya      (PyTorch, any platform)    pip install laya   or  pip install torch transformers laya

Both expose `system_one(state, questions)` with the same request and response
shape as TypeSafe's /v1/systemone, so this server is a thin wrapper:

  POST /v1/systemone   {"state": ..., "questions": {...}}  ->  {"model", "answers", "usage"}
  GET  /v1/models      which checkpoint and runtime are loaded
  GET  /health         {"ok": true}

Run it, then pick "Laya (local)" as the opponent on the battle page:

  python tools/laya_server.py                       # default checkpoint, port 8765
  python tools/laya_server.py --model convaiinnovations/laya-typed-decisions   # bigger 1024-token context
  python tools/laya_server.py --model aac6fef/laya-mlx        # MLX checkpoint on a Mac

CORS is open so the page (local or on jev-tetris.vercel.app) can call it
directly from the browser; browsers allow https pages to reach http://localhost.
"""

import argparse
import json
import platform
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ThreadingHTTPServer6(ThreadingHTTPServer):
    address_family = socket.AF_INET6

DEFAULT_MODELS = {
    "mlx": "aac6fef/laya-mlx",
    "torch": "convaiinnovations/laya",
}


def load_agent(model_id, runtime):
    if runtime in ("auto", "mlx"):
        try:
            import laya_mlx  # noqa: F401  (Apple Silicon only)

            agent = laya_mlx.load(model_id or DEFAULT_MODELS["mlx"])
            return agent, "laya-mlx", model_id or DEFAULT_MODELS["mlx"]
        except ImportError:
            if runtime == "mlx":
                raise
    import laya  # PyTorch reference implementation

    chosen = model_id or DEFAULT_MODELS["torch"]
    agent = laya.load(chosen)
    return agent, "laya (torch, %s)" % getattr(getattr(agent, "device", None), "type", "cpu"), chosen


def make_handler(agent, runtime_name, model_id, verbose):
    stats = {"requests": 0, "total_ms": 0.0}

    class Handler(BaseHTTPRequestHandler):
        server_version = "laya-tetris/1.0"

        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass  # the page gave up on this move (piece landed first)

        def end_headers(self):
            # Every response carries the CORS headers, including the error pages
            # BaseHTTPRequestHandler writes itself, so the browser's preflight
            # never sees a reply without them.
            self._cors()
            super().end_headers()

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "600")
            # Chrome's Private Network Access preflight for https -> http://localhost
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self):  # CORS preflight, any path
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "runtime": runtime_name, "model": model_id})
            elif self.path.startswith("/v1/models"):
                self._send(200, {"models": [{"name": model_id, "runtime": runtime_name, "platform": platform.platform()}],
                                 "stats": {**stats, "avg_ms": stats["total_ms"] / stats["requests"] if stats["requests"] else None}})
            else:
                self._send(404, {"detail": {"error_type": "not_found", "message": "Unknown route."}})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                self._send(404, {"detail": {"error_type": "not_found", "message": "POST /v1/systemone"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                state, questions = request.get("state"), request.get("questions")
                if state is None or not isinstance(questions, dict) or not questions:
                    self._send(422, {"detail": {"error_type": "validation_error", "message": "Body needs `state` and a non-empty `questions` map."}})
                    return
                started = time.perf_counter()
                result = agent.system_one(state, questions)
                elapsed = (time.perf_counter() - started) * 1000
                stats["requests"] += 1
                stats["total_ms"] += elapsed
                result.setdefault("model", model_id)
                result["runtime"] = runtime_name
                result["inference_ms"] = round(elapsed, 1)
                if verbose:
                    print("%s  %d question(s)  %.0f ms  %s" % (time.strftime("%H:%M:%S"), len(questions), elapsed, list(questions)[:3]), flush=True)
                self._send(200, result)
            except Exception as error:  # noqa: BLE001
                self._send(500, {"detail": {"error_type": "inference_error", "message": str(error)}})

        def log_message(self, *args):  # quiet by default
            if verbose:
                super().log_message(*args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="checkpoint id or local path (default depends on runtime)")
    parser.add_argument("--runtime", choices=["auto", "mlx", "torch"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args()

    print("Loading Laya (%s)..." % args.runtime, flush=True)
    started = time.perf_counter()
    agent, runtime_name, model_id = load_agent(args.model, args.runtime)
    print("Loaded %s with %s in %.1f s" % (model_id, runtime_name, time.perf_counter() - started), flush=True)

    # Warm up once so the first real decision is not the slow one.
    agent.system_one("warm up", {"q": {"type": "noul", "instructions": "Is this a warm-up?"}})

    handler = make_handler(agent, runtime_name, model_id, args.verbose)
    servers = []
    hosts = [args.host]
    # "localhost" resolves to ::1 on some machines and 127.0.0.1 on others, so the
    # default binds both loopback addresses; anything else binds what was asked.
    if args.host in ("127.0.0.1", "localhost"):
        hosts = ["127.0.0.1", "::1"]
    for host in hosts:
        try:
            server = ThreadingHTTPServer6((host, args.port), handler) if ":" in host else ThreadingHTTPServer((host, args.port), handler)
        except OSError as error:
            if host == "::1":
                continue  # no IPv6 loopback, fine
            print("Cannot listen on %s:%d: %s" % (host, args.port, error), file=sys.stderr)
            print("Another program is probably using port %d. Run with --port 8766 and enter http://localhost:8766 on the page." % args.port, file=sys.stderr)
            return 1
        servers.append(server)
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Laya server listening on http://localhost:%d  (POST /v1/systemone; %s)" % (args.port, ", ".join(hosts[: len(servers)])), flush=True)
    print("Now open the battle page, pick 'Laya (local, open weights)' and press Start.", flush=True)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
