#!/usr/bin/env python3
"""Serve static files + /api/run/<id> to execute tasks and return JSON records."""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.tasks import TASKS
from runner.record import run_task


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/metrics/pair/"):
            pair = path.split("/api/metrics/pair/", 1)[1].strip("/")
            from runner.metrics import score_pair

            mapping = {"s1-q1": ("cuda-s1-q1", "cann-s1-q1")}
            if pair not in mapping:
                self.send_error(404, f"Unknown pair {pair}")
                return
            cuda_id, cann_id = mapping[pair]
            data = json.dumps(score_pair(cuda_id, cann_id), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/api/record/"):
            tid = path.split("/api/record/", 1)[1].strip("/")
            fp = ROOT / "simulation-records" / f"{tid}.json"
            if not fp.exists():
                self.send_error(404, f"No record for {tid}")
                return
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/run/"):
            tid = path.split("/api/run/", 1)[1].strip("/")
            if tid not in TASKS:
                self.send_error(404, f"Unknown task {tid}")
                return
            try:
                record = run_task(tid, TASKS[tid]["run"])
                data = json.dumps(record, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                self.send_error(500, str(exc))
            return
        self.send_error(404)


def main() -> None:
    port = 8765
    host = "0.0.0.0"  # 允许 localhost / 127.0.0.1 / 本机 IP 访问
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving {ROOT}", flush=True)
    print(f"  → http://127.0.0.1:{port}/", flush=True)
    print(f"  → http://localhost:{port}/operator-novice-workflow.html", flush=True)
    print("API: POST /api/run/<id>  GET /api/record/<id>  GET /api/metrics/pair/<id>", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
