#!/usr/bin/env python3
"""Launch a local HTTP server for the StoryLSTM web demo."""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
DIR = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)

    def end_headers(self):
        if self.path.endswith(".wasm"):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "/vocab.json" in str(args) or "/config.json" in str(args):
            return
        super().log_message(fmt, *args)


if __name__ == "__main__":
    os.chdir(DIR)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", PORT), Handler) as httpd:
        print(f"StoryLSTM demo -> http://localhost:{PORT}")
        print(f"Serving: {DIR}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
