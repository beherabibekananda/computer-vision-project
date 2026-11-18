import http.server
import socketserver
import os
import json
import re

PORT = 5173

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
        if self.path in ('/', '/index.html', ''):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open('preview.html', 'rb') as f:
                self.wfile.write(f.read())
            return
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'online', 'service': 'Fieldwatch UI'}).encode())
            return
        super().do_GET()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"🚀 Fieldwatch Frontend running on http://localhost:{PORT}")
    httpd.serve_forever()
