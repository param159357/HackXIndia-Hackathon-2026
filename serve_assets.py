#!/usr/bin/env python3

from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    port = 8002
    server_address = ('', port)
    
    os.chdir('../skillmap')
    
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f'Starting CORS-enabled server on port {port}...')
    print(f'Serving directory: {os.getcwd()}')
    print(f'Assets available at: http://localhost:{port}/assets/')
    httpd.serve_forever()
