from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl

class MyHttpRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        '.manifest': 'text/cache-manifest',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.svg':	'image/svg+xml',
        '.css':	'text/css',
        '.js':	'application/x-javascript',
        # '.wasm':	'application/wasm',
        # '.json':	'application/json',
        # '.webapp':	'application/x-web-app-manifest+json',
        # '.xhtml':	'application/xhtml+xml',
        # '.xml':	'application/xml',
    }

if __name__ == "__main__":
    server_address = ('localhost', 8000)
    httpd = HTTPServer(server_address, MyHttpRequestHandler)
    print('Serving at http://localhost:8000')
    httpd.serve_forever()
