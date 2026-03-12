#!/usr/bin/env python3
from http.server import HTTPServer
from progress_dashboard import Handler, PORT

if __name__ == "__main__":
    print(f"Dashboard(wrapper): http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
