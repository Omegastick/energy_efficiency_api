#!/usr/bin/env python
"""
Run a development server for the application.
"""

from server import app

if __name__ == '__main__':
    app.run(host='0.0.0.0')
