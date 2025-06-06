#!/usr/bin/env python3
"""
Main entry point for the Trail Camera Detection System
"""

from app import create_app

if __name__ == '__main__':
    app = create_app()
    print("Starting Trail Camera Web App...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)