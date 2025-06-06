"""
Flask application factory
"""

from flask import Flask
from pathlib import Path
import os

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

    # Setup paths
    current_dir = Path.cwd()
    project_dir = current_dir.parent if current_dir.name == 'src' else current_dir

    # Create necessary directories
    uploads_dir = project_dir / 'uploads'
    uploads_dir.mkdir(exist_ok=True)

    # Change to project directory
    os.chdir(project_dir)

    # Register routes
    from app.main import main
    app.register_blueprint(main)

    return app