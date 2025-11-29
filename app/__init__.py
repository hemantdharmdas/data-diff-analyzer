from flask import Flask
from app.config import Config
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # No Flask-Session initialization - use built-in sessions
    
    from app.routes import main
    app.register_blueprint(main)
    
    return app
