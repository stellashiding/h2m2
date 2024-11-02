from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app():
    app = Flask(__name__)
    
    # Load environment variables from .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    # Configure CORS to allow all origins (temporary for testing)
    CORS(app, resources={
        r"/*": {
            "origins": "*",  # Allow all origins temporarily
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True
        }
    })
    
    with app.app_context():
        from . import routes
    return app
