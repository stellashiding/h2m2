# backend/run.py
from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Get host and port from environment variables or default to 127.0.0.1:5000
    host = os.getenv('FLASK_RUN_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    app.run(host=host, port=port, debug=debug)
