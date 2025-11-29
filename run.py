from app import create_app
from app.config import Config
import os

app = create_app()
Config.init_app(app)

app.config['DEBUG'] = True

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
