from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__, 
                template_folder='../../templates',
                static_folder='../../static')
    CORS(app)
    
    return app
