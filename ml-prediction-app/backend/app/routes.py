from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from .models import HateSpeechDetector, DiabetesPredictor
from backend.utils.visualization import create_visualizations
import os

def create_app():
    app = Flask(__name__, 
                template_folder='../../templates',
                static_folder='../../static')
    CORS(app)
    
    # Initialize models
    hate_speech_model = HateSpeechDetector()
    diabetes_model = DiabetesPredictor()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/predict/hate-speech', methods=['POST'])
    def predict_hate_speech():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            prediction = hate_speech_model.predict(text)
            confidence = hate_speech_model.get_confidence(text)
            
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'text': text
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict/diabetes', methods=['POST'])
    def predict_diabetes():
        try:
            data = request.get_json()
            
            # Extract features
            features = [
                float(data.get('pregnancies', 0)),
                float(data.get('glucose', 0)),
                float(data.get('blood_pressure', 0)),
                float(data.get('skin_thickness', 0)),
                float(data.get('insulin', 0)),
                float(data.get('bmi', 0)),
                float(data.get('diabetes_pedigree', 0)),
                float(data.get('age', 0))
            ]
            
            prediction = diabetes_model.predict(features)
            probability = diabetes_model.get_probability(features)
            
            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/visualizations')
    def get_visualizations():
        try:
            viz_data = create_visualizations()
            return jsonify(viz_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app
