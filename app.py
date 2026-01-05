"""
Flask Application for Weather Prediction in Mecca
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta

# Get the current directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='.')
CORS(app)

# Load model and tools
model = None
scaler_X = None
scaler_y = None
model_info = None
last_data = None

def load_model():
    """Load the trained model"""
    global model, scaler_X, scaler_y, model_info, last_data
    
    try:
        models_dir = os.path.join(BASE_DIR, 'models')
        model = joblib.load(os.path.join(models_dir, 'weather_model.pkl'))
        scaler_X = joblib.load(os.path.join(models_dir, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(models_dir, 'scaler_y.pkl'))
        
        with open(os.path.join(models_dir, 'model_info.json'), 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        with open(os.path.join(models_dir, 'last_data.json'), 'r', encoding='utf-8') as f:
            last_data = json.load(f)
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def prepare_input(date_str, historical_data):
    """Prepare inputs for prediction"""
    df = pd.DataFrame(historical_data)
    
    # Check date column name
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    features = ['temperature', 'humidity', 'wind_speed', 
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos']
    
    # Prepare last 24 hours data
    x_seq = df[features].values.flatten()
    
    # Prepare time info for prediction (next 48 hours)
    start_date = pd.to_datetime(date_str)
    future_dates = [start_date + timedelta(hours=i) for i in range(48)]
    
    future_features = []
    for dt in future_dates:
        hour = dt.hour
        month = dt.month
        day_of_year = dt.timetuple().tm_yday
        
        future_features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365)
        ])
    
    x_combined = np.concatenate([x_seq, np.array(future_features)])
    return x_combined.reshape(1, -1), future_dates

@app.route('/')
def index():
    """Home page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Weather prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.json
        date_str = data.get('date')
        
        if not date_str:
            return jsonify({'error': 'Please enter a date'}), 400
        
        # Prepare inputs
        X, future_dates = prepare_input(date_str, last_data)
        
        # Predict
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Format results
        predictions = []
        for i in range(48):
            temp_idx = i * 3
            hum_idx = i * 3 + 1
            wind_idx = i * 3 + 2
            
            predictions.append({
                'datetime': future_dates[i].strftime('%Y-%m-%d %H:%M'),
                'hour': future_dates[i].strftime('%H:00'),
                'date': future_dates[i].strftime('%Y-%m-%d'),
                'day_name': get_day_name(future_dates[i].weekday()),
                'temperature': round(float(y_pred[0][temp_idx]), 1),
                'humidity': round(float(y_pred[0][hum_idx]), 1),
                'wind_speed': round(float(y_pred[0][wind_idx]), 1)
            })
        
        # Split by day
        day1 = [p for p in predictions if p['date'] == future_dates[0].strftime('%Y-%m-%d')]
        day2 = [p for p in predictions if p['date'] == future_dates[24].strftime('%Y-%m-%d') if len(predictions) > 24]
        
        # If no clear split, divide 24-24
        if not day2:
            day1 = predictions[:24]
            day2 = predictions[24:]
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'day1': {
                'date': future_dates[0].strftime('%Y-%m-%d'),
                'day_name': get_day_name(future_dates[0].weekday()),
                'data': day1[:24] if len(day1) >= 24 else day1,
                'summary': calculate_summary(day1[:24] if len(day1) >= 24 else day1)
            },
            'day2': {
                'date': future_dates[24].strftime('%Y-%m-%d') if len(future_dates) > 24 else '',
                'day_name': get_day_name(future_dates[24].weekday()) if len(future_dates) > 24 else '',
                'data': day2[:24] if len(day2) >= 24 else day2,
                'summary': calculate_summary(day2[:24] if len(day2) >= 24 else day2)
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

def get_day_name(weekday):
    """Get the day name in English"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[weekday]

def calculate_summary(data):
    """Calculate day summary"""
    if not data:
        return {}
    
    temps = [d['temperature'] for d in data]
    humidities = [d['humidity'] for d in data]
    winds = [d['wind_speed'] for d in data]
    
    return {
        'temp_min': round(min(temps), 1),
        'temp_max': round(max(temps), 1),
        'temp_avg': round(sum(temps) / len(temps), 1),
        'humidity_avg': round(sum(humidities) / len(humidities), 1),
        'wind_avg': round(sum(winds) / len(winds), 1)
    }

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Model information"""
    if model_info:
        return jsonify({
            'loaded': True,
            'info': model_info
        })
    return jsonify({'loaded': False})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model at startup
    if load_model():
        print("Model is ready!")
    else:
        print("Warning: Model not loaded. Please run train_model.py first")
    
    print("\nServer running at: http://localhost:5000")
    app.run(debug=True, port=5000)
