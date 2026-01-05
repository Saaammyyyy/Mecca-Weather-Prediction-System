"""
Fast training - for testing only
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 40)
print("Fast Training for Testing")
print("=" * 40)

# Load data
with open('mecca_weather_hourly.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Loaded {len(df)} records")

# Take a small sample only (last 2000 records)
df = df.tail(2000).reset_index(drop=True)

# Convert date
df['datetime'] = pd.to_datetime(df['time'])
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

lookback = 24
forecast_hours = 48

print("Creating data...")

X = []
y = []

# Take few samples for speed
for i in range(lookback, len(df) - forecast_hours, 10):  # every 10 hours
    x_seq = df[features].iloc[i-lookback:i].values.flatten()
    future = df[['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                 'day_of_year_sin', 'day_of_year_cos']].iloc[i:i+forecast_hours].values.flatten()
    X.append(np.concatenate([x_seq, future]))
    y.append(df[['temperature', 'humidity', 'wind_speed']].iloc[i:i+forecast_hours].values.flatten())

X = np.array(X)
y = np.array(y)

print(f"X: {X.shape}, y: {y.shape}")

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train simple and fast model
print("Training model...")
model = Ridge(alpha=1.0)
model.fit(X_scaled, y_scaled)

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/weather_model.pkl')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# Save model info
model_info = {
    'lookback': lookback,
    'forecast_hours': forecast_hours,
    'features': features,
    'targets': ['temperature', 'humidity', 'wind_speed'],
    'data_stats': {
        'temp_min': float(df['temperature'].min()),
        'temp_max': float(df['temperature'].max()),
        'temp_mean': float(df['temperature'].mean())
    }
}

with open('models/model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

# Save last 24 hours
last_24 = df.tail(24)[['time', 'temperature', 'humidity', 'wind_speed']].to_dict('records')
with open('models/last_data.json', 'w', encoding='utf-8') as f:
    json.dump(last_24, f, ensure_ascii=False)

print("=" * 40)
print("Training and saving completed successfully!")
print("=" * 40)
