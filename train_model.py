"""
نموذج التنبؤ بالطقس لمكة المكرمة
يستخدم Machine Learning للتنبؤ بدرجة الحرارة والرطوبة وسرعة الرياح
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_data(json_path):
    """تحميل البيانات من ملف JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"تم تحميل {len(df)} سجل من البيانات")
    print(f"الأعمدة: {df.columns.tolist()}")
    return df

def prepare_features(df):
    """تحضير الميزات للتدريب"""
    # تحويل التاريخ - التحقق من اسم العمود
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # استخراج ميزات الوقت
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # ميزات دورية للساعة واليوم
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def create_sequences(df, lookback=24, forecast_hours=48):
    """
    إنشاء تسلسلات للتدريب
    lookback: عدد الساعات السابقة للنظر فيها
    forecast_hours: عدد ساعات التنبؤ (48 ساعة = يومين)
    """
    features = ['temperature', 'humidity', 'wind_speed', 
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos']
    
    targets = ['temperature', 'humidity', 'wind_speed']
    
    X = []
    y = []
    
    for i in range(lookback, len(df) - forecast_hours):
        # الميزات: آخر lookback ساعة
        x_seq = df[features].iloc[i-lookback:i].values.flatten()
        
        # إضافة معلومات الوقت للتنبؤ
        future_hours = df[['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                          'day_of_year_sin', 'day_of_year_cos']].iloc[i:i+forecast_hours].values.flatten()
        
        x_combined = np.concatenate([x_seq, future_hours])
        X.append(x_combined)
        
        # الهدف: التنبؤ بـ 48 ساعة قادمة
        y_seq = df[targets].iloc[i:i+forecast_hours].values.flatten()
        y.append(y_seq)
    
    return np.array(X), np.array(y)

def train_model(json_path):
    """تدريب النموذج"""
    print("=" * 50)
    print("بدء تدريب نموذج التنبؤ بالطقس")
    print("=" * 50)
    
    # تحميل وتحضير البيانات
    df = load_data(json_path)
    df = prepare_features(df)
    
    # فرز البيانات حسب التاريخ
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"\nنطاق التواريخ: من {df['datetime'].min()} إلى {df['datetime'].max()}")
    print(f"إحصائيات درجة الحرارة: min={df['temperature'].min():.1f}, max={df['temperature'].max():.1f}, mean={df['temperature'].mean():.1f}")
    
    # إنشاء التسلسلات
    lookback = 24  # 24 ساعة سابقة
    forecast_hours = 48  # التنبؤ بـ 48 ساعة (يومين)
    
    print(f"\nإنشاء تسلسلات التدريب...")
    print(f"النظر في آخر {lookback} ساعة للتنبؤ بـ {forecast_hours} ساعة قادمة")
    
    X, y = create_sequences(df, lookback, forecast_hours)
    print(f"شكل X: {X.shape}, شكل y: {y.shape}")
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nبيانات التدريب: {len(X_train)}, بيانات الاختبار: {len(X_test)}")
    
    # تطبيع البيانات
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # تدريب النموذج
    print("\nتدريب النموذج... (سريع)")
    
    # استخدام Random Forest للسرعة
    base_model = RandomForestRegressor(
        n_estimators=20,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # تقييم النموذج
    print("\nتقييم النموذج...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # حساب الدقة لكل متغير
    for i, var in enumerate(['temperature', 'humidity', 'wind_speed']):
        var_indices = list(range(i, forecast_hours * 3, 3))
        y_true_var = y_test[:, var_indices]
        y_pred_var = y_pred[:, var_indices]
        
        mae = mean_absolute_error(y_true_var, y_pred_var)
        rmse = np.sqrt(mean_squared_error(y_true_var, y_pred_var))
        r2 = r2_score(y_true_var, y_pred_var)
        
        print(f"\n{var}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.3f}")
    
    # حفظ النموذج
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/weather_model.pkl')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    # حفظ معلومات إضافية
    model_info = {
        'lookback': lookback,
        'forecast_hours': forecast_hours,
        'features': ['temperature', 'humidity', 'wind_speed', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'day_of_year_sin', 'day_of_year_cos'],
        'targets': ['temperature', 'humidity', 'wind_speed'],
        'data_stats': {
            'temp_min': float(df['temperature'].min()),
            'temp_max': float(df['temperature'].max()),
            'temp_mean': float(df['temperature'].mean()),
            'humidity_min': float(df['humidity'].min()),
            'humidity_max': float(df['humidity'].max()),
            'wind_min': float(df['wind_speed'].min()),
            'wind_max': float(df['wind_speed'].max())
        }
    }
    
    with open('models/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    # حفظ آخر 24 ساعة من البيانات للتنبؤ
    last_data = df.tail(lookback).to_json(orient='records', date_format='iso')
    with open('models/last_data.json', 'w', encoding='utf-8') as f:
        f.write(last_data)
    
    print("\n" + "=" * 50)
    print("تم حفظ النموذج بنجاح!")
    print("=" * 50)
    
    return model, scaler_X, scaler_y, model_info

if __name__ == "__main__":
    # مسار ملف البيانات
    json_path = "mecca_weather_hourly.json"
    
    if os.path.exists(json_path):
        train_model(json_path)
    else:
        print(f"خطأ: الملف {json_path} غير موجود!")
        print("الرجاء نسخ ملف mecca_weather_hourly.json إلى هذا المجلد")
