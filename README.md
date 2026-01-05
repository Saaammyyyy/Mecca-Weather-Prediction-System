Mecca Weather Prediction System

A Machine Learning–Based Hourly Weather Forecasting Application

1. Introduction

This project presents a machine learning–based system for predicting weather conditions in Mecca city. The system focuses on short-term, hourly forecasting of key meteorological variables, including temperature, humidity, and wind speed. By leveraging historical weather data and supervised learning techniques, the project demonstrates an end-to-end predictive pipeline suitable for academic experimentation and evaluation.

The system integrates data preprocessing, model training, and a locally hosted web application, allowing users to request weather predictions for a selected date. The application is designed for local execution only and is not deployed on a public server.

2. System Features

The main features of the system include:

Hourly prediction of temperature, humidity, and wind speed

Forecast horizon of 48 hours (two days)

Machine learning–based regression model for improved accuracy

Web-based user interface for interacting with the prediction system

Local server execution for controlled and reproducible experimentation

3. Dataset Description

The system is trained using a historical hourly weather dataset for Mecca city, stored in JSON format (mecca_weather_hourly.json). Each data record includes:

Temperature (°C)

Humidity (%)

Wind speed (m/s)

Timestamp information (date and hour)

The dataset provides the historical context required for learning temporal weather patterns. For prediction, the system uses a lookback window of 24 hours to generate forecasts for the next 48 hours.

4. Machine Learning Methodology
4.1 Feature Engineering

In addition to raw weather measurements, time-based features are engineered to capture daily and seasonal variations. These include:

Hour of day (encoded using sine and cosine functions)

Month of year (cyclical encoding)

Day of year (cyclical encoding)

Cyclical encoding is applied to avoid discontinuities in time representation. All features and prediction targets are normalized using scaling techniques to improve model stability.

4.2 Learning Algorithm

The system employs a Gradient Boosting Regressor, a tree-based ensemble learning algorithm suitable for nonlinear regression problems. The model is trained to predict three output variables simultaneously:

Temperature

Humidity

Wind speed

This approach allows the model to learn complex relationships between historical weather conditions and future observations.

5. Model Training Process

Model training is performed offline using the train_model.py script and follows these steps:

Load and preprocess the historical weather dataset

Generate input–output sequences using a sliding window approach

Apply feature scaling to both inputs and outputs

Train the Gradient Boosting regression model

Save the trained model and preprocessing artifacts

The trained model and related files are stored locally and reused during prediction.

6. System Architecture and Workflow

The system consists of two main layers:

6.1 Backend (Local Server)

Implemented using Flask

Responsible for loading the trained model and scalers

Handles prediction requests from the frontend

Returns hourly weather forecasts for 48 hours

6.2 Frontend (Web Interface)

Implemented using HTML, CSS, and JavaScript

Allows the user to select a date

Displays the predicted weather results in a structured format

Prediction Workflow

The user selects a target date through the web interface

The frontend sends a request to the backend API

The backend loads the trained model and recent data

Predictions for the next 48 hours are generated

Results are returned and displayed to the user

7. Local Server Requirement

This application is designed to run exclusively on a local machine. The Flask backend must be running locally for the system to function correctly.

Key points:

The backend is accessed via http://localhost:5000

No cloud or external deployment is used

If the local server is not running, predictions cannot be generated

This design ensures reproducibility, security, and suitability for academic use rather than public deployment.

8. Installation and Execution
8.1 Install Dependencies
pip install -r requirements.txt

8.2 Prepare the Dataset

Copy the mecca_weather_hourly.json file into the project directory.

8.3 Train the Model
python train_model.py

8.4 Run the Application
python app.py

8.5 Access the System

Open a web browser and navigate to:
http://localhost:5000

9. Project Structure
mecca_weather_forecast/
├── app.py                   # Flask backend application
├── train_model.py           # Model training script
├── index.html               # Web user interface
├── mecca_weather_hourly.json # Historical dataset
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── models/                  # Trained model artifacts
    ├── weather_model.pkl
    ├── scaler_X.pkl
    ├── scaler_y.pkl
    ├── model_info.json
    └── last_data.json

10. Limitations

The system relies entirely on historical data quality

Forecasting is limited to a 48-hour horizon

The model does not incorporate real-time data updates

The application is restricted to local execution

11. Conclusion

This project demonstrates a complete and reproducible machine learning pipeline for short-term weather prediction in Mecca. By combining historical data, feature engineering, and ensemble regression techniques within a locally hosted web application, the system provides accurate and interpretable hourly forecasts. The project serves as a strong academic example of applied machine learning for time-series forecasting.

## Demonstration

Below are screenshots of the Mecca Weather Prediction System in action:

<p align="center">
  <img src="https://github.com/Saaammyyyy/Mecca-Weather-Prediction-System/blob/cb278b741afbef04bd501e9b62a781f6e12d5843/demo1.png" alt="Demo 1" width="800" />
</p>
<p align="center">
  <img src="https://github.com/Saaammyyyy/Mecca-Weather-Prediction-System/blob/d00869b3535895f48f824fc61b3092b4fee0f376/demo2.png" alt="Demo 2" width="800" />
</p>
<p align="center">
  <img src="[docs/images/demo3.png](https://github.com/Saaammyyyy/Mecca-Weather-Prediction-System/blob/d00869b3535895f48f824fc61b3092b4fee0f376/demo3.png)" alt="Demo 3" width="800" />
</p>

These images show the user interface, prediction results for two days, and the overall look and feel of the deployed application.



