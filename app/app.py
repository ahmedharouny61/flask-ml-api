import logging
import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Get path to the current directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler safely using absolute paths
try:
    model = joblib.load(os.path.join(BASE_DIR, "optimized_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    logging.info("✅ Model and scaler loaded successfully.")
    FEATURES = list(model.feature_names_in_)
    logging.info(f"✅ Model expects features: {FEATURES}")
except Exception as e:
    logging.error(f"❌ Failed to load model or scaler: {e}")
    raise e

# Exercise mapping
exercise_mapping = {
    0: "Lunges",
    1: "Squats",
    2: "Calf Raises",
    3: "Hamstring Curls",
    4: "Quadriceps Stretch",
    5: "Heel Slides",
    6: "Leg Press",
    7: "Standing Leg Lifts",
    8: "Straight Leg Raise",
    9: "Seated Exercises",
    10: "Side Lying Leg Lift",
    11: "Step Up"
}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        logging.info(f"📩 Received data: {data}")

        # Initialize DataFrame with all features set to 0
        input_df = pd.DataFrame(columns=FEATURES, data=[[0]*len(FEATURES)])

        # Process numerical features
        numerical_fields = ['Age', 'Pain_Level', 'Morning_Stiffness__min_', 'Functional_Score']
        for field in numerical_fields:
            if field in data:
                input_df[field] = data[field]

        # Process categorical features
        categorical_mapping = {
            'Gender': {
                'Male': 'Gender_M',
                'Female': None  # Base case
            },
            'Difficulty_Walking': {
                'Mild': 'Difficulty_Walking_Mild',
                'Moderate': 'Difficulty_Walking_Moderate',
                'Severe': 'Difficulty_Walking_Severe'
            },
            'Swelling': {
                'Yes': 'Swelling_Yes',
                'No': None  # Base case
            },
            'Assistive_Device': {
                'None': 'Assistive_Device_None',
                'Walker': 'Assistive_Device_Walker',
                'Wheelchair': 'Assistive_Device_Wheelchair'
            },
            'Xray_Findings': {
                'Normal': 'Xray_Findings_Normal',
                'Mild Osteoarthritis': 'Xray_Findings_Mild_OA',
                'Moderate Osteoarthritis': 'Xray_Findings_Moderate_OA',
                'Severe Osteoarthritis': 'Xray_Findings_Severe_OA'
            }
        }

        for field, mapping in categorical_mapping.items():
            if field in data:
                value = data.get(field)
                column = mapping.get(value)
                if column:
                    input_df[column] = 1

        # Ensure correct feature order
        input_data = input_df[FEATURES]
        logging.info(f"📊 Final input data:\n{input_data}")

        # Scale input
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)[0]
        exercise_name = exercise_mapping.get(int(prediction), "Unknown Exercise")
        logging.info(f"🏋️ Recommended exercise: {exercise_name}")

        return jsonify({"exercise_name": exercise_name})

    except KeyError as ke:
        error_msg = f"Missing required field: {str(ke)}"
        logging.error(f"❗ {error_msg}")
        return jsonify({"error": error_msg}), 400

    except ValueError as ve:
        error_msg = f"Invalid data format: {str(ve)}"
        logging.error(f"❗ {error_msg}")
        return jsonify({"error": error_msg}), 400

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logging.error(f"❗ {error_msg}")
        return jsonify({"error": "Internal server error"}), 500
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask ML API is running. Use /recommend endpoint for predictions."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
