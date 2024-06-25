import os
import sys
import json
import joblib
import pandas as pd


# Function to load the trained model
def load_model(model_path):
    return joblib.load(model_path)


# Function to make a prediction
def predict(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)
    return prediction


def main(input_json):
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    base_dir = os.path.abspath(
        os.path.join(script_dir, "../")
    )  # Project root directory

    # Load trained model
    model_path = os.path.join(base_dir, "models/lgb_model.pkl")
    model = load_model(model_path)

    # Parse input data
    input_data = json.loads(input_json)

    # Make prediction
    prediction = predict(model, input_data)

    # Output prediction
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    # Example usage: python predict.py '{"feature1": value1, "feature2": value2, ...}'
    if len(sys.argv) != 2:
        print("Usage: python predict.py '<input_json>'")
        sys.exit(1)

    input_json = sys.argv[1]
    main(input_json)
