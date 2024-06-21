import os
import sys
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    base_dir = os.path.abspath(
        os.path.join(script_dir, "../../")
    )  # Project root directory
    data_file_path = os.path.join(
        base_dir, "data/processed/transformed_games.csv"
    )  # Path to your processed data

    # Load processed data
    processed_data = load_data(data_file_path)

    # Parameters for LightGBM
    params = {
        "num_leaves": 218,
        "learning_rate": 0.012299865257179486,
        "feature_fraction": 0.7205287309983145,
        "reg_alpha": 5.534157704462869e-06,
        "reg_lambda": 0.008453734323058607,
        "bagging_fraction": 0.7417796313309262,
        "min_child_samples": 50,
        "is_unbalance": True,
        "random_state": 42,
    }

    # Prepare training data and target
    train_data = processed_data.drop(columns=["target_success", "total_reviews"])
    train_target = processed_data["target_success"]

    # Initialize LightGBM classifier
    lgb = LGBMClassifier(**params)

    # Fit the model
    lgb.fit(train_data, train_target)

    # Optionally, calibrate the model
    y_pred_prob = lgb.predict_proba(train_data)[:, 1]
    threshold = 0.5721598510577628
    train_target_adjusted = (y_pred_prob >= threshold).astype(int)
    platt_calibrated = CalibratedClassifierCV(lgb, method="sigmoid", cv="prefit")
    platt_calibrated.fit(train_data, train_target_adjusted)

    # Save the trained model using joblib
    model_output_path = os.path.join(base_dir, "src/models/lgb_model.pkl")
    joblib.dump(platt_calibrated, model_output_path)

    print(f"Model trained and saved successfully at: {model_output_path}")


if __name__ == "__main__":
    main()
