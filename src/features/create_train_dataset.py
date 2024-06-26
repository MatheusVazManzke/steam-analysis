from transformer_classes import (
    DropColumnsTransformer,
    DataframeFilterTransformer,
)

from sklearn.pipeline import Pipeline
import os
import sys
from utils import load_data, save_data


columns_to_keep = [
    "n_supported_languages",
    "n_tags",
    "achievements",
    "n_about_the_game",
    "n_screenshots",
    "has_support_url",
    "target_success",
]

dataframe_filter = DataframeFilterTransformer(
    filter_dict={
        "metacritic_score": 0,
        "total_reviews": 0,
        "year": 2020,
        "genres": "Indie",
    }
)


def main(raw_data_file):
    """Main function to execute the data transformation pipeline."""
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath("create_train_dataset.py"))

    # Navigate to the project root directory (assuming it is two levels up)
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Construct the absolute path to the data file
    data_file_path = os.path.join(base_dir, f"data/processed/{raw_data_file}")

    data = load_data(data_file_path)

    # Transform data
    filtered_data = dataframe_filter.fit_transform(data)
    filtered_data = filtered_data[columns_to_keep]

    # Save transformed data
    save_data(
        filtered_data,
        os.path.abspath(os.path.join(base_dir, "data/processed/train_dataset.csv")),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <raw_data_file>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    main(raw_data_file)
