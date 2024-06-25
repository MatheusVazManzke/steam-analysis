from transformer_classes import (
    DropColumnsTransformer,
    DataframeFilterTransformer,
)

from sklearn.pipeline import Pipeline
import os
import sys
from utils import load_data, save_data


def create_transformation_pipeline():
    return Pipeline(
        [
            (
                "filter_dataframe",
                DataframeFilterTransformer(
                    filter_dict={
                        "metacritic_score": 0,
                        "total_reviews": 0,
                        "year": 2020,
                        "genres": "Indie",
                    }
                ),
            ),
            (
                "drop_columns",
                DropColumnsTransformer(
                    [
                        "Achievements",
                        "n_about_the_game",
                        "n_screenshots",
                        "n_movies",
                        "n_tags",
                        "n_supported_languages",
                        "has_publishers",
                        "has_support_email",
                        "has_support_url",
                        "has_website",
                    ]
                ),
            ),
        ]
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

    train_dataset_pipeline = create_transformation_pipeline()

    # Transform data
    transformed_data = train_dataset_pipeline.fit_transform(data)

    # Save transformed data
    save_data(
        transformed_data,
        os.path.abspath(os.path.join(base_dir, "data/processed/train_dataset.csv")),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <raw_data_file>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    main(raw_data_file)
