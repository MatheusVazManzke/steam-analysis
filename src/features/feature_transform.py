from transformer_classes import (
    LowerCaseColumnsTransformer,
    BinaryColumnsTransformer,
    FillNATransformers,
    DropNATransformers,
    CounterColumnsTransformer,
    DateColumnsTransformer,
    ArithmeticColumnTransformer,
    DropColumnsTransformer,
    DataframeFilterTransformer,
    CreateTargetColumns,
)

from sklearn.pipeline import Pipeline
import os
import sys
from utils import load_data, save_data


def create_transformation_pipeline():
    """Create the transformation pipeline with the specified steps."""
    return Pipeline(
        steps=[
            ("lowercase_columns", LowerCaseColumnsTransformer()),
            (
                "create_binary_columns",
                BinaryColumnsTransformer(
                    columns=["publishers", "support_email", "support_url", "website"]
                ),
            ),
            ("fill_NA_columns", FillNATransformers(columns=["tags", "about_the_game"])),
            (
                "drop_na_columns",
                DropNATransformers(columns=["screenshots", "movies", "genres"]),
            ),
            (
                "create_counter_columns",
                CounterColumnsTransformer(
                    [
                        "tags",
                        "screenshots",
                        "movies",
                        "supported_languages",
                        "about_the_game",
                    ]
                ),
            ),
            ("create_date_columns", DateColumnsTransformer("release_date")),
            (
                "create_total_reviews_columns",
                ArithmeticColumnTransformer(
                    "positive", "negative", "add", "total_reviews"
                ),
            ),
            (
                "create_perceived_quality_columns",
                ArithmeticColumnTransformer(
                    "positive", "negative", "truediv", "perceived_quality"
                ),
            ),
            ("create_target_columns", CreateTargetColumns(threshold=500)),
        ]
    )


def list_columns_to_exclude(data, columns_to_exlude):
    columns_to_exlude = list(data[data.columns.difference(columns_to_exlude)].columns)
    columns_to_exlude = [col.lower().replace(" ", "_") for col in columns_to_exlude]
    return columns_to_exlude


def main(raw_data_file):
    """Main function to execute the data transformation pipeline."""
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath("feature_transform.ipynb"))

    # Navigate to the project root directory (assuming it is two levels up)
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Construct the absolute path to the data file
    data_file_path = os.path.join(base_dir, f"data/raw/{raw_data_file}")

    data = load_data(data_file_path)

    transformation_pipeline = create_transformation_pipeline()

    # Transform data
    transformed_data = transformation_pipeline.fit_transform(data)

    # Save transformed data
    save_data(
        transformed_data,
        os.path.abspath(os.path.join(base_dir, "data/processed/transformed_games.csv")),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <raw_data_file>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    main(raw_data_file)
