import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SteamDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.dropna(subset=["Screenshots"])
        X_transformed = X_transformed.dropna(subset=["Movies"])
        X_transformed = X_transformed.dropna(subset=["Genres"])
        X_transformed["Tags"] = X_transformed["Tags"].fillna("empty")
        X_transformed["About the game"] = X_transformed["About the game"].fillna(
            "empty"
        )

        X_transformed["about_length"] = X_transformed["About the game"].apply(
            lambda x: len(x)
        )

        X_transformed["has_publisher"] = (
            X_transformed["Developers"] != X_transformed["Publishers"]
        )
        X_transformed["perceived_quality"] = X_transformed["Positive"] / (
            X_transformed["Negative"] + 1
        )
        X_transformed["no_user_reactions"] = (
            X_transformed["Positive"] + X_transformed["Negative"]
        ) == 0
        X_transformed["total_reviews"] = (
            X_transformed["Positive"] + X_transformed["Negative"]
        )
        X_transformed["Supported languages"] = X_transformed[
            "Supported languages"
        ].apply(lambda x: set(x.split(",")))
        X_transformed["n_screens"] = X_transformed["Screenshots"].apply(
            lambda x: len(set(x.split(",")))
        )
        X_transformed["n_movies"] = X_transformed["Movies"].apply(
            lambda x: len(set(x.split(",")))
        )
        X_transformed["n_tags"] = X_transformed["Tags"].apply(
            lambda x: len(set(x.split(",")))
        )
        X_transformed["n_languages"] = X_transformed["Supported languages"].apply(
            lambda n: len(n)
        )
        X_transformed["has_support_email"] = ~X_transformed["Support email"].isna()
        X_transformed["has_support_url"] = ~X_transformed["Support url"].isna()
        X_transformed["has_website"] = ~X_transformed["Website"].isna()

        # Convert the 'date_string' column to datetime format
        X_transformed["Release date"] = pd.to_datetime(
            X_transformed["Release date"], format="mixed"
        )

        # Extract month, day, and year into separate datetime columns
        X_transformed["month"] = X_transformed["Release date"].dt.month
        X_transformed["day"] = X_transformed["Release date"].dt.day
        X_transformed["year"] = X_transformed["Release date"].dt.year

        # Create target variables
        X_transformed["success_1000_threshold"] = (
            X_transformed["total_reviews"] > 1000
        ).astype(int)
        X_transformed["success_500_threshold"] = (
            X_transformed["total_reviews"] > 500
        ).astype(int)
        return X_transformed
