import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


columns_to_keep = [
    "Achievements",
    "about_length",
    "n_screens",
    "n_movies",
    "n_tags",
    "n_languages",
    "has_publisher",
    "perceived_quality",
    "has_support_email",
    "has_support_url",
    "has_website",
    "success_500_threshold" "month",
    "day",
    "year" "target_success",
]


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # I will keep this for loop here for the sake of consistency.
        missing_columns = [
            col for col in [self.columns_to_drop] if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.drop(columns=[self.columns_to_drop])
        return X_transformed


class CounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        screenshots_column="screenshots",
        movies_column="movies",
        tags_column="tags",
        languages_column="supported_languages",
    ):
        self.screenshots_column = screenshots_column
        self.movies_column = movies_column
        self.tags_column = tags_column
        self.languages_column = languages_column

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [
            col
            for col in [
                self.screenshots_column,
                self.movies_column,
                self.tags_column,
                self.languages_column,
            ]
            if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        # Ensure the columns exist in the input data
        missing_columns = [
            col
            for col in [
                self.screenshots_column,
                self.movies_column,
                self.tags_column,
                self.languages_column,
            ]
            if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )

        X_transformed = X.copy()

        # Apply transformations
        X_transformed["n_screens"] = X_transformed[self.screenshots_column].apply(
            lambda x: len(set(x.split(",")))
        )
        X_transformed["n_movies"] = X_transformed[self.movies_column].apply(
            lambda x: len(set(x.split(",")))
        )
        X_transformed["n_tags"] = X_transformed[self.tags_column].apply(
            lambda x: len(set(x.split(",")))
        )

        X_transformed["supported_languages"] = X_transformed[
            self.languages_column
        ].apply(
            lambda x: x.replace("[", "").replace("]", "")
        )  # Change string format
        X_transformed["supported_languages"] = X_transformed[
            "supported_languages"
        ].apply(
            lambda x: set(x.split(","))
        )  # Transform into a set
        X_transformed["n_languages"] = X_transformed["supported_languages"].apply(
            lambda n: len(n)
        )  # Count the number of members in a set

        return X_transformed


class CreateBinaryColumns(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        developers_column="developers",
        publishers_column="publishers",
        positive_column="positive",
        negative_column="negative",
        support_email_column="support_email",
        support_url_column="support_url",
        website_column="website",
    ):
        self.developers_column = developers_column
        self.publishers_column = publishers_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.support_email_column = support_email_column
        self.support_url_column = support_url_column
        self.website_column = website_column

    def fit(self, X, y=None):
        # I will keep this for loop here for the sake of consistency.
        missing_columns = [
            col
            for col in [
                self.developers_column,
                self.publishers_column,
                self.positive_column,
                self.negative_column,
                self.support_email_column,
                self.support_url_column,
                self.website_column,
            ]
            if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed["has_publisher"] = (
            X_transformed[self.publishers_column]
            != X_transformed[self.developers_column]
        )
        X_transformed["total_reviews"] = (
            X_transformed[self.positive_column] + X_transformed[self.negative_column]
        )

        X_transformed["perceived_quality"] = X_transformed[self.positive_column] / (
            X_transformed[self.negative_column] + 1
        )
        X_transformed["has_support_email"] = ~X_transformed[
            self.support_email_column
        ].isna()
        X_transformed["has_support_url"] = ~X_transformed[
            self.support_url_column
        ].isna()
        X_transformed["has_website"] = ~X_transformed[self.website_column].isna()
        return X_transformed


class CreateTargetColumns(BaseEstimator, TransformerMixin):
    def __init__(self, total_reviews_column="total_reviews", threshold=500):
        self.total_reviews_column = total_reviews_column
        self.threshold = threshold

    def fit(self, X, y=None):
        # I will keep this for loop here for the sake of consistency.
        missing_columns = [
            col for col in [self.total_reviews_column] if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed["target_success"] = (
            X_transformed[self.total_reviews_column] > self.threshold
        )

        return X_transformed


class DateColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="release_date"):
        self.date_column = date_column

    def fit(self, X):
        # Check if the date column exists in the input data
        if self.date_column not in X.columns:
            raise ValueError(
                f"The specified date column '{self.date_column}' does not exist in the DataFrame."
            )
        return self

    def transform(self, X):
        # Check if the date column exists in the input data
        if self.date_column not in X.columns:
            raise ValueError(
                f"The specified date column '{self.date_column}' does not exist in the DataFrame."
            )

        X_transformed = X.copy()

        # Convert the date column to datetime format
        X_transformed[self.date_column] = pd.to_datetime(
            X_transformed[self.date_column], format="mixed"
        )

        # Extract month, day, and year into separate datetime columns
        X_transformed["month"] = X_transformed[self.date_column].dt.month
        X_transformed["day"] = X_transformed[self.date_column].dt.day
        X_transformed["year"] = X_transformed[self.date_column].dt.year

        return X_transformed


class DeleteNaNRows(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        screenshots_column="screenshots",
        movies_column="movies",
        tags_column="tags",
    ):
        self.screenshots_column = screenshots_column
        self.movies_column = movies_column
        self.tags_column = tags_column

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [
            col
            for col in [self.screenshots_column, self.movies_column, self.tags_column]
            if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.dropna(subset=self.screenshots_column)
        X_transformed = X_transformed.dropna(subset=self.movies_column)
        X_transformed = X_transformed.dropna(subset=self.tags_column)

        return X_transformed


class YearThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selected_year=2023, year_column="year"):
        self.selected_year = selected_year
        self.year_column = year_column

    def fit(self, X, y=None):
        # I will keep this for loop here for the sake of consistency.
        missing_columns = [col for col in [self.year_column] if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed[
            X_transformed[self.year_column] >= self.selected_year
        ]
        return X_transformed


class IndieTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        metacritic_score_column="metacritic_score",
        metacritic_score=0,
        total_reviews_column="total_reviews",
        total_reviews=10,
    ):
        self.metacritic_score_column = metacritic_score_column
        self.total_reviews_column = total_reviews_column
        self.total_reviews = total_reviews
        self.metacritic_score = metacritic_score

    def fit(self, X, y=None):
        # I will keep this for loop here for the sake of consistency.
        missing_columns = [
            col
            for col in [self.total_reviews_column, self.metacritic_score_column]
            if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed[
            X_transformed[self.metacritic_score_column] == self.metacritic_score
        ]
        X_transformed = X_transformed[
            X_transformed[self.total_reviews_column] >= self.total_reviews
        ]

        return X_transformed
