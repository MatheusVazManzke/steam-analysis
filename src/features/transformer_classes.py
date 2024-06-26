import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import operator


class RandomNoiseColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        np.random.seed(self.random_seed)
        X_transformed["random_noise"] = np.random.randn(len(X_transformed))
        return X_transformed


class StringLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[f"n_{self.column}"] = X_transformed[self.column].apply(
            lambda x: len(x)
        )
        return X_transformed


class LowerCaseColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.columns = [
            col.lower().replace(" ", "_") for col in X_transformed.columns
        ]
        return X_transformed


class BinaryColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Count the number of unique items in each specified column
        X_transformed = X.copy()

        for col in self.columns:
            X_transformed[f"has_{col}"] = ~X_transformed[col].isna()
        return X_transformed


class FillNATransformers(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Count the number of unique items in each specified column
        X_transformed[self.columns] = X_transformed[self.columns].fillna("empty")

        return X_transformed


class DropNATransformers(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Count the number of unique items in each specified column
        X_transformed = X_transformed.dropna(subset=self.columns)

        return X_transformed


class CounterColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # Check if the specified columns exist in the input data
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"The following specified columns do not exist in the DataFrame: {missing_columns}"
            )
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Count the number of unique items in each specified column
        for col in self.columns:
            X_transformed[col] = X_transformed[col].apply(
                lambda x: x.replace("[", "").replace("]", "")
            )
            X_transformed[f"n_{col}"] = X_transformed[col].apply(
                lambda x: len(set(x.split(",")))
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


class ArithmeticColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column1, column2, operation, new_column_name):
        self.column1 = column1
        self.column2 = column2
        self.operation = operation
        self.new_column_name = new_column_name

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Count the number of unique items in each specified column
        # Perform the arithmetic operation
        op_func = getattr(operator, self.operation)
        X_transformed[self.new_column_name] = op_func(
            X_transformed[self.column1], (X_transformed[self.column2] + 1)
        )

        return X_transformed


class CreateTargetColumns(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        total_reviews_column="total_reviews",
        threshold=500,
        target_column="target_success",
    ):
        self.total_reviews_column = total_reviews_column
        self.threshold = threshold
        self.target_column = target_column

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
        X_transformed[self.target_column] = (
            X_transformed[self.total_reviews_column] > self.threshold
        )
        return X_transformed


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.columns_to_exlude = list(X[X.columns.difference(self.columns)].columns)
        self.columns_to_exlude = [col.lower().replace(" ", "_") for col in self.columns]
        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed = X_transformed.drop(columns=self.columns_to_exlude)

        return X_transformed


class DataframeFilterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        filter_dict={
            "metacritic_score": 0,
            "total_reviews": 10,
            "year": 2023,
            "genres": "indie",
        },
    ):
        self.filter_dict = filter_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed[
            X_transformed[list(self.filter_dict.keys())[3]]
            .str.contains(list(self.filter_dict.values())[3])
            .fillna(False)
        ]

        X_transformed = X_transformed[
            X_transformed[list(self.filter_dict.keys())[0]]
            == list(self.filter_dict.values())[0]
        ]
        X_transformed = X_transformed[
            X_transformed[list(self.filter_dict.keys())[1]]
            > list(self.filter_dict.values())[1]
        ]
        X_transformed = X_transformed[
            X_transformed[list(self.filter_dict.keys())[2]]
            >= list(self.filter_dict.values())[2]
        ]

        return X_transformed
