import pandas as pd


def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def save_data(data, filepath):
    """Save data to a CSV file."""
    data.to_csv(filepath, index=False)


# def list_columns_to_exclude(data, columns_to_exclude):
#     """
#     List the columns to exclude during processing.

#     Args:
#         data (pd.DataFrame): The input DataFrame.
#         columns_to_exclude (list): List of columns to exclude.

#     Returns:
#         list: List of columns to exclude, in lower case and without spaces.
#     """
#     # Find the difference between the DataFrame's columns and the columns to exclude
#     columns_to_exclude = list(data.columns.difference(columns_to_exclude))

#     # Ensure all strings are lower case without spaces
#     columns_to_exclude = [col.lower().replace(" ", "_") for col in columns_to_exclude]

#     return columns_to_exclude
