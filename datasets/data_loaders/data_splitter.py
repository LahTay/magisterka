from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from misc.instrument_number_map import instruments_map
import pandas as pd
import numpy as np

"""
This file will split training data into validation and testing data.
There are multiple ways of splitting the files.
1. Just split randomly (not recommended)
2. Make sure every instrument is split accordingly, so that it's in proper % in testing and validation data
This is done with scikit-multilearn
3. Same thing as 2 but not every instrument but rather every set of instruments. (As an example,
if there are 3 instruments and testing data is 10% of the training data then 10% of combination (0, 1), 10% of (1, 2)
and 10% of (0, 2) would be included in the testing data.
"""


def transform_labels_to_names(labels, instruments_map):
    """Transforms list of lists of numeric labels to list of lists of string labels."""
    transformed_labels = []
    for label_set in labels:
        transformed_labels.append([instruments_map[label] for label in label_set if label in instruments_map])
    return transformed_labels

def split_data(filenames, labels, mlb, val_split=0.1, test_split=0.1, are_named=False):
    if are_named:
        labels_named = labels
    else:
        labels_named = transform_labels_to_names(labels, instruments_map)
    binary_labels = mlb.fit_transform(labels_named)
    filenames_np = np.array(filenames).reshape(-1, 1)

    X_temp, y_temp, X_test, y_test = iterative_train_test_split(filenames_np, binary_labels, test_size=test_split)

    val_split_adjusted = val_split / (1 - test_split)  # adjust validation split
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_temp, y_temp, test_size=val_split_adjusted)

    def generate_combination_counts(y_data):
        matrix = get_combination_wise_output_matrix(y_data, order=2)
        return Counter(str(combination) for row in matrix for combination in row)

        # Generating combination counts for all sets

    train_counts = generate_combination_counts(y_train)
    val_counts = generate_combination_counts(y_val)
    test_counts = generate_combination_counts(y_test)

    df = pd.DataFrame({
        'train': train_counts,
        'validation': val_counts,
        'test': test_counts
    }).T.fillna(0.0)

    print(df)

    return X_train, y_train, X_val, y_val, X_test, y_test

















    a=0
