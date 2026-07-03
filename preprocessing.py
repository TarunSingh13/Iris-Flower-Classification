"""
Preprocessing Module
--------------------
Handles train-test split and feature scaling.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE, RANDOM_STATE


def split_data(df):
    """
    Split dataset into features and target,
    then divide into training and testing sets.
    """

    # Features (Input)
    X = df.drop("species", axis=1)

    # Target (Output)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize the feature values.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler