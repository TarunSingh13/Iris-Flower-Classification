import pandas as pd
from sklearn.datasets import load_iris


def load_dataset():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target

    return iris, df


def explore_dataset(df, iris):
    """
    Display a concise summary of the dataset.
    """

    print("\n" + "=" * 65)
    print("🌸 IRIS DATASET SUMMARY")
    print("=" * 65)

    print(f"{'Total Samples':<25}: {df.shape[0]}")
    print(f"{'Total Features':<25}: {df.shape[1] - 1}")
    print(f"{'Target Classes':<25}: {len(iris.target_names)}")
    print(f"{'Missing Values':<25}: {df.isnull().sum().sum()}")

    print("\nFlower Species")
    print("-" * 30)

    for species in iris.target_names:
        print(f"✓ {species.title()}")

    print("\nFeature Names")
    print("-" * 30)

    for feature in iris.feature_names:
        print(f"• {feature}")

    print("=" * 65)