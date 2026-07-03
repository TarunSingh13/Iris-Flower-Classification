"""
Evaluation Module
-----------------
Handles model evaluation and visualization.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.neighbors import KNeighborsClassifier

from config import (
    OUTPUT_FOLDER,
    CONFUSION_MATRIX_FILE,
    ACCURACY_GRAPH_FILE,
    K_VALUES
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    print("\nConfusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return y_pred, cm


def plot_confusion_matrix(cm, iris):
    """
    Save confusion matrix as an image.
    """

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_FOLDER}/{CONFUSION_MATRIX_FILE}"
    )

    plt.close()

    print("\n✅ Confusion Matrix saved successfully!")


def compare_k_values(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Compare different K values and plot accuracy.
    """

    accuracies = []

    for k in K_VALUES:

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        accuracies.append(accuracy)

    plt.figure(figsize=(8, 5))

    plt.plot(
        K_VALUES,
        accuracies,
        marker="o"
    )

    plt.title("Accuracy vs K Value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig(
        f"{OUTPUT_FOLDER}/{ACCURACY_GRAPH_FILE}"
    )

    plt.close()

    print("✅ Accuracy vs K graph saved successfully!")