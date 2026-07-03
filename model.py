import os
import joblib
from sklearn.neighbors import KNeighborsClassifier

from config import (
    N_NEIGHBORS,
    MODEL_PATH
)


def train_model(X_train, y_train):
    """
    Train the KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model.fit(X_train, y_train)

    return model


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\n✅ Model saved successfully!")


def load_model():

    return joblib.load(MODEL_PATH)


def predict_flower(scaler, iris):

    print("\n" + "=" * 60)
    print("🌸 Predict a New Flower")
    print("=" * 60)

    try:
        # Load saved model
        model = load_model()

        # User Input
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

        # Create sample
        sample = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Scale sample
        sample = scaler.transform(sample)

        # Prediction
        prediction = model.predict(sample)

        # Prediction Probability
        probabilities = model.predict_proba(sample)[0]

        # Display Prediction
        flower = iris.target_names[prediction[0]]

        print("\n" + "=" * 60)
        print("🌼 FLOWER CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"Predicted Species : {flower.title()}")

        # Display Confidence
        print("\n📊 Prediction Confidence")
        print("-" * 35)

        for species, probability in zip(
                iris.target_names,
                probabilities):
            print(f"{species.title():12} : {probability * 100:.2f}%")

        print("=" * 60)

    except ValueError:
        print("\n❌ Invalid Input! Please enter only numeric values.")

    except FileNotFoundError:
        print("\n❌ Trained model not found. Please train the model first.")