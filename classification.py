from data_loader import (
    load_dataset,
    explore_dataset
)

from preprocessing import (
    split_data,
    scale_features
)

from model import (
    train_model,
    save_model,
    predict_flower
)

from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    compare_k_values
)


def main():

    print("=" * 60)
    print("🌸 IRIS FLOWER CLASSIFICATION SYSTEM")
    print("=" * 60)

    # Load Dataset
    iris, df = load_dataset()

    # Explore Dataset
    explore_dataset(df, iris)

    # Split Dataset
    X_train, X_test, y_train, y_test = split_data(df)

    # Scale Dataset
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    # Train Model
    model = train_model(
        X_train_scaled,
        y_train
    )

    # Save Model
    save_model(model)

    # Evaluate Model
    y_pred, cm = evaluate_model(
        model,
        X_test_scaled,
        y_test
    )

    # Generate Confusion Matrix
    plot_confusion_matrix(
        cm,
        iris
    )

    # Compare K Values
    compare_k_values(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    # Interactive Prediction
    predict_flower(
        scaler,
        iris
    )


if __name__ == "__main__":
    main()