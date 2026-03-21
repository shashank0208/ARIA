

from src.data_loader import load_data
from src.preprocess import build_preprocessor
from src.train import train_model
from src.evaluate import evaluate
from src.drift_analysis import compare_performance

def main():


    train, test_clean, test_drifted = load_data()


    X_train = train.drop(columns=["churn", "user_id"])
    y_train = train["churn"]

    X_test_clean = test_clean.drop(columns=["churn", "user_id"])
    y_test_clean = test_clean["churn"]

    X_test_drifted = test_drifted.drop(columns=["churn", "user_id"])
    y_test_drifted = test_drifted["churn"]


    preprocessor = build_preprocessor(X_train)


    model = train_model(X_train, y_train, preprocessor)

    clean_metrics = evaluate(model, X_test_clean, y_test_clean, "CLEAN DATA")
    drift_metrics = evaluate(model, X_test_drifted, y_test_drifted, "DRIFTED DATA")


    compare_performance(clean_metrics, drift_metrics)

if __name__ == "__main__":
    main()