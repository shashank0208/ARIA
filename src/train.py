

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODEL_OUTPUT_PATH, RANDOM_STATE

def train_model(X_train, y_train, preprocessor):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_OUTPUT_PATH)

    return pipeline