from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import prepare_xy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model", choices=["logistic_regression", "svc"], default="logistic_regression")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.model == "logistic_regression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000)),
        ])
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
        }
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42, probability=True)),
        ])
        param_grid = [
            {"model__C": [0.1, 1, 10, 50], "model__kernel": ["linear"]},
            {"model__C": [0.1, 1, 10, 50], "model__kernel": ["rbf"], "model__gamma": [0.1, 0.01, 0.001, "scale"]},
        ]

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best parameters:", grid.best_params_)
    print("Cross-validation accuracy:", grid.best_score_)
    print("Test accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, args.output_dir / f"best_{args.model}.joblib")


if __name__ == "__main__":
    main()
