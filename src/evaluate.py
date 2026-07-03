from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocess import prepare_xy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = prepare_xy(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
