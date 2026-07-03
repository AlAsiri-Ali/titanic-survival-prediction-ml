from __future__ import annotations

import re
import pandas as pd


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    title = match.group(1).strip() if match else "Rare"
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Rare"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Titanic survival prediction features."""
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["Deck"] = df["Cabin"].fillna("M").astype(str).str[0]
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Fare_log"] = (df["Fare"].fillna(df["Fare"].median()) + 1).apply(lambda x: pd.NA if x <= 0 else x)
    df["Fare_log"] = pd.to_numeric(df["Fare_log"]).apply(lambda x: __import__("numpy").log1p(x))
    df["Age"] = df["Age"].fillna(df.groupby(["Pclass", "Title"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median()).clip(upper=65)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"],
        include_lowest=True,
    )
    df["FareBin"] = pd.qcut(
        df["Fare"].fillna(df["Fare"].median()),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex_Encoded"] = df["Sex"].map({"female": 0, "male": 1})
    age_order = {"Child": 0, "Teenager": 1, "Young Adult": 2, "Adult": 3, "Senior": 4}
    fare_order = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    df["AgeBin_Ordinal"] = df["AgeBin"].map(age_order).astype(float)
    df["FareBin_Ordinal"] = df["FareBin"].map(fare_order).astype(float)
    df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], drop_first=True)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Sex", "AgeBin", "FareBin", "Age", "Fare", "SibSp", "Parch"]
    return df.drop(columns=drop_cols, errors="ignore")


def prepare_xy(df: pd.DataFrame):
    df = encode_features(add_features(df))
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y
