import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    base_path = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base_path / "Dataset" / "Heart.csv")

    df["RestingBP"] = df["RestingBP"].replace(0, pd.NA)
    df["Cholesterol"] = df["Cholesterol"].replace(0, pd.NA)

    df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())
    df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    )

    df["MaxHR_Percent"] = df["MaxHR"] / (220 - df["Age"])

    df["BPCategory"] = pd.cut(
        df["RestingBP"],
        bins=[0, 120, 130, 140, float("inf")],
        labels=["Normal", "Elevated", "High", "Very High"]
    )

    df["CholesterolCategory"] = pd.cut(
        df["Cholesterol"],
        bins=[0, 200, 240, float("inf")],
        labels=["Normal", "Borderline", "High"]
    )

    df = df.dropna()

    df = pd.get_dummies(
        df,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope",
            "AgeGroup",
            "BPCategory",
            "CholesterolCategory"
        ],
        drop_first=True
    )

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test