import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
def load_data():
    df = pd.read_csv("Dataset/Heart.csv")
    return df
def split_features_target(df):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return X, y
def create_preprocessor(X):
    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numerical", numerical_pipeline, numerical_columns), ("categorical", categorical_pipeline, categorical_columns)])
    return preprocessor