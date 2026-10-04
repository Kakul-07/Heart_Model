from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os
from preprocessing import load_data, split_features_target, create_preprocessor
df = load_data()
X, y = split_features_target(df)
preprocessor = create_preprocessor(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
params = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipeline,params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
os.makedirs("model", exist_ok=True)
joblib.dump(grid.best_estimator_, "model/heart_logistic_model.pkl")
print("Model trained successfully.")
print("Best parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model saved at: model/heart_logistic_model.pkl")