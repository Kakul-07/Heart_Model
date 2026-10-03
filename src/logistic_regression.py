from sklearn.linear_model import LogisticRegression
from preprocessing import load_and_prepare_data
from evaluation import evaluate_model

X_train, X_test, y_train, y_test = load_and_prepare_data()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = evaluate_model(
    "Logistic Regression",
    y_test,
    predictions
)

print("Final Accuracy:", round(accuracy * 100, 2), "%")