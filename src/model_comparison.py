from sklearn.linear_model import LinearRegression, LogisticRegression
from preprocessing import load_and_prepare_data
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_and_prepare_data()

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
linear_predictions = (linear_predictions >= 0.5).astype(int)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

linear_accuracy = accuracy_score(y_test, linear_predictions)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("Model Comparison")
print("=" * 40)
print("Linear Regression Accuracy:", round(linear_accuracy * 100, 2), "%")
print("Logistic Regression Accuracy:", round(logistic_accuracy * 100, 2), "%")