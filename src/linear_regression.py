from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import load_and_prepare_data
from evaluation import evaluate_model
X_train, X_test, y_train, y_test = load_and_prepare_data()
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
class_predictions = (predictions >= 0.5).astype(int)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Linear Regression Metrics")
print("=" * 40)
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("R2 Score:", round(r2, 4))
accuracy = evaluate_model("Linear Regression", y_test, class_predictions)
print("Final Accuracy:", round(accuracy * 100, 2), "%")