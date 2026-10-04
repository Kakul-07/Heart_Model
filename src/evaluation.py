import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from preprocessing import load_data, split_features_target
df = load_data()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = joblib.load("model/heart_logistic_model.pkl")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report( y_test, y_pred, target_names=["No Heart Disease", "Heart Disease"]))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No Heart Disease", "Heart Disease"])
display.plot()
plt.title("Logistic Regression - Heart Disease")
plt.show()