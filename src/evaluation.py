import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def evaluate_model(model_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print("=" * 50)
    print(model_name)
    print("=" * 50)
    print("\nAccuracy:")
    print(round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(model_name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return accuracy