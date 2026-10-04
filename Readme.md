Heart Disease Prediction using Logistic Regression

About the Project

This project uses Machine Learning to predict whether a person is likely to have heart disease based on different medical and clinical features.

The dataset was first explored through Exploratory Data Analysis (EDA). After understanding the data, preprocessing was performed and a Logistic Regression model was trained and evaluated.

Dataset

The dataset used for this project is Heart.csv.

It contains information about patients such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Induced Angina
* Oldpeak
* ST Slope

The target variable is HeartDisease.

* 0 = No Heart Disease
* 1 = Heart Disease

Exploratory Data Analysis

EDA was performed before model training to understand the dataset.

The analysis included:

* Checking the shape of the dataset
* Checking data types
* Checking missing values
* Studying numerical features
* Studying categorical features
* Checking the target distribution
* Understanding relationships between features and heart disease

The EDA notebook is available in the Notebook folder.

Data Preprocessing

The preprocessing is handled in src/preprocessing.py.

The following steps are performed:

1. Load the dataset.
2. Separate input features and the target variable.
3. Identify numerical and categorical columns.
4. Fill missing numerical values using the median.
5. Fill missing categorical values using the most frequent value.
6. Standardize numerical features using StandardScaler.
7. Convert categorical features into numerical form using OneHotEncoder.

The preprocessing is included in a pipeline so that the same transformations are applied during training and prediction.

Model Used

The model used in this project is Logistic Regression.

Logistic Regression is suitable for this problem because the target variable is binary. The model predicts one of two classes:

* No Heart Disease
* Heart Disease

The model is trained using the training portion of the dataset and evaluated using the test portion.

Model Training

The dataset is divided into:

* 80% training data
* 20% testing data

GridSearchCV is used to test different values of the Logistic Regression regularization parameter C.
The values tested are:
0.01
0.1
1
10
100

Five-fold cross-validation is used to select the best value.
The best value obtained was:
C = 0.1

*Model Performance*
The final Logistic Regression model achieved:
Test Accuracy: 89.13%
Classification report:
                  precision    recall  f1-score   support

No Heart Disease       0.91      0.84      0.87        82
Heart Disease          0.88      0.93      0.90       102

accuracy                           0.89       184
macro avg              0.89      0.89      0.89       184
weighted avg           0.89      0.89      0.89       184
Confusion matrix:
[[69 13]
 [ 7 95]]
The model correctly classified 69 patients without heart disease and 95 patients with heart disease.

*Training the Model*
From the project root, run:
python3 src/train.py

The trained model will be saved as:
model/heart_logistic_model.pkl

Evaluating the Model
Run:
python3 src/evaluate.py
This displays:
* Accuracy
* Classification report
* Confusion matrix

Running the Streamlit App
Install the required packages:
pip3 install -r requirements.txt
Run the application:
python3 -m streamlit run app.py

The Streamlit application allows the user to enter patient information and receive a prediction.

Conclusion

This project covers the complete Machine Learning workflow starting from EDA and preprocessing, followed by Logistic Regression training and evaluation.

The final model achieved an accuracy of 89.13% on the test dataset and was also integrated into a Streamlit application for easy use.