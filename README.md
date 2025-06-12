# 🩺 Diabetes Prediction Web App

This is a Streamlit-powered web application that predicts the likelihood of diabetes using five machine learning models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machine (SVM). Users can input health parameters and get instant prediction results from each model.

## 🚀 Features

- Predict diabetes using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
- Clean and intuitive Streamlit UI
- Real-time predictions based on user input
- Model-wise comparison of results

## 📊 Input Parameters

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🌐
- Scikit-learn ⚙️
- Pandas, NumPy, Matplotlib 📊

## 📷 Screenshots

<img width="1440" alt="Screenshot 2025-06-12 at 7 25 02 PM" src="https://github.com/user-attachments/assets/9038283e-0b3f-4577-8ffb-20c5b0fff5b2" />

<img width="1440" alt="Screenshot 2025-06-12 at 7 27 14 PM" src="https://github.com/user-attachments/assets/81a44d35-dc1f-45ca-9b35-70539cb36dd3" />

<img width="1440" alt="Screenshot 2025-06-12 at 7 27 26 PM" src="https://github.com/user-attachments/assets/59ecdcf6-cb16-40de-a537-81bcd1bf27e4" />


## 🧠 Model Training
The models are trained on the popular Healthcare Diabetes dataset. Each model is trained, evaluated, and used for real-time prediction in the app.

## 🔧 Model Hyperparameters

The following hyperparameter grids were used for model tuning via `GridSearchCV`:

### Logistic Regression
- `C`: [0.1, 1, 10]
- `penalty`: ['l1', 'l2']
- `solver`: 'liblinear'

### Decision Tree
- `max_depth`: [None, 5, 10, 15]
- `min_samples_leaf`: [1, 5, 10]
- `min_samples_split`: [2, 5, 10]

### Random Forest
- `n_estimators`: [100, 200]
- `max_features`: ['sqrt', 'log2']
- `max_depth`: [10, 20, None]
- `min_samples_leaf`: [1, 5]

### Gradient Boosting
- `n_estimators`: 100 

### Linear SVM (LinearSVC)
- `C`: [0.1, 1, 10]
- `penalty`: ['l1', 'l2']
- `dual`: False

