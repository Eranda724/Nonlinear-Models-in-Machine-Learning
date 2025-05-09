import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Load Dataset
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Handle missing values
df = df.dropna()

# Encode categorical variables
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

# Split into Input and Target
X = df.drop(columns=['Diabetes_binary'])
y = df['Diabetes_binary']

# Split dataset (70% Train, 20% Validation, 10% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Radial Basis Function Model
rbf_model = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test) > 0.5

# Perceptron Model
perc_model = Perceptron()
perc_model.fit(X_train, y_train)
perc_pred = perc_model.predict(X_test)

# Multi-Layer Perceptron Model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200)
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")

# Evaluate All Models
evaluate_model(y_test, rbf_pred, "Radial Basis Function Model")
evaluate_model(y_test, perc_pred, "Perceptron Model")
evaluate_model(y_test, mlp_pred, "Multi-Layer Perceptron Model")
