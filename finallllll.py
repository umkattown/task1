import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI

# Load the datasets
train_data = pd.read_csv("credit_train.csv")
test_data = pd.read_csv("credit_test.csv")

# Separate features and target
X_train = train_data.drop("open_account_flg", axis=1)
y_train = train_data["open_account_flg"]

# Handle missing values in numerical data
imputer = SimpleImputer(strategy="mean")
X_train_num = X_train.select_dtypes(include=["float64", "int64"])
X_train_num = imputer.fit_transform(X_train_num)

# Handle missing values in categorical data
imputer_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = X_train.select_dtypes(include=["object"])
X_train_cat = imputer_cat.fit_transform(X_train_cat)

# Convert categorical data to numerical data using one-hot encoding
encoder = OneHotEncoder()
X_train_cat = encoder.fit_transform(X_train_cat)

# Combine numerical and categorical data
X_train = np.concatenate((X_train_num, X_train_cat.toarray()), axis=1)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model on the validation data
y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Handle missing values in test data
X_test = test_data.drop("open_account_flg", axis=1)
y_test = test_data["open_account_flg"]

X_test_num = X_test.select_dtypes(include=["float64", "int64"])
X_test_num = imputer.transform(X_test_num)

X_test_cat = X_test.select_dtypes(include=["object"])
X_test_cat = imputer_cat.transform(X_test_cat)

X_test_cat = encoder.transform(X_test_cat)

# FastAPI
app = FastAPI()
@app.post("/predict")
def predict(X: np.ndarray):
    X_num = X[:, X_train_num.columns]
    X_cat = X[:, X_train_cat.columns]

    X_num = imputer.transform(X_num)
    X_cat = imputer_cat.transform(X_cat)
    X_cat = encoder.transform(X_cat)

    X = np.concatenate((X_num, X_cat.toarray()), axis=1)

    y_pred = clf.predict(X)

    return {"prediction": y_pred.tolist()}
