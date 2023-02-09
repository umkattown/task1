import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI

# Загрузка данных
train_data = pd.read_csv("credit_train.csv")
test_data = pd.read_csv("credit_test.csv")

# Отдельные функции и цель
X_train = train_data.drop("open_account_flg", axis=1)
y_train = train_data["open_account_flg"]

# Обработка пропущенных значений в числовых данных
imputer = SimpleImputer(strategy="mean")
X_train_num = X_train.select_dtypes(include=["float64", "int64"])
X_train_num = imputer.fit_transform(X_train_num)

# Обработка пропущенных значений в категориальных данных
imputer_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = X_train.select_dtypes(include=["object"])
X_train_cat = imputer_cat.fit_transform(X_train_cat)

# Преобразуйте категориальные данные в числовые данные, используя однократное кодирование
encoder = OneHotEncoder()
X_train_cat = encoder.fit_transform(X_train_cat)

# Объедините числовые и категориальные данные
X_train = np.concatenate((X_train_num, X_train_cat.toarray()), axis=1)

# Обучить модель логистической регрессии
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Оцените модель на данных проверки
y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Обработка пропущенных значений в тестовых данных
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
