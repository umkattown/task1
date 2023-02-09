import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Загружаем наборы данных
train_data = pd.read_csv("credit_train.csv")
test_data = pd.read_csv("credit_test.csv")

# Отдельные функции и цель
X_train = train_data.drop("open_account_flg", axis=1)
y_train = train_data["open_account_flg"]

# Encode data
X_train = pd.get_dummies(X_train)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)

# Обучите модель логистической регрессии
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