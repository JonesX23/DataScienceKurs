import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


data = pd.read_excel(
    r"C:/Users/Jones/DataScienceKurs/data/Census-Datensatz.xlsx")

# print(data.describe())

x = data['hours-per-week']
y = data['education-num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


print(x_train.describe())
print(x_test.describe())

print(y_train.describe())
print(y_test.describe())
