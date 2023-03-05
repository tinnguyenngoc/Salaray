from re import X
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('LR init')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)
print(y_test)

def Draw(x_train, y_test, y_pred):
    from sklearn.metrics import mean_absolute_error
    mse = mean_absolute_error(y_test, y_pred)
    print('mse', mse)

    # plt.scatter(x_train, y_train, color= 'blue' )
    # plt.plot(x_test, y_pred, color = 'red')
    # plt.show()

    X_grid = np.arange(min(x_train), max(x_train), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))

    plt.scatter(x, y, color= 'red' )
    plt.plot(X_grid, clf.predict(X_grid), color = 'blue')
    plt.show()
