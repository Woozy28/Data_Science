import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dt = tree.DecisionTreeClassifier()

dt.fit(X_train,y_train)

predicted = dt.predict(X_test)