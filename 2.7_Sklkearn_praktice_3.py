import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


iris = load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier(random_state=42)

# Определяем сетку параметров для перебора
param_grid = {
    'max_depth': list(range(1, 11)),           # Максимальная глубина от 1 до 10
    'min_samples_split': list(range(2, 11)),   # Минимальное число проб для разделения от 2 до 10
    'min_samples_leaf': list(range(1, 11))     # Минимальное число проб в листе от 1 до 10
}

# Создаем объект GridSearchCV
# estimator: модель, которую хотим обучить
# param_grid: словарь с параметрами для перебора
# cv: количество фолдов для кросс-валидации (по умолчанию 5, но можно указать явно)
# n_jobs: количество ядер процессора для использования (-1 означает использовать все доступные)
search = RandomizedSearchCV(estimator=clf,
                      param_distributions=param_grid,
                      cv=5,      # Обычно 5 или 10 фолдов
                      n_jobs=-1) # Использовать все доступные ядра CPU

# Обучаем GridSearchCV на данных, выполняя перебор и кросс-валидацию

search.fit(X, y)

# Получаем лучшую модель из результатов поиска
best_tree = search.best_estimator_

