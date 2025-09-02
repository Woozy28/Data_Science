import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('F:\Phyton\heart-disease.csv')


X = data.drop(['target'], axis=1)
y = data.target

X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=0.33, random_state=42)

np.random.seed(0)
rf = RandomForestClassifier(10, max_depth=5)

parameters = {'n_estimators' : [10, 20, 30], 'max_depth' : [2,5,7,10]} 
grid_search_cv_clf = GridSearchCV(rf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({
    'features' : list(X_train),
    'feature_importances': feature_importances
})
print(feature_importances_df.sort_values('feature_importances', ascending=False))
