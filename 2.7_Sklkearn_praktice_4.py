import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree


train = []
test = []

y = train.y
X = train.drop('y', axis=1)

parameters = {'max_depth':range(1, 11),'min_samples_split':range(2,11),'min_samples_leaf':range(1,11)}

clf = tree.DecisionTreeClassifier()

search = GridSearchCV(estimator=clf, param_grid=parameters)

search.fit(X,y)

best_tree = search.best_estimator_

predictions = best_tree.predict(test)