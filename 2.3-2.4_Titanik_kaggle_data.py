from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from IPython.display import HTML
style = "<style>.svg(width:70% !important;height:70% !important;)</style>"
HTML(style)



titanic_data = pd.read_csv("F:\Phyton\Train.csv")


X = titanic_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)
y = titanic_data.Survived

X = pd.get_dummies(X) #break columns to 1\0 bool meaning from. sex = male, female. after dummies = sex_male = 1\0 and sex_female = 1\0  columns

X = X.fillna({'Age' : X.Age.median()})

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth= 3)

#tree.plot_tree(clf.fit(X,y))
#plt.show()

X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=0.33, random_state=42)

max_depth_values = range(1, 100)

scores_data = pd.DataFrame()

for max_depth in max_depth_values: 
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth= max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_score_data = pd.DataFrame({
        'max_depth' : [max_depth],
        'train_score' : [train_score],
        'test_score' : [test_score],
        'cross_score': [mean_cross_val_score]
    })
    scores_data = scores_data._append(temp_score_data)


scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score','test_score','cross_score'], var_name='set_type', value_name='score')

sns.lineplot(data=scores_data_long, x='max_depth', y='score', hue='set_type')

params = {'criterion':['gini','entropy'], 'max_depth' : range(1,30)}

clf = tree.DecisionTreeClassifier()
gridSearch_clf = GridSearchCV(clf, params, cv=5)

gridSearch_clf.fit(X_train,y_train)
#gridSearch_clf.best_params_ - show best collection of parameters
 
best_clf = gridSearch_clf.best_estimator_



clf_rf = RandomForestClassifier()
parameters = {'n_estimators' : [10, 20, 30], 'max_depth' : [2,5,7,10]} 
grid_search_cv_clf = GridSearchCV(clf_rf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({
    'features' : list(X_train),
    'feature_importances': feature_importances
})
print(feature_importances_df.sort_values('feature_importances', ascending=False))