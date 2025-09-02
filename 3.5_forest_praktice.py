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

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv') #train data
test_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/testing_mush.csv') #test data. This data we need to predict

data = data.rename(columns={
    'class' : 'mushroom_class'
}) # rename because we cant use name class


X = data.drop(['mushroom_class'], axis=1) 
y = data.mushroom_class

clf = RandomForestClassifier(random_state=0) 
parameters = {
    'n_estimators' : [10,20,30,40,50],
    'max_depth' : [1,3,5,7,9,11],
    'min_samples_leaf' : [1,2,3,4,5,6,7],
    'min_samples_split' : [2,4,6,8]
} # parameters of trees

grid_sech = GridSearchCV(clf,parameters,cv=3,n_jobs=-1)
grid_sech.fit(X,y)

best_clf = grid_sech.best_estimator_
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({
    'features' : list(X),
    'feature_importances': feature_importances
})

predict = grid_sech.predict(test_data)
prediction_counts = pd.Series(predict).value_counts()
print(prediction_counts)
