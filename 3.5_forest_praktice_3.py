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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/space_can_be_a_dangerous_place.csv')

print(data.head())


X = data.drop(['dangerous'], axis=1) 
y = data.dangerous

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X,y)

feature_importances = clf.feature_importances_
feature_importances_df = pd.DataFrame({
    'features' : list(X),
    'feature_importances': feature_importances
})

print(feature_importances_df)