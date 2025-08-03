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

from IPython.display import HTML
style = "<style>.svg(width:70% !important;height:70% !important;)</style>"
HTML(style)

train_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/dogs_n_cats.csv') #get train data
test_data = pd.read_json('F:\Phyton\dataset_209691_15 (1).txt') #get test data


clf = tree.DecisionTreeClassifier(criterion = 'entropy')
X = train_data.drop(['Вид'], axis=1)
y = train_data['Вид']
clf.fit(X,y)

predicted_labels = clf.predict(test_data) # The .predict() method takes input and returns an array of predicted labels.
prediction_counts = pd.Series(predicted_labels).value_counts()
print(prediction_counts)