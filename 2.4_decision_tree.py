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

train_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv') #get train data
test_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/test_iris.csv') #get test data


X_train =  train_data.drop(['species','Unnamed: 0'], axis=1) 
y_train = train_data.species
X_test  = test_data.drop(['species','Unnamed: 0'], axis=1)
y_test = test_data.species


max_depth_values = range(1, 100) #how many times we train 

scores_data = pd.DataFrame() #Frame for score 

for max_depth in max_depth_values: 
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth= max_depth, random_state=0) # decision tree params 
    clf.fit(X_train,y_train) # train
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)

    temp_score_data = pd.DataFrame({
        'max_depth' : [max_depth],
        'train_score' : [train_score],
        'test_score' : [test_score]
    })
    scores_data = scores_data._append(temp_score_data)

#Change the DataFrame format from wide to long
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                           value_vars=['train_score','test_score'], 
                           var_name='set_type', 
                           value_name='score')

sns.lineplot(data=scores_data_long, x='max_depth', y='score', hue='set_type')
plt.show()