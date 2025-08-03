import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("https://stepik.org/media/attachments/course/4852/train_data_tree.csv")


print(df.head())

X = df.drop('num', axis=1)
y = df['num']

clf = tree.DecisionTreeClassifier(criterion='entropy')

one_two_tree = clf.fit(X,y)

#tree.plot_tree(one_two_tree) 
#plt.show() #look tree for take the data 

e_parent = 0.996 #parent tree entropy
e_sub_1 = 0.903 #sub tree entropy
e_sub_2 = 0.826 #sub tree entropy
col_1 = 157 #col objects sub tree
col_2 = 81 #col objects sub tree
col = 238  #all objects 

ig = e_parent - ((col_1 / col * e_sub_1 + col_2 / col * e_sub_2))
print(ig)

"""
Now we try to use .tree_ functions  

"""
e_parent = clf.tree_.impurity[0] #get entropy 
e_sub_1 = clf.tree_.impurity[clf.tree_.children_left[0]] #get entropy 
e_sub_2 = clf.tree_.impurity[clf.tree_.children_right[0]] #get entropy  
col_1 = clf.tree_.n_node_samples[clf.tree_.children_left[0]] #get objects sub tree left
col_2 = clf.tree_.n_node_samples[clf.tree_.children_right[0]] #get objects sub tree right
col = clf.tree_.n_node_samples[0]  #get all objects 

ig = e_parent - ((col_1 / col * e_sub_1 + col_2 / col * e_sub_2))
print(ig)