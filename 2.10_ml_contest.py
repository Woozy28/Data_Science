import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree


"""
Давайте найдем такой стэп, используя данные о сабмитах. Для каждого 
пользователя найдите такой шаг, который он не смог решить, и после этого 
не пытался решать другие шаги. Затем найдите id шага,  который стал финальной 
точкой практического обучения на курсе для максимального числа пользователей. """

data = pd.read_csv('F:\Phyton\submissions_data_train.csv')
group_data = data.groupby(['step_id','submission_status'], as_index=False).agg(
    {
        'timestamp' : 'count'
    }
)

print(group_data[group_data.submission_status == 'wrong'].sort_values(by='timestamp', ascending=False))