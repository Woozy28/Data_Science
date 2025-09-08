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
from time import time


movie = pd.read_csv('F:\Phyton\movie_metadata.csv')

genres = movie[['movie_title','genres']]

#def reservation(value):
#    return value[::-1]

#for row in genres.values:
#    for value in row:
#        print(reservation(value))

budget = movie[['budget','duration']]
budget.apply(lambda x: x+1)


def mm(col):
    return np.mean(col) +1 

budget.apply(mm)


np.mean(budget['budget'].dropna().values)

before = time()
budget.mean(axis=0)
after = time()
print(after - before)

before = time()
budget.apply(np.mean)
after = time()
print(after - before)

before = time()
budget.describe().loc['mean']
after = time()
print(after - before)

before = time()
budget.apply('mean')
after = time()
print(after - before)

stock=pd.read_csv('https://raw.githubusercontent.com/PacktPublishing/Pandas-Cookbook/master/data/amzn_stock.csv', index_col='Date', parse_dates=True)

stock['2010']
stock['2010-02':'2011-03']
stock.resample('2h').asfreq()
stock.resample('1w').mean()
stock.head(10)
stock.rolling(3)
stock.rolling(3).mean()
stock.rolling(3, min_periods=1).mean()
stock.expanding(min_periods=3).mean()
stock.ewm(alpha=0.7).mean()
stock['Open'].plot()
ns=stock['Open'].rolling(10, min_periods=1).mean()
stock['Open'].plot()
ns.plot()


