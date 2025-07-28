from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m

data = pd.DataFrame({
    'Wool': [1,1,1,1,1,1,1,1,1,0],
    'bark': [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    'climbs_trees': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    'Y': ['dog','dog','dog','dog','cat','cat','cat','cat','cat','cat']
})

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
X = data[['Wool','bark','climbs_trees']]
y = data.Y
clf.fit(X,y)

tree.plot_tree(clf.fit(X,y)) #watch decision tree


E_sh_sob=(1/1)*m.log2((1/1)) - 0
E_sh_kot=-(4/9)*m.log2((4/9)) - (5/9)*m.log2((5/9))
E_gav_sob=0 - (5/5)*m.log2((5/5))
E_gav_kot=-(4/5)*m.log2((4/5)) - (1/5)*m.log2((1/5))
E_laz_sob=0 - (6/6)*m.log2((6/6))
E_laz_kot=-(4/4)*m.log2((4/4)) - 0

print(E_gav_kot,E_gav_sob)

# Шерстист
N = 10 
E_YX_SHERSTIT = (1/N) * 0 + (9/10) * 0.99
IG_SHERSTIT = 0.97 - E_YX_SHERSTIT

# Гавкает
E_YX_GAVKAET = (5/N) * 0 + (5/10) * 0.72
IG_GAVKAET = 0.97 - E_YX_GAVKAET

# Лазает
E_YX_LAZAET = (4/N) * 0 + (6/10) * 0
IG_LAZAET = 0.97 - E_YX_LAZAET

print(f"IG_Шерстист | IG_Гавкает | IG_Лазает\n{round(IG_SHERSTIT, 2)} {IG_GAVKAET} {IG_LAZAET}")