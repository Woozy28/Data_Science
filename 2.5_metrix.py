import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
precisions = precision_score(y_test, predictions, average='micro')



sum_photo = 100
tru_positive = 15
false_positive = 15
false_negative = 30

pr = tru_positive / (tru_positive + false_positive)
re = tru_positive / (tru_positive + false_negative)
f1 = (2*tru_positive) / ((2 * tru_positive) + false_positive + false_negative)


