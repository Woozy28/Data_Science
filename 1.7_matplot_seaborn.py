import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

students_performance = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\StudentsPerformance.csv')
students_performance = students_performance.rename(columns=
    {
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score',
        'reading score': 'reading_score',
        'writing score': 'writing_score'
    })

#matplotlib diagram
students_performance.plot.scatter(x='math_score', y='reading_score')
# plt.show() - need to show diagram

#seaborn diagram
ax = sns.lmplot(x='math_score',y='reading_score', hue='gender', data=students_performance)
ax.set_xlabels('math score')
ax.set_ylabels('reading score')
#plt.show()

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/income.csv')

sns.lineplot(x=df.index, y=df.income)
plt.show()

