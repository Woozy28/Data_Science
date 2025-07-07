import pandas as pd

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/accountancy.csv')

#need to compare sales of lupa and pupa
print(data.groupby(['Executor','Type'], as_index=False).aggregate({'Salary':'mean'}))