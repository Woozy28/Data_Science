import pandas as pd
import numpy as np

titanic_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/titanic.csv')
students_perfomance = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\StudentsPerformance.csv')

print(students_perfomance.dtypes.value_counts())
print(students_perfomance.shape)
print(students_perfomance.iloc[:1,:])



