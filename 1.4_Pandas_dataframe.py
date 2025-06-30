import pandas as pd
import numpy as np

#можно читать с сылок (я в ахуе если честно)
titanic_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/titanic.csv')

#читаем файлик
students_perfomance = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\StudentsPerformance.csv')

#Dtype - выодит типы данных.. количество значений - количество значений
print(students_perfomance.dtypes.value_counts())

# колонки строки. size выведет их произведение
print(students_perfomance.shape)

#получаем выбранные строки и столбцы по индексам 
print(students_perfomance.iloc[:1,:])



