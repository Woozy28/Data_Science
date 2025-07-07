import pandas as pd
import numpy as np


#читаем файлик
students_perfomance = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\StudentsPerformance.csv')

#filter on gender
print(students_perfomance.loc[students_perfomance.gender == 'female'])

writingscore_min_value = students_perfomance['writing score'].mean()

#filter on condition
print(students_perfomance.loc[students_perfomance['writing score'] > writingscore_min_value])

#get boolean mask
query = (students_perfomance['writing score'] > writingscore_min_value) & (students_perfomance.gender == 'female')

#filter on query
print(students_perfomance.loc[query])

# get % students with standart lunch
print(students_perfomance['lunch'].value_counts(normalize=True)) 

#get statistic group by 'lunch' column
print(students_perfomance.groupby('lunch').describe())

#use qwery
print(students_perfomance.query("gender == 'female'"))

#use filte - filter on lables only(columns)
print(students_perfomance.filter(like = 'score'))
