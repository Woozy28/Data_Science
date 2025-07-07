import pandas as pd 
import numpy as np

students_perfomans = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\StudentsPerformance.csv')


#change symbols in column name " " to "_". We need that for easу acsess to the columns
students_perfomans = students_perfomans.rename(columns={
    'parental level of education' : 'parental_level_of_education', 
    'test preparation course': 'test_preparation_course', 
    'math score': 'math_score', 
    'reading score': 'reading_score',
    'writing score' : 'writing_score'}
)

#groupby and calculate mean off scores + rename columns
print(students_perfomans.groupby('gender').aggregate({'math_score' : 'mean', 'reading_score' : 'mean', 'writing_score' : 'mean'}).rename(
    columns={
        'math_score': 'mean_math_score',
        'reading_score': 'mean_reading_score',
        'writing_score': 'mean_writing_score'

    }
))

#get top 5 male and female students
print(students_perfomans.sort_values(['gender','math_score'], ascending=False) \
    .groupby('gender').head())

#add new columns without pd way
students_perfomans['total_score'] = students_perfomans.writing_score + students_perfomans.reading_score + students_perfomans.math_score



#add new column with pd way
students_perfomans = students_perfomans.assign(total_score_log = np.log(students_perfomans.total_score))

#delete columns
print(students_perfomans.drop(['total_score'], axis=1))



