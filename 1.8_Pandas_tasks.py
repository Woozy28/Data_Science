import pandas as pd
import numpy as np

data = {'type':['A','A','B','B'],'value':[10,14,12,23]}
my_data = pd.DataFrame(data=data)


my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat.csv')

subset_11 = my_stat.iloc[0:9,[0,2]]
subset_22 = my_stat.iloc[:,[1,3]].drop(my_stat.index[[0,4]])

"""В переменную subset_1 сохраните только те наблюдения, у которых значения переменной V1  строго больше 0, и значение переменной V3  равняется 'A'.
В переменную  subset_2  сохраните только те наблюдения, у которых значения переменной V2  не равняются 10, или значения переменной V4 больше или равно 1."""

subset_1 = my_stat.query("V1 > 0 and V3 == 'A'")
subset_2 = my_stat.query("V2 != 10 or V4 >= 1")

"""V5 = V1 + V4
V6 = натуральный логарифм переменной V2"""

my_stat['V5'] = my_stat['V1'] + my_stat['V4']
my_stat['V6'] = np.log(my_stat['V2']) 

"""Переименуйте колонки в данных  my_stat следующим образом:
V1 -> session_value
V2 -> group
V3 -> time
V4 -> n_users"""

my_stat = my_stat.rename(
    columns={
        'V1' : 'session_value',
        'V2' : 'group',
        'V3' : 'time',
        'V4' : 'n_users'
    }
)

"""В переменной session_value замените все пропущенные значения на нули.
В переменной n_users замените все отрицательные значения на медианное значение переменной n_users (без учета отрицательных значений, разумеется)."""

my_stat['session_value'] = my_stat['session_value'].fillna(0)
n_users_median = my_stat[my_stat.n_users  >= 0].n_users.median()
my_stat.loc[my_stat['n_users'] < 0, 'n_users'] = n_users_median 
print(my_stat)

"""В этой задаче для данных my_stat рассчитайте среднее значение переменной session_value для каждой группы (переменная group), 
в получившемся dataframe  переменная group не должна превратиться в индекс. Также переименуйте колонку со средним значением session_value в mean_session_value.
Получившийся результат сохраните в dataframe с именем mean_session_value_data."""

mean_session_value_data = my_stat.groupby('group', as_index=False).agg({
    'session_value' : 'mean'
}).rename(
    columns={
        'session_value' : 'mean_session_value'
    }
)
print(mean_session_value_data)