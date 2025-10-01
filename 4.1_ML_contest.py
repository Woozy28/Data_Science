import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def time_filter(data, days=2):
    
    """Фильтрация данных до порогового значения"""
    
    # создаем таблицу с первым и последним действием юзера
    min_max_user_time = data.groupby('user_id').agg({'timestamp': 'min'}) \
                            .rename(columns={'timestamp': 'min_timestamp'}) \
                            .reset_index()
    
    data_time_filtered = pd.merge(data, min_max_user_time, on='user_id', how='outer')
    
    # отбираем те записи, которые не позднее двух дней с начала учебы
    learning_time_threshold = days * 24 * 60 * 60
    data_time_filtered = data_time_filtered.query("timestamp <= min_timestamp + @learning_time_threshold")
    
    assert data_time_filtered.user_id.nunique() == data.user_id.nunique()
    
    return data_time_filtered.drop(['min_timestamp'], axis=1)

def base_features(events_data, submission_data):
    
    """Создание датасета с базовыми фичами: действия юзера 
    и правильные\неправильные ответы"""
    
    # построим таблицу со всеми действиями юзеров
    users_events_data = pd.pivot_table(data=events_data, values='step_id',
                                   index='user_id', columns='action',
                                   aggfunc='count', fill_value=0) \
                                   .reset_index() \
                                   .rename_axis('', axis=1)
    
    # таблица с колво правильных и неправильных попыток
    users_scores = pd.pivot_table(data=submission_data, 
                              values='step_id',
                              index='user_id',
                              columns='submission_status',
                              aggfunc='count',
                              fill_value=0).reset_index() \
                              .rename_axis('', axis=1)
    
    # соединяем в один датасет
    users_data = pd.merge(users_scores, users_events_data, on='user_id', how='outer').fillna(0)
    
    assert users_data.user_id.nunique() == events_data.user_id.nunique()
    
    return users_data

def target(submission_data, threshold=40):
    
    """Вычисление целевой переменной. Если юзер сделал 40 практический заданий,
    то будем считать, что он пройдет курс до конца"""
    
    # считаем колво решенных заданий у каждого пользователя
    users_count_correct = submission_data[submission_data.submission_status == 'correct'] \
                .groupby('user_id').agg({'step_id': 'count'}) \
                .reset_index().rename(columns={'step_id': 'corrects'})
    
    # если юзер выполнил нужное колво заданий, то он пройдет курс до конца
    users_count_correct['passed_course'] = (users_count_correct.corrects >= threshold).astype('int')
    
    return users_count_correct.drop(['corrects'], axis=1)

def time_features(events_data):
    
    """Создание временных фичей"""
    
    # добавление колонок с датами
    events_data['date'] = pd.to_datetime(events_data['timestamp'], unit='s')
    events_data['day'] = events_data['date'].dt.date
    
    # создаем таблицу с первым\последним действием юзера и колвом уникальных дней, проведенных на курсе
    users_time_feature = events_data.groupby('user_id').agg({'timestamp': ['min', 'max'], 'day': 'nunique'}) \
                        .droplevel(level=0, axis=1) \
                        .rename(columns={'nunique': 'days'}) \
                        .reset_index()
    
    # добавление колонки с разницей между первым и последним появлением юзера,
    # другими словами, сколько времени юзер потратил на прохождение в часах
    users_time_feature['hours'] = round((users_time_feature['max'] - users_time_feature['min']) / 3600, 1)
    
    
    return users_time_feature.drop(['max', 'min'], axis=1)
def steps_tried(submission_data):
    
    """Создание фичи с колвом уникальных шагов, которые пользователь пытался выполнить"""
    
    # сколько степов юзер попытался сделать
    steps_tried = submission_data.groupby('user_id').step_id.nunique().to_frame().reset_index() \
                                        .rename(columns={'step_id': 'steps_tried'})
    
    return steps_tried
def correct_ratio(data):
    
    """Создание фичи с долей правильных ответов"""
    
    data['correct_ratio'] = (data.correct / (data.correct + data.wrong)).fillna(0)
    
    return data

def create_df(events_data, submission_data):
    
    """функция для формирования X датасета и y с целевыми переменными"""
    
    # фильтруем данные по дням от начала учебы
    events_2days = time_filter(events_data)
    submissions_2days = time_filter(submission_data)
    
    # создаем таблицу с базовыми фичами
    users_data = base_features(events_2days, submissions_2days)
    
    # создаем целевую переменную
    users_target_feature = target(submission_data, threshold=40)
    
    # создаем таблицу с временными фичами
    users_time_feature = time_features(events_2days)
    
    # создаем фичи с попытками степов и долей правильных ответов
    users_steps_tried = steps_tried(submissions_2days)
    users_data = correct_ratio(users_data)
    
    # соединяем шаги
    first_merge = users_data.merge(users_steps_tried, how='outer').fillna(0)
    
    # соединяем фичи со временем
    second_merge = first_merge.merge(users_time_feature, how='outer')
    
    # присоединяем целевую переменную
    third_merge = second_merge.merge(users_target_feature, how='outer').fillna(0)
    
    # отделяем целевую переменную и удаляем ее из основного датасета
    y = third_merge['passed_course'].map(int)
    X = third_merge.drop(['passed_course'], axis=1)
    
    return X, y
def create_test_df(events_data, submission_data):
    
    """функция для формирования test датасета без целевой переменной"""
    
    # фильтруем данные по дням от начала учебы
    events_2days = time_filter(events_data)
    submissions_2days = time_filter(submission_data)
    
    # создаем таблицу с базовыми фичами
    users_data = base_features(events_2days, submissions_2days)
    
    
    # создаем таблицу с временными фичами
    users_time_feature = time_features(events_2days)
    
    # создаем фичи с попытками степов и долей правильных ответов
    users_steps_tried = steps_tried(submissions_2days)
    users_data = correct_ratio(users_data)
    
    # соединяем шаги
    first_merge = users_data.merge(users_steps_tried, how='outer').fillna(0)
    
    # соединяем фичи со временем
    X = first_merge.merge(users_time_feature, how='outer')
       
    return X

events_data_train = pd.read_csv('./datasets/event_data_train.zip')
submission_data_train = pd.read_csv('./datasets/submissions_data_train.zip')

X_train, y = create_df(events_data_train, submission_data_train)