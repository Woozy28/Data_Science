import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

events_data = pd.read_csv('F:\Phyton\event_data_train.csv')
submissions_data = pd.read_csv('F:\Phyton\submissions_data_train.csv')


"""
print(events_data.action.unique())
'viewed' - степ просмотрен 
'passed' решен
'discovered' - открыть впервые 
'started_attempt' - начато решение
"""

events_data['date'] = pd.to_datetime(events_data.timestamp, unit= 's') # add new column with data time type instead timestamp
events_data['day'] = events_data.date.dt.date # add new column with data without time

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit= 's') # add new column with data time type instead timestamp
submissions_data['day'] = submissions_data.date.dt.date # add new column with data without time

plt.figure(1)
events_data.groupby('day').user_id.nunique().plot() #unique people by day

#plt.figure(2)
#events_data[events_data.action == 'passed']\
#    .groupby('user_id', as_index = False)\
#        .agg({'step_id' : 'count'})\
#        .rename(columns={'step_id' : 'passed_steps'}).passed_steps.hist() #count of passed course

plt.figure(3)
users_events_data =  events_data.pivot_table(index='user_id', 
                        columns='action',
                        values='step_id', 
                        aggfunc='count', 
                        fill_value=0).reset_index() #count of passed course



plt.figure(4)
users_scores = submissions_data.pivot_table(index='user_id', 
                                            columns='submission_status', 
                                            values='step_id', 
                                            aggfunc='count', 
                                            fill_value=0).reset_index() #count of passed course

#here we get data of date gap by users
gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day'])\
    .groupby('user_id')['timestamp'].apply(list)\
    .apply(np.diff).values

gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_data = gap_data / (24 * 60 * 60 ) #get days from timestamp

gap_data[gap_data < 200].hist()


#print(gap_data.quantile(0.90))

#data_new = events_data.groupby('user_id')['action'].agg(lambda x: (x == 'viewed').sum())
#print(data_new.sort_values(ascending=True))


users_data = events_data.groupby('user_id', as_index= False).agg(
    {
        'timestamp' : 'max'
    }
).rename(columns={
    'timestamp' : 'max_timestamp'
})

now = 15267772811
drop_out = 25920000


users_data['is_gone_user'] = (now - users_data.max_timestamp) > drop_out


users_data = users_data.merge(users_scores, on = 'user_id', how = 'outer') #merge tables on outer join
users_data = users_data.fillna(0) # change NaN to zero


users_days = events_data.groupby('user_id').day.unique().to_frame().reset_index()
users_days['day'] =  users_days['day'].apply(len) # get unique count of days by users. I dont now why previous line is does'nt work

users_data = users_data.merge(users_events_data, on='user_id', how='left')
users_data = users_data.merge(users_days, on='user_id', how='left')
users_data['passed_corse'] = users_data.passed > 170

user_min_time = events_data.groupby('user_id', as_index=False).agg({
    'timestamp' : 'min'
}).rename({
    'timestamp' : 'min_timestamp'
},axis=1)
users_data = users_data.merge(user_min_time, how='left',on='user_id')

events_data_train = pd.DataFrame()
#for user_id in users_data.user_id:
#    min_user_time = users_data[users_data.user_id == user_id].min_timestamp.item()
#    time_threshold = min_user_time + 3 * 24 * 60 * 60
#    users_events_data = events_data[(events_data.user_id == user_id) & (events_data.timestamp < time_threshold)]
#    event_data_train = pd.concat([event_data_train, users_events_data])
events_data['user_time'] = events_data.user_id.map(str) + '_' + events_data.timestamp.map(str)

learning_time_threshold =  3 * 24 * 60 * 60
user_learning_time_threshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_threshold).map(str)

user_min_time['user_learning_time_threshold'] = user_learning_time_threshold
events_data = events_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')

events_data_train = events_data[events_data.user_time <= events_data.user_learning_time_threshold]

submissions_data['users_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.users_time <= submissions_data.user_learning_time_threshold]

X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index()
steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index()\
    .rename(columns={
        'step_id' : 'step_tried'
    })

X = X.merge(steps_tried, on = 'user_id', how= 'outer')

X = X.merge(submissions_data_train.pivot_table(index='user_id', 
                                            columns='submission_status', 
                                            values='step_id', 
                                            aggfunc='count', 
                                            fill_value=0).reset_index()) #count of passed course


X['correct_ratio'] = X.correct / (X.correct + X.wrong)

X = X.merge(events_data_train.pivot_table(index='user_id', 
                                            columns='action', 
                                            values='step_id', 
                                            aggfunc='count', 
                                            fill_value=0).reset_index()[['user_id','viewed']], how='outer') #count of passed course

X = X.fillna(0)

X = X.merge(users_data[['user_id','passed_corse','is_gone_user']], how='outer')

X = X[~((X.is_gone_user == False) & (X.passed_corse == False))]

y= X.passed_corse.map(int)
X = X.drop(['passed_corse','is_gone_user'], axis=1)

X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1) 
print(X.head())