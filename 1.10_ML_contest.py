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

