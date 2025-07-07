import pandas as pd

dota_hero = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')

#count of legs in dota2 heroes
print(dota_hero.groupby('legs').aggregate({'legs': 'count'}))

#To find most popular attack type
print(dota_hero.groupby(['attack_type','primary_attr'], as_index=False).aggregate({'id':'sum'}).sort_values('id'))