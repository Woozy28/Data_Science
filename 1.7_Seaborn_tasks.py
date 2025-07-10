import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#find count clusters 
plt.figure(1)
data = pd.read_csv('F:\Phyton\EDU_Data_Science\Data_Science\dataset_209770_6.txt', sep=" ")
sns.lmplot(x='x' , y='y' , data=data)


#do heatmap of genome types 
plt.figure(2)
genome_types = pd.read_csv('https://stepik.org/media/attachments/course/4852/genome_matrix.csv' ,index_col=0)
sns.heatmap(data=genome_types, cmap="viridis")

#dota2 heroes again. need to find the most popular role in dota2
plt.figure(3)
dota_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')

dota_data.roles.str.split().str.len().hist() #here we calculate counts of heroes roles and show histogram. How is it work? i dnt now =) 


#flowers data set
flowers_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')

flowers_data = flowers_data.drop(columns='Unnamed: 0')

for columns in flowers_data:
       sns.displot(flowers_data, x=columns, kde=True)
plt.show()