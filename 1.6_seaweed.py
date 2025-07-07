import pandas as pd

concentrations = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')

#get mean concentrations of substans
mean_concentrations = concentrations.groupby('genus').aggregate({
    'sucrose' : 'mean',
    'alanin' : 'mean',
    'citrate' : 'mean',
    'glucose' : 'mean',
    'oleic_acid' : 'mean'
})

#find max min mean of alanin in Fucus
alanin_data = concentrations.loc[concentrations.genus == 'Fucus'].groupby('genus').agg(
    
        alanin_mean =('alanin' , 'mean'), #create new columns
        alanin_max=('alanin' , 'max'),
        alanin_min = ('alanin' , 'min')
    
)
#find statistical values
print(concentrations.groupby('group')[['sucrose','citrate']].describe())