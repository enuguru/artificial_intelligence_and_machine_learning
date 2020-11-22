import pandas as pd
df=pd.read_csv('data.csv')
data=pd.crosstab(df.Nationality,df.Handedness)
data2=pd.crosstab(df.Sex,df.Nationality)
data3=pd.crosstab(df.Sex,[df.Nationality,df.Handedness])
print(data)
print()
print(data2)
print()
print(data3)

