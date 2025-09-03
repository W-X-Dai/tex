import pandas as pd

data = pd.read_csv('Bike.csv')

print("std of 2011 : ", data.loc[data["year"]==2011, "cnt"].std())
print("std of 2012 : ", data.loc[data["year"]==2012, "cnt"].std())
