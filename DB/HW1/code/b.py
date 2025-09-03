import pandas as pd

data = pd.read_csv('Bike.csv')

monthly = data.groupby("month")[["registered", "casual"]].sum()

print(monthly)

check = (monthly["registered"] > monthly["casual"])
print(check)
