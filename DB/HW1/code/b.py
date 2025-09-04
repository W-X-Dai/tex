import pandas as pd

data = pd.read_csv('Bike.csv')

monthly = data.groupby(["year", "month"])[["registered", "casual"]].sum()
monthly = monthly.sort_index()
print(monthly)

# print(monthly)

# check = (monthly["registered"] > monthly["casual"])
# print(check)
