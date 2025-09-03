import pandas as pd

data = pd.read_csv('Bike.csv')

summary = data.groupby("workday")[["registered", "casual"]].sum()
print(summary)

print("sum of workday", (data["workday"] == 1).sum())
print("sum of non-workday", (data["workday"] == 0).sum())

# not a workday, registered mean
avg_registered_nonwork = data.loc[data["workday"]==0, "registered"].mean()

# not a workday, registered median
med_registered_nonwork = data.loc[data["workday"]==0, "registered"].median()

# a workday, casual mean
avg_casual_work = data.loc[data["workday"]==1, "casual"].mean()

# a workday, casual median
med_casual_work = data.loc[data["workday"]==1, "casual"].median()

# cmp by mean
registered_support_days = (data.loc[data["workday"]==1, "registered"] > avg_registered_nonwork).sum()
casual_support_days = (data.loc[data["workday"]==0, "casual"] > avg_casual_work).sum()

print("sum of registered support work > no work: ", registered_support_days)
print("sum of casual support no work > work: ", casual_support_days)

# cmp by median
registered_support_days = (data.loc[data["workday"]==1, "registered"] > med_registered_nonwork).sum()
casual_support_days = (data.loc[data["workday"]==0, "casual"] > med_casual_work).sum()
print("sum of registered support work > no work: ", registered_support_days)
print("sum of casual support no work > work: ", casual_support_days)