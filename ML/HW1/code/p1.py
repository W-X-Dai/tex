import pandas as pd

df = pd.read_excel('AcromegalyFeatureSet.xlsx')
print(df.head())
df = df.iloc[1:].reset_index(drop=True)

print(df.head())


# prior probability
n_pos = len(df[df['GroundTruth'] == 1])
n_neg = len(df[df['GroundTruth'] == 0])
n = n_pos+n_neg
p_pos, p_neg = n_pos/n, n_neg/n

print(p_pos, p_neg, n)
