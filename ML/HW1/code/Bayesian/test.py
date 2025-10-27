import pandas as pd

df = pd.read_excel('AcromegalyFeatureSet.xlsx')
print(df.head())


X = df.drop(columns=['SeqNum', 'GroundTruth',  'Gender']).to_numpy()
y = df['GroundTruth'].to_numpy()
print(X.shape, y.shape)

# get parameters
n_pos = len(df[df['GroundTruth'] == 1])
n_neg = len(df[df['GroundTruth'] == 0])
n = n_pos+n_neg
p_pos, p_neg = n_pos/n, n_neg/n
print(p_pos, p_neg, n)
