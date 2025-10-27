"""
Steps: 
1. group samples
2. calculate parameters
    - prior probability
    - mean
    - co matrix
3. calculate discriminant function g
4. if g1(x)<g2(x), x would be estimates as class 2

select feature -> kfold, get_parameters -> performance evaluation -> repeat
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_mask: Optional[np.ndarray] = None

def get_data(data):
    return data.X_train, data.X_test, data.y_train, data.y_test, data.feature_mask

def evaluation(predict, answer):
    # confusion matrix
    tp = np.sum((predict == 1) & (answer == 1))
    fp = np.sum((predict == 1) & (answer == 0))
    tn = np.sum((predict == 0) & (answer == 0))
    fn = np.sum((predict == 0) & (answer == 1))
    # todo: using g value to draw roc



def get_parameters(data):
    # get data
    X_train, _, y_train, _, selected_features = get_data(data)
    X_train = X_train[:, selected_features]
    y_train = y_train[:, selected_features]

    # prior probability
    n_pos, n_neg = 0, 0
    for i in range(len(y_train)):
        n_pos += (y_train[i] == 1)
        n_neg += (y_train[i] == 0)

    n = n_pos+n_neg
    n_features = X_train.shape[1]
    p0, p1 = n_neg/n, n_pos/n
    print(p0, p1, n)
    print(X_train.shape, y_train.shape) 

    # mean, shape[0]: rows, shape[1]: features
    mu0 = np.zeros(shape = n_features, dtype=float)
    mu1 = np.zeros(shape = n_features, dtype=float)
    for feature in range(n_features):
        for i in range(n):
            mu0[feature] += (y_train[i] == 0)*X_train[i][feature]/n_neg
            mu1[feature] += (y_train[i] == 1)*X_train[i][feature]/n_pos

    # co-matrix
    cm0 = np.zeros(shape = (n_features, n_features), dtype=float)
    cm1 = np.zeros(shape = (n_features, n_features), dtype=float)
    for i in range(n_features):
        for j in range(n_features):
            for t in range(n):
                cm0[i][j] += (y_train[t] == 0)*(X_train[t][i]-mu0[i])*(X_train[t][j]-mu0[j])
                cm1[i][j] += (y_train[t] == 1)*(X_train[t][i]-mu1[i])*(X_train[t][j]-mu1[j])
    cm0 /= n_neg
    cm1 /= n_pos


def estimate(data):
    X_train, y_train, X_test, y_test, _ = get_data(data)
    get_parameters(data)

def feature_selection(n_features):
    n_selected_features = 0
    best_auc = 0
    test_result = estimate
    if n_selected_features > 2 :
        pass
    

def loocv(data):
    for i in range(len(data)):
        test = data.iloc[[i]]
        train = data.drop(index = i)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        X_train = train.drop(columns = ['SeqNum', 'GroundTruth',  'Gender']).to_numpy()
        X_test = test.drop(columns = ['SeqNum', 'GroundTruth',  'Gender']).to_numpy()

        y_train = train['GroundTruth'].to_numpy()
        y_test = test['GroundTruth'].to_numpy()
        data = Dataset(
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test
        )
        estimate(data)    
        

if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print(df.head())
    loocv(df)