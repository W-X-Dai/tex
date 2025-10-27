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
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_mask: Optional[np.ndarray] = None

def get_data(data: Dataset):
    return data.X_train, data.X_test, data.y_train, data.y_test, data.feature_mask

# x is the value, y is the label
def evaluation(x, y, diagram=0):
    """evaluate model with auc, diagram = 1 means sava the roc curve"""
    thresh = np.sort(np.unique(x))[::-1]
    fpr_list, tpr_list = [], []
    neg, pos = np.sum(y == 0), np.sum(y == 1)

    for t in thresh:
        y_pred = (x>=t).astype(int)
        tp, fp = np.sum((y_pred == 1) & (y == 1)), np.sum((y_pred == 1) & (y == 0))
        tpr, fpr = tp/pos, fp/neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    tpr_list = [0]+tpr_list+[1]
    fpr_list = [0]+fpr_list+[1]

    auc = np.trapz(tpr_list, fpr_list)
    print(f"AUC = {auc:.4f}")

    if diagram == 1:
        plt.plot(fpr_list, tpr_list, marker='o')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid()
        plt.savefig('roc_curve.png', dpi=400)
        print("ROC Curve is saved as roc_curve.png")
    return auc

def get_parameters(data: Dataset):
    """get parameters with feature selection mask"""
    # get data
    X_train, _, y_train, _, selected_features = get_data(data)
    X_train = X_train[:, selected_features]

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

    return p0, p1, mu0, mu1, cm0, cm1

def estimate(data: Dataset):
    X_train, X_test, y_train, y_test, _ = get_data(data)
    p0, p1, mu0, mu1, cm0, cm1 = get_parameters(data)

    inv0 = np.linalg.inv(cm0)
    inv1 = np.linalg.inv(cm1)
    det0 = np.linalg.det(cm0)
    det1 = np.linalg.det(cm1)
    d = X_train.shape[1]
    mu0_v = mu0.reshape(-1, 1)
    mu1_v = mu1.reshape(-1, 1)

    # train scores
    scores = []
    for x in X_train:
        x = x.reshape(-1, 1)

        # log likelihood
        log_px0 = -0.5 * np.log((2*np.pi)**d * det0) - 0.5 * ((x - mu0_v).T @ inv0 @ (x - mu0_v))
        log_px1 = -0.5 * np.log((2*np.pi)**d * det1) - 0.5 * ((x - mu1_v).T @ inv1 @ (x - mu1_v))

        # posterior
        log_num = np.log(p1) + log_px1
        log_den = np.log(np.exp(np.log(p1) + log_px1) + np.exp(np.log(p0) + log_px0))
        p1_x = np.exp(log_num - log_den)
        scores.append(p1_x.item())
    performance = evaluation(x=scores, y=y_train)

    # test posterior
    x = X_test.reshape(-1, 1)
    log_px0 = -0.5 * np.log((2*np.pi)**d * det0) - 0.5 * ((x - mu0_v).T @ inv0 @ (x - mu0_v))
    log_px1 = -0.5 * np.log((2*np.pi)**d * det1) - 0.5 * ((x - mu1_v).T @ inv1 @ (x - mu1_v))
    log_num = np.log(p1) + log_px1
    log_den = np.log(np.exp(np.log(p1) + log_px1) + np.exp(np.log(p0) + log_px0))
    test_prior = np.exp(log_num - log_den).item()

    return performance, test_prior
    
def feature_selection(data: Dataset):
    n_features = data.X_test.shape[1]

    best_mask = np.zeros(n_features, dtype=int)
    best_auc = 0.0
    best_prior = 0.0

    n = 0
    while n < n_features:
        local_best_mask = best_mask.copy()
        local_best_auc = -1.0

        for i in range(n_features):
            if best_mask[i] == 1:
                continue

            local_mask = best_mask.copy()
            local_mask[i] = 1

            local_data = deepcopy(data)
            local_data.feature_mask = local_mask

            local_auc, best_prior = estimate(local_data)
            if local_auc > local_best_auc:
                local_best_auc = local_auc
                local_best_mask = local_mask

        if local_best_auc <= best_auc:
            break
        else:
            best_auc = local_best_auc
            best_mask = local_best_mask
            n += 1

    return best_auc, best_mask, best_prior

def loocv(data):
    prior = np.zeros(shape=data.X_train[0], dtype=float)
    mask_stat = np.zeros(shape=data.X_train[1], dtype=int)
    for i in range(len(data)):
        test = data.iloc[[i]]
        train = data.drop(index=i)

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
        # best auc, mask, and prior of test data under the selected mask
        auc, mask, pr = feature_selection(data)
        prior[i] = pr
        mask_stat += mask

        

if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print(df.head())
    loocv(df)