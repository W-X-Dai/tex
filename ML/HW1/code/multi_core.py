import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
from joblib import Parallel, delayed
import os

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
    x = np.array(x)
    valid_indices = ~np.isnan(x)
    x = x[valid_indices]
    y = y[valid_indices]

    if len(np.unique(y)) < 2:
        print("Warning: Only one class present in evaluation. AUC is undefined.")
        return 0.5

    thresh = np.sort(np.unique(x))[::-1]
    fpr_list, tpr_list = [], []
    neg, pos = np.sum(y == 0), np.sum(y == 1)

    if pos == 0 or neg == 0:
        print("Warning: Only one class present after filtering. AUC is undefined.")
        return 0.5

    for t in thresh:
        y_pred = (x>=t).astype(int)
        tp, fp = np.sum((y_pred == 1) & (y == 1)), np.sum((y_pred == 1) & (y == 0))
        tpr, fpr = tp/pos, fp/neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    tpr_list = [0]+tpr_list+[1]
    fpr_list = [0]+fpr_list+[1]

    auc = np.trapezoid(tpr_list, fpr_list)

    if diagram == 1:
        plt.figure()
        plt.plot(fpr_list, tpr_list, marker='o')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid()
        plt.savefig('roc_curve.png', dpi=400)
        print("ROC Curve is saved as roc_curve.png")
    return auc

def log_sum_exp(a, b):
    m = np.maximum(a, b)
    diff_a = a - m
    diff_b = b - m
    term1 = np.exp(diff_a) if diff_a > -np.inf else 0
    term2 = np.exp(diff_b) if diff_b > -np.inf else 0
    return m + np.log(term1 + term2)

def get_parameters(data: Dataset):
    """get parameters with feature selection mask"""
    # get data
    X_train_all, _, y_train, _, selected_features = get_data(data)
    X_train = X_train_all[:, selected_features]

    # prior probability
    n_pos, n_neg = 0, 0
    for i in range(len(y_train)):
        n_pos += (y_train[i] == 1)
        n_neg += (y_train[i] == 0)

    n = n_pos+n_neg
    
    if n_pos == 0 or n_neg == 0:
        print("Warning: Training fold contains only one class.")
        return None

    n_features = X_train.shape[1]
    p0, p1 = n_neg/n, n_pos/n
    # print(p0, p1, n)
    # print(X_train.shape, y_train.shape) 

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

    reg = 1e-6 
    cm0 += np.eye(cm0.shape[0]) * reg
    cm1 += np.eye(cm1.shape[0]) * reg

    return p0, p1, mu0, mu1, cm0, cm1

def estimate(data: Dataset):
    X_train_all, X_test_all, y_train, y_test, selected_features = get_data(data)
    
    params = get_parameters(data)
    if params is None:
        return 0.0, 0.5 

    p0, p1, mu0, mu1, cm0, cm1 = params

    X_train = X_train_all[:, selected_features]
    X_test = X_test_all[:, selected_features]

    try:
        inv0 = np.linalg.inv(cm0)
        inv1 = np.linalg.inv(cm1)
        det0 = np.linalg.det(cm0)
        det1 = np.linalg.det(cm1)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix encountered despite regularization.")
        return 0.0, 0.5

    d = X_train.shape[1] 
    mu0_v = mu0.reshape(-1, 1)
    mu1_v = mu1.reshape(-1, 1)

    # train scores
    scores = []
    for x_vec in X_train:
        x = x_vec.reshape(-1, 1)

        # log likelihood
        log_px0 = -0.5 * np.log((2*np.pi)**d * det0) - 0.5 * ((x - mu0_v).T @ inv0 @ (x - mu0_v))
        log_px1 = -0.5 * np.log((2*np.pi)**d * det1) - 0.5 * ((x - mu1_v).T @ inv1 @ (x - mu1_v))

        log_post1 = np.log(p1) + log_px1
        log_post0 = np.log(p0) + log_px0
        log_den = log_sum_exp(log_post1, log_post0)
        p1_x = np.exp(log_post1 - log_den)
        
        scores.append(p1_x.item())
        
    performance = evaluation(x=scores, y=y_train)

    # test posterior
    x = X_test.reshape(-1, 1)
    log_px0 = -0.5 * np.log((2*np.pi)**d * det0) - 0.5 * ((x - mu0_v).T @ inv0 @ (x - mu0_v))
    log_px1 = -0.5 * np.log((2*np.pi)**d * det1) - 0.5 * ((x - mu1_v).T @ inv1 @ (x - mu1_v))
    
    log_post1_test = np.log(p1) + log_px1
    log_post0_test = np.log(p0) + log_px0
    log_den_test = log_sum_exp(log_post1_test, log_post0_test)
    test_prior = np.exp(log_post1_test - log_den_test).item()

    return performance, test_prior
    
def feature_selection(data: Dataset):
    n_features = data.X_test.shape[1]
    
    best_mask = np.zeros(n_features, dtype=bool)
    best_auc = 0.0
    best_prior_for_best_mask = 0.5

    n = 0
    while n < n_features:
        local_best_mask = best_mask.copy()
        local_best_auc = -1.0
        local_best_prior = 0.5

        for i in range(n_features):
            if best_mask[i] == True:
                continue

            local_mask = best_mask.copy()
            local_mask[i] = True 

            local_data = deepcopy(data)
            local_data.feature_mask = local_mask

            local_auc, test_prior_for_this_mask = estimate(local_data)
            
            if local_auc > local_best_auc:
                local_best_auc = local_auc
                local_best_mask = local_mask
                local_best_prior = test_prior_for_this_mask

        if local_best_auc <= best_auc:
            break
        else:
            best_auc = local_best_auc
            best_mask = local_best_mask
            best_prior_for_best_mask = local_best_prior
            n += 1

    return best_auc, best_mask, best_prior_for_best_mask

def process_fold(i, data_df):
    print(f"--- Processing LOOCV Fold {i+1}/{data_df.shape[0]} ---")
    test = data_df.iloc[[i]]
    train = data_df.drop(index=i)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    X_train = train.drop(columns = ['SeqNum', 'GroundTruth',  'Gender']).to_numpy()
    X_test = test.drop(columns = ['SeqNum', 'GroundTruth',  'Gender']).to_numpy()

    y_train = train['GroundTruth'].to_numpy()
    y_test = test['GroundTruth'].to_numpy()
    
    fold_data = Dataset(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test
    )
    
    auc, mask, pr = feature_selection(fold_data)
    
    print(f"Fold {i+1} Test Posterior: {pr:.4f} (True Label: {y_test[0]})")
    print(f"Fold {i+1} Selected {np.sum(mask)} features.")
    
    return pr, mask.astype(int)

def loocv(data_df):
    n_samples = data_df.shape[0]
    n_features = data_df.shape[1] - len(['SeqNum', 'GroundTruth', 'Gender'])
    
    prior = np.zeros(shape=n_samples, dtype=float)
    mask_stat = np.zeros(shape=n_features, dtype=int)
    
    all_y = data_df['GroundTruth'].to_numpy()
    
    cpu_count = os.cpu_count()
    n_cores = cpu_count - 2 if cpu_count and cpu_count > 2 else 1
    print(f"--- Starting LOOCV on {n_cores} cores using joblib ---")
    
    results = Parallel(n_jobs=n_cores)(
        delayed(process_fold)(i, data_df) for i in range(n_samples)
    )
    
    print("\n--- LOOCV Finished (All folds complete) ---")

    for i in range(n_samples):
        pr, mask = results[i]
        prior[i] = pr
        mask_stat += mask
        
    print(f"Final posterior probabilities for each sample:\n{prior}")
    print(f"Feature selection frequency mask:\n{mask_stat}")
    
    print("\n--- Overall Model Performance (from LOOCV posteriors) ---")
    overall_auc = evaluation(x=prior, y=all_y, diagram=1)
    print(f"Overall LOOCV AUC: {overall_auc:.4f}")

if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print("Data loaded successfully:")
    print(df.head())
    print("Starting Leave-One-Out Cross-Validation...")
    loocv(df)