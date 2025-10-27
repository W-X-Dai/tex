"""using numpy to accelerate computing with multi core allowed, using aic to select features"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

# multi core
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
        print("[ERROR] Only one class present in evaluation. AUC is undefined.")
        return 0.5 # or np.nan
        
    fpr_list, tpr_list = [], []
    neg, pos = np.sum(y == 0), np.sum(y == 1)

    if pos == 0 or neg == 0:
        print("[ERROR] Only one class present after filtering. AUC is undefined.")
        return 0.5 # or np.nan
    
    order = np.argsort(-x)
    x_sorted = x[order]
    y_sorted = y[order]

    pos = np.sum(y_sorted == 1)
    neg = np.sum(y_sorted == 0)
    
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    idx = np.where(np.diff(x_sorted, prepend=np.inf) != 0)[0]

    tpr_list = tp_cum[idx] / pos
    fpr_list = fp_cum[idx] / neg

    # (0, 0) and (0, 1)
    tpr_list = np.concatenate(([0], tpr_list, [1]))
    fpr_list = np.concatenate(([0], fpr_list, [1]))

    auc = np.trapz(tpr_list, fpr_list)

    if diagram == 1:
        plt.figure()
        plt.plot(fpr_list, tpr_list, marker='o')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid()
        plt.savefig('roc_curve.png', dpi=400)
        print("[INFO] ROC Curve is saved as roc_curve.png")
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
    counts = np.bincount(y_train.astype(int))
    n_neg = counts[0] if len(counts) > 0 else 0
    n_pos = counts[1] if len(counts) > 1 else 0

    n = n_pos+n_neg
    
    if n_pos == 0 or n_neg == 0:
        print("[ERROR] Training fold contains only one class.")
        return None

    p0, p1 = n_neg/n, n_pos/n
    # print(p0, p1, n)
    # print(X_train.shape, y_train.shape) 

    # mean, shape[0]: rows, shape[1]: features
    mask0 = (y_train == 0)
    mask1 = (y_train == 1)
    mu0 = X_train[mask0].mean(axis=0)
    mu1 = X_train[mask1].mean(axis=0)
    
    # co-matrix
    X0_centered = X_train[mask0] - mu0
    X1_centered = X_train[mask1] - mu1

    cm0 = (X0_centered.T @ X0_centered) / n_neg
    cm1 = (X1_centered.T @ X1_centered) / n_pos

    reg = 1e-6 
    cm0 += np.eye(cm0.shape[0]) * reg
    cm1 += np.eye(cm1.shape[0]) * reg

    return p0, p1, mu0, mu1, cm0, cm1

def estimate(data: Dataset):
    X_train_all, X_test_all, y_train, y_test, selected_features = get_data(data)
    
    params = get_parameters(data)
    if params is None:
        return -np.inf, 0.5
    p0, p1, mu0, mu1, cm0, cm1 = params

    X_train = X_train_all[:, selected_features]
    X_test = X_test_all[:, selected_features]

    try:
        inv0 = np.linalg.inv(cm0)
        inv1 = np.linalg.inv(cm1)
        det0 = np.linalg.det(cm0)
        det1 = np.linalg.det(cm1)
    except np.linalg.LinAlgError:
        print("[ERROR] Singular matrix encountered despite regularization.")
        return -np.inf, 0.5

    d = X_train.shape[1]
    mu0_v = mu0.reshape(-1, 1)
    mu1_v = mu1.reshape(-1, 1)

    const0 = -0.5 * (d * np.log(2 * np.pi) + np.log(det0))
    const1 = -0.5 * (d * np.log(2 * np.pi) + np.log(det1))

    # score
    Xc0 = X_train - mu0
    Xc1 = X_train - mu1

    log_px0 = const0 - 0.5 * np.sum((Xc0 @ inv0) * Xc0, axis=1)
    log_px1 = const1 - 0.5 * np.sum((Xc1 @ inv1) * Xc1, axis=1)

    # posterior
    log_post0 = np.log(p0) + log_px0
    log_post1 = np.log(p1) + log_px1
    log_den = np.logaddexp(log_post0, log_post1)
    

    total_log_likelihood = np.sum(log_den)
    k = d**2 + 3*d + 1
    aic = 2 * k - 2 * total_log_likelihood
    performance = aic

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
    best_auc = -np.inf
    best_prior_for_best_mask = 0.5

    n = 0
    while n < n_features:
        local_best_mask = best_mask.copy()
        local_best_auc = -np.inf
        local_best_prior = 0.5

        for i in range(n_features):
            if best_mask[i] == 1:
                continue

            local_mask = best_mask.copy()
            local_mask[i] = 1

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

def plot_bivariate_decision_boundary(data_df, feature1_name, feature2_name):
    """This section is helped by AI"""
    print(f"[INFO] Plotting decision boundary for: {feature1_name} and {feature2_name}")
    
    X = data_df[[feature1_name, feature2_name]].to_numpy()
    y = data_df['GroundTruth'].to_numpy()
    d = X.shape[1]
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    n = n_pos + n_neg
    p1 = n_pos / n
    p0 = n_neg / n
    
    mu0 = np.zeros(shape=d, dtype=float)
    mu1 = np.zeros(shape=d, dtype=float)
    for i in range(n):
        mu0 += (y[i] == 0) * X[i]
        mu1 += (y[i] == 1) * X[i]
    mu0 /= n_neg
    mu1 /= n_pos
    
    cm0 = np.zeros(shape=(d, d), dtype=float)
    cm1 = np.zeros(shape=(d, d), dtype=float)
    for t in range(n):
        if y[t] == 0:
            diff = (X[t] - mu0).reshape(-1, 1)
            cm0 += diff @ diff.T
        else:
            diff = (X[t] - mu1).reshape(-1, 1)
            cm1 += diff @ diff.T
    cm0 /= n_neg
    cm1 /= n_pos
    
    reg = 1e-6
    cm0 += np.eye(d) * reg
    cm1 += np.eye(d) * reg

    try:
        inv_cm0 = np.linalg.inv(cm0)
        inv_cm1 = np.linalg.inv(cm1)
        det_cm0 = np.linalg.det(cm0)
        det_cm1 = np.linalg.det(cm1)
    except np.linalg.LinAlgError:
        print("[ERROR] Singular matrix during plotting.")
        return

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    def gaussian_pdf(X_grid, mu, inv_C, det_C):
        coeff = 1.0 / (np.sqrt((2 * np.pi)**d * det_C))
        diff = X_grid - mu
        exponent = -0.5 * np.sum((diff @ inv_C) * diff, axis=1)
        return coeff * np.exp(exponent)

    Z0_like = gaussian_pdf(grid_points, mu0, inv_cm0, det_cm0)
    Z1_like = gaussian_pdf(grid_points, mu1, inv_cm1, det_cm1)
    
    Z_boundary = (Z1_like * p1) - (Z0_like * p0)
    
    Z0_plot = Z0_like.reshape(xx.shape)
    Z1_plot = Z1_like.reshape(xx.shape)
    Z_boundary_plot = Z_boundary.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    
    plt.scatter(X[y==0, 0], X[y==0, 1], 
                marker=MarkerStyle('o'), c='blue', alpha=0.7, label='Class 0 (Circles)')
    plt.scatter(X[y==1, 0], X[y==1, 1], 
                marker=MarkerStyle('+'), c='red', alpha=0.7, label='Class 1 (Plus Signs)')
    
    plt.contour(xx, yy, Z0_plot, colors='blue', linestyles='--', alpha=0.5)
    plt.contour(xx, yy, Z1_plot, colors='red', linestyles=':', alpha=0.5)
    
    plt.contour(xx, yy, Z_boundary_plot, levels=[0], colors='black', linewidths=2)
    
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title('Bivariate Gaussian Decision Boundary and Contours')
    plt.legend()
    plt.grid(True)
    plt.savefig('decision_boundary_plot.png', dpi=400)
    print("[INFO] Decision boundary plot saved as 'decision_boundary_plot.png'")
    plt.show()

def process_fold(i, data, feature_names):
    """one fold of LOOCV that can be parallelized"""
    print(f"LOOCV Fold {i+1}/{data.shape[0]}")
    test = data.iloc[[i]]
    train = data.drop(index=i)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    X_train = train.drop(columns=['SeqNum', 'GroundTruth', 'Gender']).to_numpy()
    X_test = test.drop(columns=['SeqNum', 'GroundTruth', 'Gender']).to_numpy()

    y_train = train['GroundTruth'].to_numpy()
    y_test = test['GroundTruth'].to_numpy()

    fold_data = Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    auc, mask, pr = feature_selection(fold_data)

    selected_feature_names = [name for name, m in zip(feature_names, mask) if m]
    print(f"Fold {i+1} Selected Features: {selected_feature_names}")
    print(f"Fold {i+1} Test Posterior: {pr:.4f} (True Label: {y_test[0]}) with {np.sum(mask)} features.")

    return auc, mask, pr, selected_feature_names, y_test[0]


def loocv(data): 
    non_feature_cols = ['SeqNum', 'GroundTruth', 'Gender']
    feature_names = data.drop(columns=non_feature_cols).columns.to_list()
    n_features = len(feature_names)    
    n_samples = data.shape[0]

    prior = np.zeros(shape=n_samples, dtype=float)
    mask_stat = np.zeros(shape=n_features, dtype=int)
    all_y = data['GroundTruth'].to_numpy()
    selected_features_per_fold = []

    # multi core
    n_cores = max(1, os.cpu_count() - 2) #type: ignore
    print(f"[INFO] Starting LOOCV on {n_cores} cores (multi-core acceleration enabled) ...")

    results = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(process_fold)(i, data, feature_names) for i in range(n_samples)
    )

    # result
    for i, (auc, mask, pr, selected_feats, y_test) in enumerate(results): #type: ignore
        prior[i] = pr
        mask_stat += mask.astype(int)
        selected_features_per_fold.append(selected_feats)

    print("\n[INFO] LOOCV Finished")
    print(f"- Final posterior probabilities for each sample:\n{prior}")

    print("\n- Feature sets per fold:")    
    for i, feats in enumerate(selected_features_per_fold, 1):
        print(f"Fold {i}: {feats}")

    print(f"- Feature selection frequency mask:\n{mask_stat}")
    
    feature_counts_series = pd.Series(mask_stat, index=feature_names)
    sorted_features = feature_counts_series.sort_values(ascending=False)
    
    print("\n- Feature Selection Count:")
    print(sorted_features)

    print("\n- Overall Model Performance (from LOOCV posteriors):")
    overall_auc = evaluation(x=prior, y=all_y, diagram=1)
    thresh = 0.5
    y_pred, y_true = (prior >= thresh).astype(int), all_y
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    total_pos = tp + fn
    total_neg = tn + fp

    accuracy = (tp + tn) / n_samples
    sensitivity = tp / total_pos if total_pos > 0 else 0.0
    specificity = tn / total_neg if total_neg > 0 else 0.0

    print(f"\n Metrics (using threshold = {thresh}):")
    print(f"    Accuracy:    {accuracy:.4f} ({(tp+tn)}/{n_samples})")
    print(f"    Sensitivity: {sensitivity:.4f} ({tp}/{total_pos})")
    print(f"    Specificity: {specificity:.4f} ({tn}/{total_neg})")
    print(f" Area Under Curve: {overall_auc:.4f}")

    top_feature_1 = sorted_features.index[0]
    top_feature_2 = sorted_features.index[1]
    print("\n- Tie-breaking Strategy")
    print("[INFO] Select the top 2 features by frequency count.")
    
    if len(sorted_features) > 2 and sorted_features.iloc[1] == sorted_features.iloc[2]:
        print(f"[INFO] A tie was detected for second place at {sorted_features.iloc[1]} counts.")
        print(f"[INFO] Selected '{top_feature_2}' based on default sort order (alphabetical).")
    else:
        print("[INFO] No tie detected for the top 2 features.")

    print(f"[INFO] Automatically selected features: {top_feature_1} (Count: {sorted_features.iloc[0]}), {top_feature_2} (Count: {sorted_features.iloc[1]})")

    plot_bivariate_decision_boundary(data, top_feature_1, top_feature_2)

def information():
    print("This module performs Leave-One-Out Cross-Validation (LOOCV) with multi-core acceleration for a Gaussian classifier with feature selection.")
    print("It evaluates model performance using AUC and plots decision boundaries for the top selected features.")

if __name__ == '__main__':
    information()
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print("[INFO] Data loaded successfully:")
    print(df.head())
    print("[INFO] Starting Leave-One-Out Cross-Validation...")
    np.random.seed(98)
    loocv(df)