"""using numpy to accelerate computing"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def get_data(data: Dataset):
    return data.X_train, data.X_test, data.y_train, data.y_test

# X: N*d, X_T: d*N
def PCA(X: np.ndarray, thresh=0.95):
    """perform PCA on data X, return projection matrix W"""
    # covariance = X^T X / n
    XTX = (X.T @ X) / X.shape[0]   # d*d

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(XTX)

    # sort
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    # numerical stabilization
    eigvals = np.maximum(eigvals, 0)

    # variance explained
    total_var = np.sum(eigvals)
    var_exp = np.cumsum(eigvals) / total_var

    n_components = np.searchsorted(var_exp, thresh) + 1
    print(f"[INFO] selected {n_components} principal components to retain {var_exp[n_components-1]*100:.2f}% variance.")

    return eigvecs[:, :n_components]

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
    """get parameters with all projected training data"""
    # get data
    # N*k _ N*1 _
    X_train, _, y_train, _ = get_data(data)

    # prior probability
    counts = np.bincount(y_train.astype(int))
    n_neg = counts[0] if len(counts) > 0 else 0
    n_pos = counts[1] if len(counts) > 1 else 0

    n = n_pos+n_neg
    
    if n_pos == 0 or n_neg == 0:
        print("[ERROR] Training fold contains only one class.")
        return None

    n_features = X_train.shape[1]
    p0, p1 = n_neg/n, n_pos/n
    # print(p0, p1, n)
    # print(X_train.shape, y_train.shape) 

    # mean, shape[0]: rows, shape[1]: features
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)
    
    # co-matrix
    X0_centered = X_train[y_train == 0] - mu0
    X1_centered = X_train[y_train == 1] - mu1

    cm0 = (X0_centered.T @ X0_centered) / n_neg
    cm1 = (X1_centered.T @ X1_centered) / n_pos

    reg = 1e-6 
    cm0 += np.eye(cm0.shape[0]) * reg
    cm1 += np.eye(cm1.shape[0]) * reg

    return p0, p1, mu0, mu1, cm0, cm1

def estimate(data: Dataset):
    # N*k 1*k N*1 1*1
    X_train, X_test, y_train, y_test = get_data(data)
    
    params = get_parameters(data)
    if params is None:
        return 0.0, 0.5
    p0, p1, mu0, mu1, cm0, cm1 = params

    try:
        inv0 = np.linalg.inv(cm0)
        inv1 = np.linalg.inv(cm1)
        det0 = np.linalg.det(cm0)
        det1 = np.linalg.det(cm1)
    except np.linalg.LinAlgError:
        print("[ERROR] Singular matrix encountered despite regularization.")
        return 0.0, 0.

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
    
    scores = np.exp(log_post1 - log_den)
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

def loocv(data, thresh=0.95): 
    print(f"start with thresh={thresh}")
    n_samples = data.shape[0]
    prior = np.zeros(shape=n_samples, dtype=float)
    all_y = data['GroundTruth'].to_numpy()

    for i in range(n_samples):
        print(f"LOOCV Fold {i+1}/{n_samples}")
        test = data.iloc[[i]]
        train = data.drop(index=i)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # origin data
        X_train_raw = train.drop(columns=['SeqNum','GroundTruth','Gender']).to_numpy()
        X_test_raw  = test.drop(columns=['SeqNum','GroundTruth','Gender']).to_numpy()

        std, mu = X_train_raw.std(axis=0), X_train_raw.mean(axis=0)
        std[std == 0] = 1.0
        X_train_std = (X_train_raw - mu) / std
        X_test_std  = (X_test_raw  - mu) / std

        pca = PCA(X_train_std, thresh=thresh) # d×k

        X_train_centered = X_train_std # N×d
        X_test_centered  = X_test_std # 1×d

        # projected data
        Z_train = X_train_centered @ pca # N×k
        Z_test  = X_test_centered  @ pca # 1×k

        y_train = train['GroundTruth'].to_numpy()
        y_test = test['GroundTruth'].to_numpy()
        
        fold_data = Dataset(
            X_train = Z_train, # N×k
            X_test = Z_test, # 1×k
            y_train = y_train, # N×1
            y_test = y_test # 1×1
        )
        # best auc, mask, and prior of test data under the selected mask
        
        perf, pr = estimate(fold_data)

        print(f"Fold {i+1} Training AUC: {perf:.4f}")

        prior[i] = pr
        
        print(f"Fold {i+1} Test Posterior: {pr:.4f} (True Label: {y_test[0]})")
        
    # show the results
    print("\n[INFO] LOOCV Finished")
    # print(f"- Final posterior probabilities for each sample:\n{prior}")

    # show the performance
    print("\n- Overall Model Performance (from LOOCV posteriors)")
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

    print(f"\nMetrics (using threshold = {thresh}):")
    print(f"    Accuracy:    {accuracy:.4f} ({(tp+tn)}/{n_samples})")
    print(f"    Sensitivity: {sensitivity:.4f} ({tp}/{total_pos})")
    print(f"    Specificity: {specificity:.4f} ({tn}/{total_neg})")
    print(f"Area Under Curve: {overall_auc:.4f}")
    
if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print("[INFO] Data loaded successfully:")
    print(df.head())
    print("[INFO] Starting Leave-One-Out Cross-Validation...")
    np.random.seed(98)

    thresh = 0.95
    loocv(df, thresh=thresh)