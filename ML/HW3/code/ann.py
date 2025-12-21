"""using numpy to accelerate computing"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from myann import ANN # type: ignore

@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def Network(data: Dataset, LR=0.1, epochs=500):
    # N*k
    X_train, X_test, y_train, y_test = get_data(data)
    N, d = X_train.shape


    # model
    ann=ANN(lr=LR)
    ann.add_linear(d, 20)
    ann.add_relu()
    ann.add_linear(20, 10)
    ann.add_relu()
    ann.add_linear(10, 1)
    ann.add_sigmoid()

    for e in range(epochs):
        epoch_loss = 0.0
        for i in range(N):
            x = X_train[i].astype(float)
            y = float(y_train[i])

            out = ann.forward(x)
            y_pred = float(out[0])

            loss = ann.bce(y, y_pred)
            epoch_loss += loss

            ann.backward(y, y_pred)
            ann.update()

        # if(e==epochs-1):
        #     print(f"[epoch {e}], Loss: {epoch_loss / N:.4f}")

    train_scores = np.zeros(N, dtype=float)
    for i in range(N):
        x = X_train[i].astype(float)
        out = ann.forward(x)
        train_scores[i] = float(out[0])

    train_auc = evaluation(x=train_scores, y=y_train)

    x_test = X_test[0].astype(float)
    out_test = ann.forward(x_test)
    test_prob = float(out_test[0])

    return train_auc, test_prob

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

def loocv(data, thresh=0.95): 
    print(f"start with thresh={thresh}")
    n_samples = data.shape[0]
    prior = np.zeros(shape=n_samples, dtype=float)
    all_y = data['GroundTruth'].to_numpy()
    train_auc = 0.0

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
        
        perf, pr = Network(data=fold_data, LR=0.005, epochs=700)

        print(f"Fold {i+1} Training AUC: {perf:.4f}")
        train_auc+=perf

        prior[i] = pr
        
        print(f"Fold {i+1} Test Probability: {pr:.4f} (True Label: {y_test[0]})")
        
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
    print(f"Test AUC: {overall_auc:.4f}")
    print(f"Train AUC: {train_auc/n_samples:.4f}")
if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print("[INFO] Data loaded successfully:")
    print(df.head())
    print("[INFO] Starting Leave-One-Out Cross-Validation...")
    np.random.seed(98)

    thresh = 0.95
    loocv(df, thresh=thresh)