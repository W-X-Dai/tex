"""This program is used to plot PCA scree graph and 2D projection with Gaussian contours and decision boundary. It's modified from numpy_accr.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.markers import MarkerStyle


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def get_data(data: Dataset):
    return data.X_train, data.X_test, data.y_train, data.y_test

def plot_pca_scree(eigvals, var_exp):
    d = len(eigvals)
    x = np.arange(1, d+1)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x, eigvals, marker='x')
    plt.title("(a) Scree graph")
    plt.xlabel("Eigenvectors")
    plt.ylabel("Eigenvalues")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(x, var_exp, marker='+')
    plt.title("(b) Proportion of variance explained")
    plt.xlabel("Eigenvectors")
    plt.ylabel("Prop of Var")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("pca_scree_plot.png", dpi=400)
    print("[INFO] PCA scree plot saved as pca_scree_plot.png")

def plot_PCA_2D_gaussian(df):
    """plot PCA 2D projection with Gaussian contours and decision boundary, this section is helped by ChatGPT"""
    feature_cols = [c for c in df.columns if c not in ['SeqNum', 'GroundTruth', 'Gender']]
    X = df[feature_cols].to_numpy().astype(float)
    y = df['GroundTruth'].to_numpy().astype(int)

    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mu) / std

    eigvecs = PCA(X_std)
    W2 = eigvecs[:, :2]
    Z = X_std @ W2

    # 用投影後的 Z 來算 Gaussian 參數
    proj_dataset = Dataset(
        X_train=Z,
        X_test=Z,
        y_train=y,
        y_test=y
    )

    params = get_parameters(proj_dataset)

    if params is None:
        print("[ERROR] Cannot plot due to insufficient class data.")
        return

    p0, p1, mu0, mu1, S0, S1 = params

    def gaussian_pdf(xx, yy, mu, cov):
        pos = np.dstack((xx, yy))
        diff = pos - mu
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, inv, diff)
        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        return norm * np.exp(exponent)

    # QDA decision function
    def g_diff(xx, yy):
        pos = np.dstack((xx, yy))
        diff0 = pos - mu0
        diff1 = pos - mu1

        inv0 = np.linalg.inv(S0)
        inv1 = np.linalg.inv(S1)

        part0 = -0.5 * np.einsum('...i,ij,...j->...', diff0, inv0, diff0) \
                - 0.5 * np.log(np.linalg.det(S0)) + np.log(p0)
        part1 = -0.5 * np.einsum('...i,ij,...j->...', diff1, inv1, diff1) \
                - 0.5 * np.log(np.linalg.det(S1)) + np.log(p1)

        return part1 - part0       # g1(x) - g0(x)

    x_min, x_max = Z[:, 0].min() - 1, Z[:, 0].max() + 1
    y_min, y_max = Z[:, 1].min() - 1, Z[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    pdf0 = gaussian_pdf(xx, yy, mu0, S0)
    pdf1 = gaussian_pdf(xx, yy, mu1, S1)
    g = g_diff(xx, yy)

    plt.figure(figsize=(8, 6))

    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], marker=MarkerStyle('o'), label='Class 0', alpha=0.7, edgecolors='k')
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], marker=MarkerStyle('+'), label='Class 1', alpha=0.7)

    plt.contour(xx, yy, pdf0, levels=5, linestyles='--')
    plt.contour(xx, yy, pdf1, levels=5, linestyles=':')

    plt.contour(xx, yy, g, levels=[0], colors='red', linewidths=2)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection with Gaussian contours and decision boundary")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig("pca_gaussian_decision.png", dpi=400)
    print("[INFO] PCA Gaussian decision plot saved as pca_gaussian_decision.png")


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

    # list all eigenvalues and eigenvectors
    print("[INFO] Eigenvalues and Eigenvectors:")
    for i in range(len(eigvals)):
        print(f"[{i+1}] Eigenvalue = {eigvals[i]:.6f}")
        print(f"     Eigenvector = {eigvecs[:, i]}\n")


    plot_pca_scree(eigvals, var_exp)
    return eigvecs



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

def loocv(data, thresh=0.95): 
    print(f"start with thresh={thresh}")

    train = data.drop(columns=['SeqNum','GroundTruth','Gender']).to_numpy()

    std, mu = train.std(axis=0), train.mean(axis=0)
    std[std == 0] = 1.0
    train_std = (train - mu) / std
    PCA(train_std, thresh=thresh) # d×k


if __name__ == '__main__':
    df = pd.read_excel('AcromegalyFeatureSet.xlsx')
    print("[INFO] Data loaded successfully:")
    print(df.head())
    print("[INFO] Starting Leave-One-Out Cross-Validation...")
    np.random.seed(98)

    thresh = 0.95
    loocv(df, thresh=thresh)
    plot_PCA_2D_gaussian(df)