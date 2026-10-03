"""
PCA: From Scratch (NumPy) vs Scikit-Learn
Dataset: Iris Dataset
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    Returns:
        Principal components (eigen vector) of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """

    def standardize(data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0))

    data = standardize(data)
    cov = np.matmul(data.T, data) / len(data)

    # tại sao lại tính np.linalg.eigh(cov)?
    # G/s dữ liệu X được chuẩn hóa (tb=0, cov=1) chiếu lên vector đơn vị u.
    # Tọa độ mới z = X.u
    # Phương sai của ảnh chiếu z: Var(z) = 1/N z.T z = 1/N (u.T X.T) (X u) = u.T (1/N (X.T X)) u = u.T SIGMA u
    # -> Bài toán tối ưu hóa max_u u.T SIGMA u với điều kiện u.T u = 1 (để tìm hướng u đơn vị có phương sai max)
    # PP nhân tử langrange: f = u.T SIGMA u, g = u.T u - 1. giải pt gradient(f) = lambda gradient(g)
    # ra dc 2 SIGMA u - 2 lambda u = 0 -> SIGMA u  = lambda u
    # vậy phương sai chính là giá trị riêng lambda: cov(z) =  u.T SIGMA u = u.T (lambda u) = lambda
    # do vậy cần tìm eigh vector t/u với giá trị riêng của ma trận cov (SIGMA), ở đây mình lười tính nên dùng luôn thư viện
    # Thực tế có nhiều cách tính, phân tích SVD cũng là 1 cách

    eigh_val, eigh_vec = np.linalg.eigh(cov)  # returns in ascending order

    # Reverse columns to descending order (largest variance first) and slice top k
    pcs = eigh_vec[:, ::-1][:, :k]

    # Fix sign ambiguity: ensure the first entry is non-negative
    for j in range(k):
        if np.abs(pcs[0, j]) > 1e-10:
            if pcs[0, j] < 0:
                pcs[:, j] *= -1

    return np.round(pcs, 4)


def pca_sklearn(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA using scikit-learn and return components aligned to the same convention.
    """
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    model = PCA(n_components=k)
    model.fit(data_std)

    # scikit-learn stores components as (n_components, n_features); transpose to (n_features, k)
    pcs = model.components_.T

    # Align sign convention with original function
    for j in range(k):
        if np.abs(pcs[0, j]) > 1e-10:
            if pcs[0, j] < 0:
                pcs[:, j] *= -1

    return np.round(pcs, 4)


if __name__ == "__main__":
    # Example Dataset: Iris (150 samples, 4 features)
    iris = load_iris()
    X = iris.data
    k = 2

    pcs_scratch = pca(X, k=k)
    pcs_sk = pca_sklearn(X, k=k)

    print("=" * 50)
    print("Top", k, "Principal Components (from Scratch):")
    print(pcs_scratch)

    print("\nTop", k, "Principal Components (from Scikit-Learn):")
    print(pcs_sk)

    print("\nExact Match:", np.allclose(pcs_scratch, pcs_sk))
    print("=" * 50)
