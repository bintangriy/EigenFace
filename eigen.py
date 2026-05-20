import numpy as np

def power_iteration(A, num_simulations=100):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    return eigenvalue, b_k

def compute_eigen_manual(A, k=10):
    eigenvalues = []
    eigenvectors = []
    B = np.copy(A)
    for _ in range(k):
        val, vec = power_iteration(B)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        B = B - val * np.outer(vec, vec)
    return np.array(eigenvalues), np.array(eigenvectors)