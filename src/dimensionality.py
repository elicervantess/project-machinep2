import numpy as np
from .config import N_COMPONENTS

def pca_fit_transform(X, n_components=N_COMPONENTS):
    """
    PCA manual optimizado:
    - Si n_features > n_samples, usa SVD (más eficiente en memoria)
    - Si n_features <= n_samples, usa eigendecomposition de covarianza
    1. centrar
    2. cov = X^T X / (n-1) o usar SVD
    3. autovalores/vectores
    4. ordenar por energía
    5. proyectar
    Retorna: X_proj, mean_vec, components (matriz de autovectores)
    """
    # centramos
    mean_vec = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_vec
    
    n_samples, n_features = Xc.shape
    k = min(n_components, min(n_samples, n_features))
    
    # Si tenemos muchas más features que muestras, usar SVD es más eficiente
    if n_features > n_samples or n_features > 1000:
        print(f"   Usando SVD (n_samples={n_samples}, n_features={n_features})")
        # SVD: Xc = U @ diag(S) @ Vt
        # Componentes principales están en Vt (filas)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt[:k, :].T  # d x k (columnas son componentes)
        X_proj = U[:, :k] * S[:k]  # n x k
        # Varianza explicada: S^2 / (n-1)
        vals = (S[:k] ** 2) / (n_samples - 1)
    else:
        print(f"   Usando eigendecomposition (n_samples={n_samples}, n_features={n_features})")
        # covarianza (dxd)
        cov = np.dot(Xc.T, Xc) / (n_samples - 1)

        # autovalores / autovectores
        vals, vecs = np.linalg.eigh(cov)  # eigh porque cov es simétrica

        # ordenar de mayor a menor autovalor
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # truncar
        components = vecs[:, :k]              # d x k
        X_proj = np.dot(Xc, components)       # n x k

    return X_proj, mean_vec, components, vals[:k]

def pca_transform(X, mean_vec, components):
    """
    Proyecta nuevos datos usando PCA ya entrenado.
    """
    Xc = X - mean_vec
    return np.dot(Xc, components)

def svd_fit_transform(X, n_components=N_COMPONENTS):
    """
    SVD manual (usando numpy.linalg.svd):
    1. centrar
    2. SVD de Xc = U S V^T
    3. tomar primeras k columnas de V
    4. X_proj = Xc @ V_k
    Retorna: X_proj, mean_vec, V_k, singular_vals[:k]
    """
    mean_vec = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_vec

    # SVD completa
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt: k x d  (filas son componentes principales tipo PCA también)
    k = min(n_components, Vt.shape[0])
    V_k = Vt[:k, :].T           # d x k
    X_proj = np.dot(Xc, V_k)    # n x k

    return X_proj, mean_vec, V_k, S[:k]

def pca_2d_projection(X):
    """
    PCA for visualization ONLY (2D).
    Siempre devuelve n x 2.
    """
    X2d, _, _, _ = pca_fit_transform(X, n_components=2)
    return X2d