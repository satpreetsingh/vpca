import numpy as np

class rPCA(object):
    """Robust principal components analysis (rPCA)

    Solve super position of sparse and low-rank matrix
    using alternating direction methods (ADM). Problem statement
    is as following

        min_(A,B) t * |A|_1 + (1 - t) |B|_* s.t. C = A + B

    Parameters
    ----------
    t : float
        weight between of l1 norm array
    max_iter : int
        maximum iteration of the model
        default: 1000
    tol : float
        convergence criteria
        default: 1e-3

    Reference
    ---------
    Yang2010, "Sparse and low-rank matrix decomposition via
        alternating direction method"
    """
    def __init__(self, t=0.5, max_iter=1000,
                 beta=None, tol=1e-3):

        self.t = float(t)
        self.gamma = t/(1-t)
        self.beta = None
        self.max_iter = max_iter

    def _shrink(self, X):
        """
        Shrink eigenvalue of X matrix
        """
        beta = self.beta
        [U, s, V] = np.linalg.svd(X, full_matrices=False)
        s_soft = np.maximum(s - (1/beta), 0)
        S_soft = np.diag(s_soft)
        return U.dot(S_soft).dot(V)

    def _proj(self, X):
        """
        Threshold operator
        projection on to
        """
        p = self.gamma/self.beta
        X_p = np.clip(X, -p, p)
        return X_p

    def fit_transform(self, C):
        """fit array C to sparse components A and low-rank components B

        Parameters
        ----------
        C : ndarray
            superposition of sparse and low-rank matrix

        Returns
        -------
        A : ndarray
            Sparse matrix components of input array C
        B : ndarray
            Low-rank components of input array C
        """

        self.C = np.array(C)
        m, n = C.shape
        beta =  0.25 * m * n/np.linalg.norm(C,1)
        self.beta = beta
        Z = np.zeros_like(C)
        A = np.zeros_like(C)
        B = np.zeros_like(C)

        for _ in range(self.max_iter):
            D = (1/beta) * Z - B + C # temporary
            A = D - self._proj(D)
            B = self._shrink(C - A + (1/beta) * Z)
            Z = Z - beta * (A + B - C)

        self.A = A
        self.B = B
        return A, B
