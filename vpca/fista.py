import numpy as np

class FISTA(object):
    """FISTA, 1D Fast Iterative Shrinkage-Threshold Algorithm

    find approximate sparse array that minimize following cost function,
        argmin (1/2)*|b - Ax|_2^2 + lambda * |x|_1

    Parameters
    ----------

    t0 : float
        initial weight for iteration, default : 1.0
    t1 : float
        initial weight for iteration, default : 1.0
    beta : float range (0, 1)
    l : float
        lambda parameter
    l_bar : float, greater than 0
        lambda bar parameter
    max_iter : int
        number of iteration
    tol : convergence criteria
        stop the algorithm after criteria

    Example
    -------
    fista = FISTA(beta=0.5, l_bar=0.001, max_iter=100)
    x_sparse = fista.fit_transform(A, b)

    Reference
    ---------
    A. Yang et. al., A review of fast l1-minimization algorithms for
        robust face recognition
    """
    def __init__(self, t0=1.0, t1=1.0, l=0.01, l_bar=0.001,
                 beta=0.5, max_iter=100, tol=1e-3):
        self.t0 = t0
        self.t1 = t1
        self.l = l
        self.l_bar = l_bar
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

    def _update(self, t):
        t = 0.5 * (1. + np.sqrt(1 + 4 * (t**2)))
        return t

    def _prox(self, x, l):
        """proximal operator"""
        return np.sign(x) * (np.abs(x) - l) * (np.abs(x) > l)

    def fit_transform(self, A, b):
        """fit sparse
        """
        m, n = A.shape
        b = b.ravel()
        x0 = np.zeros(n)
        x1 = np.zeros(n)
        l = self.l
        l_bar = self.l_bar
        t0 = self.t0
        t1 = self.t1

        for _ in range(self.max_iter):
            y = x0 + ((t0 - 1.0)/t1) * (x1 - x0)
            u = y - l * (A.T).dot(A.dot(y) - b)
            t0 = t1
            t1 = self._update(t1)
            x0 = x1
            x1 = self._prox(u, l)
            l = np.maximum(self.beta * l, l_bar)

        self.t0 = t0
        self.t1 = t1
        self.l = l
        self.fit_ = x1

        return x1
