# Variety of PCA (vpca)

variety implementation of principal components analysis (PCA) in Python


## rPCA

Robust principal components analysis (sparse, low-rank matrix decomposition)

- [Candes et. al., Robust Principal Component Analysis?](https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf)
- [Yang et. al., "Sparse and low-rank matrix decomposition via alternating direction method"](http://www.optimization-online.org/DB_FILE/2009/11/2447.pdf)

```python
import numpy as np
from vpca import rPCA

m, n = 100, 80 # array dimension
r = 5 # rank ratio
spr = 5./100. # sparse ratio (in percent)
B = np.dot(np.random.randn(m,r), np.random.randn(r,n)) # creat low-rank matrix
A = np.zeros((m,n))
p = np.random.permutation(m * n)
L = np.int(spr * m * n)
A = np.reshape(A, [m * n, 1])
A[p[0:L]] = np.random.randn(L, 1)
A = np.reshape(A, [m, n]) # reshape back to m x n
C = A + B # addition of sparse and low-rank matrix

rpca = rPCA(t=0.5)
Aest, Best = rpca.fit_transform(C)
```


## jPCA

Python implementation of jPCA from [Matlab](http://churchlandlab.neuroscience.columbia.edu/links.html)
implementation. Description of algorithm can be fully found from

- [Churchland, Cunningham, _Neural population dynamics during reaching_, Nature, 2012](http://stat.columbia.edu/~cunningham/pdf/nature11129_all.pdf)

Download example neural data file using `download` function,

```python
import vpca
vpca.download()
```

## FISTA

1-dimensional Fast Iterative Shrinkage-Threshold Algorithm to solve basis pursuit
problem.

```python
from vpca import FISTA

spr = 0.2
lambda_bar = 0.001
m, n = 200, 20
A = np.random.randn(m,n)
L = np.int(spr*n)
p = np.random.permutation(n)
x_spr = np.zeros(n)
x_spr[p[0:L]] = np.random.randn(L) # sparse signal
b = A.dot(x_spr) # measurement signal

fista = FISTA()
x_est = fista.fit_transform(A, b)
```


## Dependencies

- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
