# Variety of PCA (vpca)

variety implementation of principal components analysis (PCA) in Python


## rPCA

Robust principal components analysis (sparse, low-rank matrix decomposition)

- [Candes et. al., Robust Principal Component Analysis?](https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf)
- [Yang et. al., "Sparse and low-rank matrix decomposition via alternating direction method"](http://www.optimization-online.org/DB_FILE/2009/11/2447.pdf)


## jPCA

Python implementation of jPCA from [Matlab](http://churchlandlab.neuroscience.columbia.edu/links.html)
implementation. Description of algorithm can be fully found from

- [Churchland, Cunningham, _Neural population dynamics during reaching_, Nature, 2012](http://stat.columbia.edu/~cunningham/pdf/nature11129_all.pdf)

Download example neural data file using `download` function,

```python
import vpca
vpca.download()
```

## Dependencies

- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
