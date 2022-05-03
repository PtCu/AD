This is the implementation on Bayesian Hierarchical Clustering for multivariate and bernoulli data.

## Installation
1. Open Terminal, clone the git repo by

```sh
git clone https://github.com/qxxxd/bhc.git
```

2. Go to the `bhc` directory, type

```sh
python3 setup.py install
```

3. Ready to use!

4. Then uninstall, if using `pip`

```sh
pip3 uninstall bhc
```

## Basic usage

Prerequisite packages: `numpy`, `scipy` and `functools`.

```python
from bhc import bhclust
from scipy.cluster.hierarchy import dendrogram
Z, color = bhclust(data, family = 'multivariate', alpha = 1)
dendrogram(Z, link_color_func=lambda k : color[k])
```

## License
MIT

## Reference
Heller, K. A. and Z. Ghahramani (2005). Bayesian Hierarchical Clustering. In *Proceedings of the 22Nd International Conference on Machine Learning*, ICML ’05, New York, NY, USA, pp. 297–304. ACM.
