# if need to re-run:
# c++ -O3 -shared -std=c++11 -I <Path-to-pybind11>/include `python3-config --cflags --ldflags` helper.cpp -o helper.so

from helper import niw
import numpy as np
from numpy import exp, log
from functools import partial
from scipy.special import gammaln

def bhclust_fast(dat, family, alpha, r = 0.001):
    """Return a matrix in the format of linkage matrix for dendrogram
        @dat: N records of data with k columns
        @family: function to specify distribution for data. {"multivariate", "bernoulli"}
        @alpha: hyperparameter for the prior
        @r: scaling factor on the prior precision of the mean
    """
    N, k = dat.shape
    la = log(alpha)

    if family == "multivariate":
        m = np.mean(dat, axis=0).reshape(k, 1)
        S = np.cov(dat.T)/10 # precision?
        def mlfunc(X):
            return niw(X, m, S, r)

    elif family == "bernoulli":
        #r=0.01
        m = np.mean(np.vstack((dat, np.ones(k)*r, np.zeros(k))), axis=0)
        alp= m*2; beta=(1-m)*2
        mlfunc = partial(bb, alp=alp, beta=beta)

    # leaf nodes
    SS = list(range(N))
    x0 = []; d0 = [la] * N
    ml = []
    for l in range(N):
        x0.append((l,))
        ml.append(mlfunc(dat[l,].reshape(1,k)))

    # paired base cases
    t = 0; PP = []
    c1 = []; c2 = []
    x = []; d = []
    lp1 = []; lp2 = []; lodds = []
    for i in range(N-1):
        for j in range(i+1, N):
            c1.append(i); c2.append(j)
            x.append(x0[i]+x0[j])
            u = la + gammaln(len(x[t]))
            v = d0[i] + d0[j]
            d.append((u + log(1 + exp(v - u))))
            lp1.append(mlfunc(dat[x[t],:]) + la + gammaln(len(x[t])) - d[t])
            lp2.append(ml[i] + ml[j] + d0[i] + d0[j] - d[t])
            lodds.append(lp1[t] - lp2[t])
            PP.append(t); t = t + 1

    # build tree, Z = [leaf1, leaf2, weight, #leaves]
    p = 0
    Z = []
    dye = {}
    while(1):
        idx = lodds.index(max([lodds[y] for y in PP]))
        Z.append([c1[idx], c2[idx], 1/lodds[idx], len(x[idx])])
        if lodds[idx] < 0:
            dye[N + p] = "#FF0000"
        else:
            dye[N + p] = "#0013FF"

        x0.append(x[idx]); d0.append(d[idx]); ml.append(lp1[idx] + log(1+exp(lp2[idx] - lp1[idx])))
        rm = set(Z[p][:2])
        SS = [y for y in SS if y not in rm]
        if len(SS) == 0:
            break

        for q in SS:
            c1.append(N+p); c2.append(q)
            x.append(x0[N+p] + x0[q])

            u = la + gammaln(len(x[t]))
            v = d0[N+p] + d0[q]
            d.append((u + log(1 + exp(v - u))))
            lp1.append(mlfunc(dat[x[t],:]) + la + gammaln(len(x[t])) - d[t])
            lp2.append(ml[N+p] + ml[q] + d0[N+p] + d0[q] - d[t])
            lodds.append(lp1[t] - lp2[t])
            PP.append(t); t = t + 1

        PP = [y for y in PP if c1[y] not in rm and c2[y] not in rm]
        SS.append(N + p); p = p + 1

    Z_ = weighted(Z, N)

    return Z_, dye


def weighted(Z, N):
    mw = max([y[2] for y in Z])
    for i in range(len(Z)):
        if Z[i][2] < 0:
            Z[i][2] = 2 * mw
        if Z[i][0] > (N - 1):
            Z[i][2] += Z[Z[i][0] - N][2]
        if Z[i][1] > (N - 1):
            Z[i][2] += Z[Z[i][1] - N][2]
    return Z



def bb(X, alp=0.001, beta=0.01):
    """Return marginal likelihood for bernoulli data using the conjugate prior distribution Bernoulli-Beta
       @X: N records of data with k columns
       @alpha, beta: hyperparmeter for Beta distribution
    """
    md = np.sum(X,axis=0)
    N = X.shape[0]
    num = gammaln(alp+beta) + gammaln(alp+md) + gammaln(beta+N-md)
    den = gammaln(alp) + gammaln(beta) + gammaln(alp+beta+N)
    return np.sum(num - den)
