import numpy as np
from numpy import exp, log
from functools import partial
from scipy.special import gamma, gammaln
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def bhclust(dat, family, alpha, r = 0.001):
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


def scale_matrix(X, N, k, r, m, S):
    """Return scale matrix for the inverse-Wishart distribution on Sigma.
        @X: N records of data with k columns
        @m: prior on the mean, k * 1
        @S: prior on the covariance, k * k
    """

    xsum = np.sum(X, axis = 0).reshape(k,1) # column sum
    t1 = X.T @ X
    t2 = r * N / (N + r) * (m @ m.T)
    t3 = 1/(N+r) * (xsum @ xsum.T)
    t4 = (r / (N + r)) * (m @ xsum.T + xsum @ m.T)

    Sprime = S + t1 + t2 - t3 - t4

    return Sprime


def niw(X, m, S, r):
    """Return marginal likelihood for multivariate normal data using the conjugate prior distribution normal-inverse-Wishart
       @X: N records of data with k columns
       @m: prior on the mean, k * 1
       @S: prior on the covariance, k * k
       @r: scaling factor on the prior precision of the mean
    """

    N, k = X.shape
    v = k
    vprime = v + N
    Sprime = scale_matrix(X, N, k, r, m, S)

    t1 = (2 * np.pi) ** (- N * k / 2)
    t2 = (r / (N + r)) ** (k/2)
    t3 = np.linalg.det(S) ** (v/2)
    t4 = np.linalg.det(Sprime) ** (-vprime/2)
    t5num = np.prod(gamma( (vprime - np.arange(k))/2 ) ) * (2 ** (vprime * k / 2))
    t5den = np.prod(gamma( (v - np.arange(k))/2 ) ) * (2 ** (v * k / 2))

    ml = t1 * t2 * t3 * t4 * (t5num/t5den)

    return np.log(ml)

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

#No consider mean relates to alphas
def bhclust_BB(X, alpha = 0.001):
    """Calculate P(Dk|Tk)
       Return linkage_matrix
    """
    linkage_list = []
    linkage_list_out = []
    nk = 2
    maximum = 0.01
    dim = X.copy().shape[0]
    merge_dim = X.shape[0]
    obs_list = [i for i in range(1,dim+1)]
    dye = {}
    while (nk < dim and maximum !=0):
        maximum = 0
        for i in obs_list:
            for j in obs_list:
                if (j>i):
                    if (i<=dim and j<=dim):
                        s, w = i-1, j-1
                        nk = 2
                        prob_DTi, prob_DTj = prob_DH1(X[s]), prob_DH1(X[w])
                        di, dj = alpha, alpha
                    elif (i<=dim and j>dim):
                        s = i-1
                        w = np.array(linkage_list[j-dim-1][:2]) - 1
                        nk = linkage_list[j-dim-1][3] + 1
                        prob_DTi, prob_DTj = prob_DH1(X[s]), linkage_list[j-dim-1][4]
                        di, dj = alpha, linkage_list[j-dim-1][5]
                    elif (i>dim and j>dim):
                        s = np.array(linkage_list[i-dim-1][:2])-1
                        w = np.array(linkage_list[j-dim-1][:2])-1
                        nk = linkage_list[i-dim-1][3] + linkage_list[j-dim-1][3]
                        prob_DTi, prob_DTj = linkage_list[i-dim-1][4], linkage_list[j-dim-1][4]
                        di, dj = linkage_list[i-dim-1][5], linkage_list[j-dim-1][5]

                    Dk_tmp = np.vstack((X[s],X[w]))

                    dk = alpha*gamma(nk)+di*dj

                    pik = alpha*gamma(nk)/dk
                    prob_DT = prob_DH1(Dk_tmp)*pik + prob_DTi * prob_DTj * di * dj / dk

                    rk = pik*prob_DH1(Dk_tmp)/prob_DT
                    if (rk > maximum):
                        maximum = rk
                        merge_i = i
                        merge_j = j
                        merge_prob_DTi = prob_DT.copy()
                        merge_Dk = Dk_tmp.copy()
                        merge_dk = dk
        if (maximum ==0):
            break
        if (maximum > 0.5):
            dye[merge_dim] = "#0013FF"
        else:
            dye[merge_dim] = "#FF0000"
        merge_dim+=1
        obs_list.append(merge_dim)

        if (merge_i) in obs_list: obs_list.remove(merge_i)    #remove merged observations' idx from list
        if (merge_j) in obs_list: obs_list.remove(merge_j)

        X = np.vstack((X,merge_Dk))
        nk = merge_Dk.shape[0]
        linkage_list.append([merge_i, merge_j, np.log(maximum/(1-maximum)), nk, merge_prob_DTi, merge_dk])
        linkage_list_out.append([merge_i-1, merge_j-1, np.log(maximum/(1-maximum)), nk])

    return (linkage_list_out, dye)

def prob_DH1(X, alpha=0.8, beta=0.2):
    """Return marginal likelihood for bernoulli data using the conjugate prior distribution Bernoulli-Beta
       @X: N records of data with k columns
       @alpha, beta: hyperparmeter for Beta distribution
    """
    md = np.sum(X,axis=0)
    N = X.shape[0]
    nominator = np.array(gamma(alpha+beta)*gamma(alpha+md))*np.array(gamma(beta+N-md))
    denominator = gamma(alpha)*gamma(beta)*gamma(alpha+beta+N)
    return np.prod(nominator/denominator)

def bb_draw(X_test):
    ttt, colorb = bhclust_BB(X=X_test)
    N = X_test.shape[0]
    Z1 = np.array(ttt)
    Z1[:,2] = 1/Z1[:,2]
    maxw = max(Z1[:,2])
    Z1[Z1[:,2] < 0,2] = 2*maxw
    for i in range(Z1.shape[0]):
        if Z1[i, 0] > (N-1):
            Z1[i, 2] += Z1[Z1[i, 0].astype("int")-N, 2]
        if Z1[i,1] > (N-1):
            Z1[i,2] += Z1[Z1[i,1].astype("int")-N, 2]

    dendrogram(Z1,link_color_func=lambda k: colorb[k])
    plt.show()
