import bhc
from bhc import bhclust
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

#-------------multivariate data------------#
#dat = np.genfromtxt('tests/dat1.csv', delimiter=',', skip_header=1)
mdat = np.array([[ 0.93637874,  1.61258974],
       [ 1.95192875,  2.84452075],
       [ 2.07671748,  3.24442548],
       [ 3.122903  ,  4.516753  ],
       [ 3.56202194,  5.17531994],
       [ 3.53211875,  5.75857675],
       [ 4.65794237,  6.66995537],
       [ 5.83738797,  8.46562797],
       [ 6.22595817,  9.28082817],
       [ 6.51552067,  9.36110867],
       [ 7.24619975,  3.68958775],
       [ 6.50554148,  3.69771048],
       [ 6.58213752,  4.31283952],
       [ 6.02279742,  4.52753342],
       [ 5.83280398,  4.85751598],
       [ 5.12305078,  4.76874878],
       [ 5.0430706 ,  5.2911986 ],
       [ 2.44081699,  6.35402999]])
mdat.shape
labm = ['A'] * 10 + ['B'] * 8
labcols = {'A': 'y', 'B':'g'}

# scatter plot
plt.scatter(mdat[:,0], mdat[:,1])
for i in range(mdat.shape[0]):
    plt.annotate(i, (mdat[i,0], mdat[i,1]))
plt.show()



# dendrogram
Zm, colorm = bhclust(mdat, family = "multivariate", alpha = 1, r = 0.001)
dendrogram(Zm, link_color_func=lambda k : colorm[k], labels = labm)
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for y in xlbls:
    y.set_color(labcols[y.get_text()])
plt.show()


#-------------bernoulli data-------------#
def Bern_gen(nobs, k, theta, seed):
    """Generate Bernoulli distributed data"""
    np.random.seed(seed)
    obs_list = []
    theta_list = (np.repeat(theta,nobs))
    theta_list[:int(nobs/3)] = np.repeat(theta-0.3, int(nobs/3))
    theta_list[-int(nobs/3):] = np.repeat(theta+0.3, int(nobs/3))
    for i in range(nobs):
        X_i = np.random.binomial(1, theta_list[i], k)
        obs_list.append(X_i)
    return np.matrix(obs_list)

#bdat = Bern_gen(30, 10, 0.5, 121) # saved
#bdat = np.genfromtxt('tests/bindat.csv', delimiter=',')

bdat = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
bdat.shape

# heatmap
plt.pcolor(bdat)
plt.show()

# dendrogram
Zb, colorb = bhclust(bdat, family = "bernoulli", alpha = 0.001)
dendrogram(Zb, link_color_func=lambda k : colorb[k])
plt.show()
