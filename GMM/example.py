import numpy as np
from smt.applications import MOE
from smt.problems import LpNorm
from smt.sampling_methods import FullFactorial

import sklearn
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

ndim = 2
nt = 200
ne = 200

# Problem: L1 norm (dimension 2)
prob = LpNorm(ndim=ndim)

# Training data
sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
np.random.seed(0)
xt = sampling(nt)
yt = prob(xt)

# Mixture of experts
print("MOE Experts: ", MOE.AVAILABLE_EXPERTS)

moe = MOE(smooth_recombination=True, n_clusters=2, deny=["RMTB", "KPLSK"])
print("Enabled Experts: ", moe.enabled_experts)
moe.set_training_values(xt, yt)
moe.train()

# Validation data
np.random.seed(1)
xe = sampling(ne)
ye = prob(xe)

# Prediction
y = moe.predict_values(xe)
fig = plt.figure(1)
fig.set_size_inches(12, 11)

# Cluster display
colors_ = list(colors.cnames.items())
GMM = moe.cluster
weight = GMM.weights_
mean = GMM.means_
if sklearn.__version__ < "0.20.0":
    cov = GMM.covars_
else:
    cov = GMM.covariances_
prob_ = moe._proba_cluster(xt)
sort = np.apply_along_axis(np.argmax, 1, prob_)

xlim = prob.xlimits
x0 = np.linspace(xlim[0, 0], xlim[0, 1], 20)
x1 = np.linspace(xlim[1, 0], xlim[1, 1], 20)
xv, yv = np.meshgrid(x0, x1)
x = np.array(list(zip(xv.reshape((-1,)), yv.reshape((-1,)))))
prob = moe._proba_cluster(x)

plt.subplot(221, projection="3d")
ax = plt.gca()
for i in range(len(sort)):
    color = colors_[int(((len(colors_) - 1) / sort.max()) * sort[i])][0]
    ax.scatter(xt[i][0], xt[i][1], yt[i], c=color)
plt.title("Clustered Samples")

plt.subplot(222, projection="3d")
ax = plt.gca()
for i in range(len(weight)):
    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
    ax.plot_trisurf(
        x[:, 0], x[:, 1], prob[:, i], alpha=0.4, linewidth=0, color=color
    )
plt.title("Membership Probabilities")

plt.subplot(223)
for i in range(len(weight)):
    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
    plt.tricontour(x[:, 0], x[:, 1], prob[:, i], 1, colors=color, linewidths=3)
plt.title("Cluster Map")

plt.subplot(224)
plt.plot(ye, ye, "-.")
plt.plot(ye, y, ".")
plt.xlabel("actual")
plt.ylabel("prediction")
plt.title("Predicted vs Actual")

plt.show()
