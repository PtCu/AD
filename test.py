import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

x = np.arange(2, 8)

# silhouette_y = [-0.1, -0.05, -0.22, -0.5, -0.6, -0.62]
# ch_y = [0.1, 0.48, 0.25, 0.27, 0.29, 0.32]
# db_y = [1.0, 0.59, 0.64, 0.50, 0.56, 0.57]
# si, = plt.plot(x, silhouette_y, label="Silhoutte score",linestyle="solid")
# ar, = plt.plot(x, ch_y, label="CH score",linestyle="dashdot")
# st, = plt.plot(x, db_y, label="DB score",linestyle="dashed")
# plt.legend([st, si, ar], ["DB score", "Silhoutte score", "CH score"])

# plt.savefig("LouvainSynthetic.png")
# plt.clf()

# x = np.arange(2, 11)
# silhouette_y = [-0.02, -0.1, -0.4, -0.65, -0.70, -0.77, -0.75, -0.82, -0.9]
# ch_y = [0.62, 0.70, 1.0, 0.82, 0.87, 0.81, 0.86, 0.79, 0.77]
# db_y = [0.3, 0.26, 0.3, 0.32, 0.31, 0.308, 0.306, 0.31, 0.32]
# si, = plt.plot(x, silhouette_y, label="Silhoutte score",linestyle="solid")
# ar, = plt.plot(x, ch_y, label="CH score",linestyle="dashdot")
# st, = plt.plot(x, db_y, label="DB score",linestyle="dashed")
# plt.legend([st, si, ar], ["DB score", "Silhoutte score", "CH score"])

# plt.savefig("LouvainSimulated.png")
# plt.clf()

x = np.arange(2, 10)
silhouette_y = [0.20, 0.75, 0.00, -0.95, -0.81, -0.49, -0.75, -0.80]
ch_y = [0.25, 0.9, 0.50, 0.25, 0.41, 0.85, 0.76, 0.67]
db_y = [0.62, 0.45, 1.0, 0.52, 0.48, 0.86, 0.82, 0.34]
si, = plt.plot(x, silhouette_y, label="Silhoutte score", linestyle="solid")
ar, = plt.plot(x, ch_y, label="CH score", linestyle="dashdot")
st, = plt.plot(x, db_y, label="DB score", linestyle="dashed")
plt.legend([st, si, ar], ["DB score", "Silhoutte score", "CH score"])

plt.savefig("LouvainReal.png")
plt.clf()
