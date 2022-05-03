import matplotlib.pyplot as plt
import numpy as np
gmm = [2, 2, 2, 2, 2]
nmf = [2, 2, 4, 2, 2]
lda = [2, 2, 2, 2, 2]
urf = [2, 2, 2, 2, 2]
hierachical = [2, 2, 2, 2, 2]
k_medians = [2, 2, 2, 2, 2]
spectral = [2, 2, 2, 2, 2]
chimera = [3, 3, 3, 3, 3]
hydra = [2, 2, 2, 2, 2]
data = {}
data["NMF"] = nmf
data["GMM"] = gmm
data["LDA"] = lda
data["URF"] = urf
data["CHIMERA"]=chimera
data["层次聚类"] = hierachical
data["K-Medians"] = k_medians
data["谱聚类"] = spectral
data["HYDRA"] = hydra



labels, data = data.keys(), data.values()
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

plt.figure(figsize=(12,7))
plt.boxplot(data, patch_artist=True)
plt.ylabel("K",fontsize=20)
plt.xlabel("Algorithms",fontsize=20)  # 我们设置横纵坐标的标题。
plt.xticks(range(1, len(labels) + 1), labels,fontsize=15)
plt.yticks(range(1,7),fontsize=16)
# plt.show()
plt.savefig("simulated_box.png")
