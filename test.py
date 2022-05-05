import matplotlib.pyplot as plt
import numpy as np
from sympy import rotations
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
data["Hierachical"] = hierachical
data["K-Medians"] = k_medians
data["Spectral"] = spectral
data["HYDRA"] = hydra



labels, data = data.keys(), data.values()
# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

plt.figure(figsize=(12,7))
plt.boxplot(data, patch_artist=True)
plt.ylabel("K",fontsize=22)
plt.xlabel("Algorithms",fontsize=22)  # 我们设置横纵坐标的标题。
plt.xticks(range(1, len(labels) + 1), labels,fontsize=15,rotation=30) 
plt.yticks(range(1,7),fontsize=16)
# plt.show()
plt.savefig("simulated_box.png")

#https://www.aiuai.cn/aifarm1335.html

from sklearn.metrics import confusion_matrix,adjusted_rand_score
label1 =[1,0,1,1,1,0,0,1,1]
label2 = [0,1,0,0,0,1,1,0,0]
labels = [0, 1]
cm = confusion_matrix(label1, label2, labels)
ari=adjusted_rand_score(label1,label2)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("demo")
plt.show()