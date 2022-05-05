import itertools
import numpy as np
from collections import Counter
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
from utilities.utils import helper_matrix

label1 = np.array([2, 2, 0, 1, 0, 1, 0, 0, 2, 0, 1, 1, 1, 1, 2, 2, 1, 2, 0, 1, 1, 2,
                   2, 2, 2, 1, 2, 2, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0])

label2 = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 0, 1, 1, 0, 1, 1, 2, 0, 1,
                   1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 0, 1, 2, 0, 1])


def helper(label1, label2, K):
    c1 = Counter(label1)
    c2 = Counter(label2)

    tuple_c1 = c1.most_common(K)
    tuple_c2 = c2.most_common(K)

    l1 = np.array([tuple_c1[i][0] for i in range(len(tuple_c1))])
    l2 = np.array([tuple_c2[i][0] for i in range(len(tuple_c2))])
    l1_idx = [0 for i in range(K)]
    l2_idx = [0 for i in range(K)]
    for i in range(K):
        idx_c1 = np.where(label1 == l1[i])
        idx_c2 = np.where(label2 == l2[i])
        l1_idx[l1[i]] = idx_c1
        l2_idx[l2[i]] = idx_c2


    rank = np.arange(K)
    for i in range(K):
        max_common = -1
        max_common_idx = i
        for j in range(K):
            set1=l1_idx[i][0]
            set2=l2_idx[j][0]
            print("i:"+str(i)+" num "+str(set1))
            print("j:"+str(j)+" num "+str(set2))
            inter = np.intersect1d(set1, set2)
            print(str(i)+" with "+str(j)+" :"+str(len(inter)))
            if len(inter) > max_common:
                max_common = len(inter)
                max_common_idx = j
            # max_common_idx=
        rank[i], rank[max_common_idx] = rank[max_common_idx], rank[i]
        l2_idx[i],l2_idx[max_common_idx]=l2_idx[max_common_idx],l2_idx[i]

    # l2=l2[rank]
    # l1_idx=np.array(l1_idx)
    # l2_idx=np.array(l2_idx)
    # l2_idx=l2_idx[rank]

    matrix = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            inter = np.intersect1d(l1_idx[i][0], l2_idx[j][0])
            matrix[i][j] = len(inter)

    return matrix


# helper(label1,label2,3)
cm = helper(label1, label2, 3)
print(cm)
labels = [0, 1, 2]

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
plt.ylim(len(labels) - 0.5, -0.5)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # plt.text(j, i, cm[i, j])
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.xlabel("URF")
plt.ylabel("Hierachical")
plt.tight_layout()
plt.show()
plt.clf()
