import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics.cluster import contingency_matrix
import csv
import sys
import numpy as np

from collections import Counter
H_K_2=np.array([[115,1],[35,152]])
H_K_3=np.array([[123,0,1],[0,40,0],[29,19,90]])
H_K_4=np.array([[40,20,2,0],[2,71,0,48],[0,0,15,0],[0,0,0,105]])

H_K_2=np.array([[115,1],[35,152]])
H_K_3=np.array([[123,0,1],[0,40,0],[29,19,90]])
H_K_4=np.array([[40,20,2,0],[2,71,0,48],[0,0,15,0],[0,0,0,105]])

U_H_2=np.array([[125,15],[25,138]])
U_H_3=np.array([[117,0,6],[2,55,32],[34,4,53]])
U_H_4=np.array([[0,0,18,34],[17,38,37,3],[0,4,31,23],[0,0,5,93]])

U_K_2=np.array([[110,30],[10,153]])
U_K_3=np.array([[108,0,15],[0,49,40],[19,0,72]])
U_K_4=np.array([[0,0,29,23],[17,56,22,0],[0,9,39,10],[0,0,9,89]])


X1=[H_K_2,H_K_3,H_K_4]
X2=[U_H_2,U_H_3,U_H_4]
X3=[U_K_2,U_K_3,U_K_4]

def get_matrix(X,filename, title,name1,name2):
    f,axes=plt.subplots(1,3)
    
    for idx in range(3):
        labels=list(range(idx+2))
        cm = X[idx]
        im=axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        tick_marks = np.arange(len(labels))
        axes[idx].set_xticks(tick_marks)
        axes[idx].set_yticks(tick_marks)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        axes[idx].set_ylim(len(labels) - 0.5, -0.5)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # plt.text(j, i, cm[i, j])
            axes[idx].text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        axes[idx].set_xlabel(name1)
        axes[idx].set_ylabel(name2)
        axes[idx].set_title("K="+str(idx+2))

    f.colorbar(im, ax=[axes[0], axes[1], axes[2]],shrink=0.5)
    # plt.tight_layout()
    plt.show()
    # f.colorbar()

get_matrix(X3,"123","title","URF","K-Medians")