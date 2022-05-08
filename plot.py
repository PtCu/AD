import os
import matplotlib.pyplot as plt
import numpy as np
from pip import main
from sympy import rotations
import pandas as pd
cwd_path = os.getcwd()
data_dir = cwd_path+"/output/"

source_list = []
prefix = data_dir+"csv/"


for f_name in os.listdir(prefix):
    if f_name.startswith('synthetic'):
        source_list.append(prefix+f_name)

label_map={}
label_map["(1)"]="GMM"
label_map["(2)"]="NMF"
label_map["(3)"]="LDA"
label_map["(4)"]="URF"
label_map["(5)"]="Hierarchical"
label_map["(6)"]="K-Medians"
label_map["(7)"]="Spectral"
label_map["(8)"]="CHIMERA"
label_map["(9)"]="HYDRA"

save_dir=data_dir+"/box/"
title_map=["t=random","t=120","t=130","t=140","t=150"]
def genPlots():
    cnt=2
    for idx in np.arange(len(source_list),step=5):
        data=[]
        for i in range(5):
            item={}
            X = pd.read_csv(source_list[idx+i],header=None,dtype=float).values
            item[label_map["(1)"]]=X[0]
            item[label_map["(2)"]]=X[1]
            item[label_map["(3)"]]=X[2]
            item[label_map["(4)"]]=X[3]
            item[label_map["(5)"]]=X[4]
            item[label_map["(6)"]]=X[5]
            item[label_map["(7)"]]=X[6]
            item[label_map["(8)"]]=X[7]
            item[label_map["(9)"]]=X[8]
            data.append(item)
       

        # f,axes=plt.subplots(1,5,sharey=True)
        f,axes=plt.subplots(1,5,sharey=True)
        # f.subplots_adjust(wspace=0)
        x_tick=range(1,9)
        y_labels=["GMM","NMF","LDA","URF","Hierarchical","K-Medians","Spectral","CHIMERA","HYDRA"]
        
        axes[0].set_yticklabels(y_labels, fontsize=15)
        for i in range(5):
            labels, to_plot = data[i].keys(), data[i].values()
            axes[i].boxplot(to_plot,vert=False)
            axes[i].set_title(title_map[i],fontsize=20)
            # axes[i].set_xticklabels(x_tick,fontsize=20)
            
  
            # axes[i].xticks(range(1, len(labels) + 1), labels,fontsize=15)
            # axes[i].yticks(range(1,9),fontsize=16)
        # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        # plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
        
        # plt.ylabel("K",fontsize=20)
        # plt.yticks(range(1,9),fontsize=16)
     

        
        # #plt.xticks(range(1, len(labels) + 1), labels,fontsize=15)
        # axes[0].set_yticklabels(x_tick,y_labels, fontsize=15)
        # plt.setp(axes,xticklabels=labels,yticks=y_tick)
        for ax in axes:
            ax.set_xticks([1,2,3,4,5,6,7,8]) # 设置刻度
            ax.set_xticklabels(x_tick,fontsize=20)
            ax.set_yticks([1,2,3,4,5,6,7,8,9]) # 设置刻度
            # ax.set_yticklabels(['one','two','three','four','five'],rotation = 30,fontsize = 'small') # 设置刻度标签

            ax.set_yticklabels(labels, fontsize=15)
            #ax.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
            # ax.set_yticklabels(["one", "two", "three", "four"], rotation=45)
            # ax.tick_params(axis="both", direction="in", pad=15)
        
        # f.xticks(rotation=45)
        f.text(0.5, 0.03, 'K', va='center',fontsize=22)
        # f.text(0.5, 0.005, 'Algorithms', ha='center',fontsize=18)
        f.set_size_inches(20, 12)
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
      
        # plt.xlabel("Algorithms",fontsize=20)
        # plt.ylabel("K",fontsize=20)
        # for i in range(5):
        #     for ticks in axes[i].get_xticklables():
        #         ticks.set_rotation(45)
        plt.savefig(save_dir+"box"+str(cnt)+".png")
        plt.clf()
        cnt+=1

if __name__ == "__main__":
    genPlots()
        


    




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