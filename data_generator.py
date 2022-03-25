from operator import imod
import numpy as np
import csv

from pip import main
from pandas import DataFrame,Series

import pandas as pd
import os
# # 三个字段 name, site, age
# nme = ["Google", "Runoob", "Taobao", "Wiki"]
# st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
# ag = [90, 40, 80, 98]
   
# # 字典https://www.jb51.net/article/185302.htm
# dict = {'name': nme, 'site': st, 'age': ag}

# title=["ID","GROUP", "COVAR", "COVAR"]
# for i in range(0,20):
#         title.append("ROI")

# df = pd.DataFrame(dict,title)
 
# # 保存 dataframe
# df.to_csv('site.csv')

TOTAL_NUM=500

#正常人标签
NC_TYPE=0
#病号标签
PT_TYPE=1

DATA_SET=0


cwd_path = os.getcwd()

data_dir=cwd_path+"/data/"

file_name = os.path.join(data_dir,"simulated_data.tsv")
#COVAR1为年龄，COVAR2为性别
sample_id=int(0)

data=[]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
def gen_a_nc_sample(isNomalized=True):
    item=[]
    global sample_id
    sample_id+=1
    volume_size=np.random.normal(1, 0.1,20)
    age=int(np.random.uniform(55.0,85.0))
    atrophy=np.random.normal(0.01*(age-55),0.005*(age-55),20)
    sex=np.random.randint(0,2)
    #随着年龄的自然萎缩作用量
    volume_size-=atrophy
    #归一化
    if isNomalized:
        volume_size=normalization(volume_size)
    #type为0时分开保存feature和covariate
    item.append(sample_id)
    item.append(NC_TYPE)
    item.append(DATA_SET)
    item.append(age)
    item.append(sex)
    item.extend(volume_size)
    data.append(item)
    # with open(file_name, 'a',newline='') as f:
    #     tsv_w = csv.writer(f, delimiter='\t')
    #     l=np.array([int(sample_id),NC_TYPE, age, sex])  
    #     a_row=np.append(l,volume_size)
    #     tsv_w.writerow(a_row) 
        
    
def gen_a_pc_sample(type, isNomalized=True):
    item = []
    global sample_id
    sample_id+=1
    volume_size=np.random.normal(1, 0.1,20)
    age=int(np.random.uniform(55.0,85.0))
    atrophy=np.random.normal(0.01*(age-55),0.005*(age-55),20)
    sex=np.random.randint(0,2)
    #随着年龄的自然萎缩作用量
    volume_size-=atrophy
     #归一化
    if isNomalized:
        volume_size=normalization(volume_size)

    #添加人为标记的萎缩量
    #前250个为1型
    if(type==0):
        #0到9号ROI区域标记为萎缩
        for i in range(0,9):
            volume_size[i]*=0.85
    #后250个为2型
    else:
        #0,1,5,6,10,11,15,16号区域标记为萎缩
        for i in range(0,4):
            volume_size[i*5]*=0.85
            volume_size[i*5+1]*=0.85

    item.append(sample_id)
    item.append(PT_TYPE)
    item.append(DATA_SET)
    item.append(age)
    item.append(sex)
    item.extend(volume_size)
    data.append(item)

    # with open(file_name, 'a',newline='') as f:
    #     tsv_w = csv.writer(f, delimiter='\t')
    #     l=np.array([int(sample_id),PT_TYPE, age, sex])  
    #     a_row=np.append(l,volume_size)
    #     tsv_w.writerow(a_row) 



def gen_samples():
    for i in range(0,TOTAL_NUM):
        gen_a_nc_sample()

    for i in range(0,TOTAL_NUM//2):
        gen_a_pc_sample(0)

    for i in range(0,TOTAL_NUM//2):
        gen_a_pc_sample(1)



if __name__=="__main__":

    with open(file_name, 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title=["ID","GROUP", "SET","COVAR", "COVAR"]
        for i in range(0,20):
            title.append("ROI")

        tsv_w.writerow(title)

    gen_samples()
    df = pd.DataFrame(data,columns=title)
    final_df=df.set_index("ID")
    
    # 保存 dataframe
    final_df.to_csv(file_name,sep='\t')

   