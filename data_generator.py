import numpy as np
import csv

from pip import main


TOTAL_NUM=500

#正常人标签
NC_TYPE=-1
#病号标签
PT_TYPE=1

sample_id=int(0)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
def gen_a_nc_sample(save_type):
    global sample_id
    sample_id+=1
    volume_size=np.random.normal(1, 0.1,20)
    age=int(np.random.uniform(55.0,85.0))
    atrophy=np.random.normal(0.01*(age-55),0.005*(age-55),20)
    sex=np.random.randint(0,2)
    #随着年龄的自然萎缩作用量
    volume_size-=atrophy
    #归一化
    volume_size=normalization(volume_size)
    #type为0时分开保存feature和covariate
    if(save_type==0):
        with open('covariate.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow([int(sample_id), age, sex,NC_TYPE])  
        
        with open('features.tsv','a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            l=np.array([int(sample_id),NC_TYPE]) 
            a_row=np.append(l,volume_size)
            tsv_w.writerow(a_row)  
    else:
         with open('covariate_feature.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            l=np.array([int(sample_id), age, sex,NC_TYPE])  
            a_row=np.append(l,volume_size)
            tsv_w.writerow(a_row) 
        
    

def gen_a_pc_sample(type,save_type):
    global sample_id
    sample_id+=1
    volume_size=np.random.normal(1, 0.1,20)
    age=int(np.random.uniform(55.0,85.0))
    atrophy=np.random.normal(0.01*(age-55),0.005*(age-55),20)
    sex=np.random.randint(0,2)
    #随着年龄的自然萎缩作用量
    volume_size-=atrophy
     #归一化
    volume_size=normalization(volume_size)
    #添加人为标记的萎缩量
    if(type==0):
        #0到9号ROI区域标记为萎缩
        for i in range(0,9):
            volume_size[i]*=0.85
    else:
        #0,1,5,6,10,11,15,16号区域标记为萎缩
        for i in range(0,4):
            volume_size[i*5]*=0.85
            volume_size[i*5+1]*=0.85

    if(save_type==0):
        with open('covariate.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow([int(sample_id), age, sex,PT_TYPE])  
        
        with open('features.tsv','a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            l=np.array([int(sample_id),PT_TYPE]) 
            a_row=np.append(l,volume_size)
            tsv_w.writerow(a_row)  
    else:
         with open('covariate_feature.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            l=np.array([int(sample_id), age, sex,PT_TYPE])  
            a_row=np.append(l,volume_size)
            tsv_w.writerow(a_row) 
   


def gen_samples(save_type):
    for i in range(0,TOTAL_NUM):
        gen_a_nc_sample(save_type)

    for i in range(0,TOTAL_NUM//2):
        gen_a_pc_sample(0,save_type)
        gen_a_pc_sample(1,save_type)

TRAIN=0
TEST=1

if __name__=="__main__":
    with open('covariate.tsv', 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(["ID", "age", "sex","diagnosis"])  
    
    with open('features.tsv','w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title=["ID","diagnosis"]
        for i in range(0,20):
            title.append("ROI")
        tsv_w.writerow(title)

    with open('covariate_feature.tsv', 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title=["ID", "age", "sex","diagnosis"]
        for i in range(0,20):
            title.append("ROI")
        tsv_w.writerow(title)

    gen_samples(TRAIN)
    sample_id =0
    gen_samples(TEST)
   