# 说明

该系统的说明主要分为三个模块，分别为算法模块，数据生成模块和评估模块。

## 算法模块
1. 包括论文中涉及的算法。每个算法以算法名作为目录，如目录GMM下即为GMM算法的实现。
2. 除了HYDRA算法外，其余算法目前都合并到接口当中。HYDRA算法的实现较为特殊，目前接口还没能合并，所以其数据的生成和评估都有单独处理。
3. 算法的接口类为 xxxClusterer，（xxx为算法名）。可以通过接口调用算法。如GMM接口为GMMClusterer，提供了构造函数__init__方法和fit方法。
   1. __init__方法接受参数K，确定算法的聚类数目
   2. fit方法接受参数X，为算法所需的数据。



## 数据生成模块
1. 包括论文中三种类型的数据，即仿真数据1，仿真数据2和真实数据。
2. 仿真数据1的生成在data_generator_simulated.py中进行了实现。直接运行此文件即可。
3. 仿真数据2的生成在data_generator_synthetic.py中进行了实现。直接运行此文件即可。
4. 真实数据的处理在data_generator_real.py中进行了实现。直接运行此文件即可。
5. HYDRA算法所需的格式不同，所以单独生成了一份该格式的。后续可以将格式进行统一。
6. 生成的数据默认在data/文件夹下。
   
## 评估模块
提供的评估功能包括最优聚类数目评估和混淆矩阵。具体实现在utilities/utils.py。
get_final_K()得到待测算法的最优聚类数目。
get_matrix()得到两个算法的混淆矩阵，并同时绘制出图。
在simulated_main.py, synthetic_main.py, real_main.py中有使用示例

### 评估标准：
1. 若三个指标都相同，则直接取
2. 若三个中有两个相同，则取这两个的
3. 若三个都不同，则取轮廓系数为准

## Reference:

Beyas: 

https://github.com/caponetto/bayesian-hierarchical-clustering-examples
https://github.com/caponetto/bayesian-hierarchical-clustering

CHIMERA:
https://github.com/aoyandong/CHIMERA
https://github.com/anbai106/CHIMERA

LDA:
pip install lda
https://github.com/lda-project/lda

Louvain
https://resetran.top/community-detection/

mixture-of-experts:
https://github.com/davidmrau/mixture-of-experts


HYDRA:
https://github.com/anbai106/mlni
