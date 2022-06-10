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

## 评估模块
提供的评估功能包括最优聚类数目评估，ARI评估和混淆矩阵。实现在utilities/utils.py。


reference:

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
