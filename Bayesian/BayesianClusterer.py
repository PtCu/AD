from pdb import main

import os
import numpy as np

import networkx as nx

from Bayesian.bhc import (BayesianHierarchicalClustering,
                          BayesianRoseTrees,
                          NormalInverseWishart)

cwd_path = os.getcwd()


class BayesianClusterer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.labels_ = []
        self.x_data = []
        self.y_data = []
        self.k_num = 0

    

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        self.y_data = X["true_label"]
        self.run_brt(self.x_data)

    def run_bhc(self, data):
        # Hyper-parameters (these values must be optimized!)
        g = 20
        scale_factor = 0.001

        model = NormalInverseWishart.create(data, g, scale_factor)
        # alpha 在0到1之间，表示一个新点变为一个新簇的概率
        bhc_result = BayesianHierarchicalClustering(data,
                                                    model,
                                                    self.alpha,
                                                    cut_allowed=True).build()

        self.get_label(bhc_result.node_ids,
                  bhc_result.arc_list)

   
    def run_brt(self, data):
        # Hyper-parameters (these values must be optimized!)
        g = 10
        scale_factor = 1

        model = NormalInverseWishart.create(data, g, scale_factor)

        brt_result = BayesianRoseTrees(data,
                                       model,
                                       self.alpha,
                                       cut_allowed=True).build()
        self.get_label(brt_result.node_ids,
                  brt_result.arc_list)
        

    def get_label(self, node_ids, arc_list):
        dag = nx.DiGraph()
        # 转化为图结构
        for id in node_ids:
            dag.add_node(id)

        for arc in arc_list:
            dag.add_edge(arc.source, arc.target)

        output_dict = {}
        # 簇的id
        id = 0
        # 将对应连通分量上的叶节点归到相应的簇上
        for c in nx.weakly_connected_components(dag):
            nodeSet = dag.subgraph(c).nodes()
            id = id + 1
            for node in nodeSet:
                if dag.in_degree(node) == 1 and dag.out_degree(node) == 0:
                    output_dict[node] = id

        output_label = np.zeros(len(self.x_data), dtype=int)

        for key in output_dict:
            output_label[key] = output_dict[key]

        self.labels_ = output_label
        self.k_num = id
