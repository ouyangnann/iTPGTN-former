from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle

# 读取 CSV 文件
file_path = 'data/PEMS07/distance.csv'
output_pkl_filename = 'data/sensor_graph/adj_mx_07.pkl'
distance_df = pd.read_csv(file_path)

# 显示数据的前几行来理解其结构
print(distance_df.head())

def create_adjacency_matrix(distance_df):
    # 提取唯一的节点
    nodes = set(distance_df['from']).union(set(distance_df['to']))
    # 节点映射到索引
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
    # 初始化邻接矩阵
    num_nodes = len(nodes)
    adj_matrix = np.inf * np.ones((num_nodes, num_nodes), dtype=float)
    np.fill_diagonal(adj_matrix, 0)  # 自环的成本设置为 0

    # 填充邻接矩阵
    for _, row in distance_df.iterrows():
        from_idx = node_to_index[row['from']]
        to_idx = node_to_index[row['to']]
        adj_matrix[from_idx, to_idx] = row['cost']
    
    return adj_matrix, node_to_index


def create_adjacency_matrix(distance_df,normalized_k=0.1):
    # 提取唯一的节点
    nodes = set(distance_df['from']).union(set(distance_df['to']))
    # 节点映射到索引
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
    # 初始化邻接矩阵
    num_nodes = len(nodes)
    adj_matrix = np.inf * np.ones((num_nodes, num_nodes), dtype=float)
    np.fill_diagonal(adj_matrix, 0)  # 自环的成本设置为 0

    # 填充邻接矩阵
    for _, row in distance_df.iterrows():
        from_idx = node_to_index[row['from']]
        to_idx = node_to_index[row['to']]
        adj_matrix[from_idx, to_idx] = row['cost']

    distances = adj_matrix[adj_matrix != np.inf]
    std = distances.std()

    # 使用高斯函数转换距离
    adj_mx = np.exp(-np.square(adj_matrix / std))
    adj_mx[adj_matrix == np.inf] = 0  # 将无连接的节点设置为0

    # 应用稀疏阈值
    adj_mx[adj_mx < normalized_k] = 0

    return adj_mx, node_to_index


# 创建邻接矩阵
adj_matrix, node_to_index = create_adjacency_matrix(distance_df)
print(adj_matrix.shape)
with open(output_pkl_filename, 'wb') as f:
    pickle.dump([ 'id' , node_to_index, adj_matrix], f, protocol=2)



