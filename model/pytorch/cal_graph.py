import numpy as np
import torch
import dgl
from lib import utils
import scipy.sparse as sp

def laplacian_positional_encoding(L_adj_sp, g, pos_enc_dim):
    """
    Graph positional encoding via Laplacian eigenvectors using precomputed normalized Laplacian matrix.
    """
    # Convert sparse matrix to dense if necessary
    if not sp.issparse(L_adj_sp):
        L_adj_sp = sp.csr_matrix(L_adj_sp)

    # Convert to numpy and compute eigenvectors
    L_np = L_adj_sp.toarray()
    EigVal, EigVec = np.linalg.eigh(L_np)
    idx = EigVal.argsort()  # Sort eigenvalues in ascending order
    EigVal, EigVec = EigVal[idx], EigVec[:, idx]

    # Store the eigenvectors as node features
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()

    return g

def calculate_diffusion_kernel(L, gamma, beta, p):
    # 计算 e^{-beta * L}
    diffusion_kernel = torch.matrix_exp(-beta * L)
    random_walk_kernel = torch.matrix_power(torch.eye(L.shape[0]) - gamma * L, p)
    return diffusion_kernel, random_walk_kernel

def create_graph_structure(support, diffusion_kernel, random_walk_kernel, sigma=0.5):
    support = support.tocoo()
    support.setdiag(0)
    support.eliminate_zeros()

    # 将 scipy 稀疏矩阵转换为 DGL 图
    g = dgl.from_scipy(support)
    # 提取边特征 - 节点间的距离
    edge_features = torch.from_numpy(support.data).float().view(-1, 1)
    g.edata['e'] = edge_features
    
    src, dst = g.edges()

    # 提取 diffusion_kernel 和 random_walk_kernel 对应边的值
    dk_values = diffusion_kernel[src, dst]
    rwk_values = random_walk_kernel[src, dst]

    dk_dist_sq = dk_values ** 2
    rwk_dist_sq = rwk_values ** 2
    kr_values_dk = torch.exp(-dk_dist_sq / (2 * sigma ** 2))
    kr_values_rwk = torch.exp(-rwk_dist_sq / (2 * sigma ** 2))

    g.edata['Kr'] = kr_values_dk.float().view(-1, 1)
    
    return g, edge_features, g.edata['Kr']

def _calculate_supports(adj_mx, threshold=0, pos_enc_dim=16, pos=False):
    """根据 adj_mx 计算图卷积所需的支持矩阵, 并添加Laplacian 位置编码。"""
    supports = []
   
    adj_mx[adj_mx < threshold] = 0
    
    # 计算规范化的拉普拉斯矩阵，只需计算一次
    L_adj_sp = utils.calculate_normalized_laplacian(adj_mx)
    supports.append(L_adj_sp.astype(np.float32))

    # 直接使用 L_adj_sp 计算 diffusion_kernel 和 random_walk_kernel
    L = torch.tensor(L_adj_sp.astype(np.float32).todense())  # 转换为 PyTorch tensor
    diffusion_kernel, random_walk_kernel = calculate_diffusion_kernel(L, beta=1, gamma=0.5, p=2)
    
    
    # 使用 L_adj_sp 创建图结构
    g, e, kr = create_graph_structure(L_adj_sp, diffusion_kernel, random_walk_kernel)
    if pos:
        # 在创建图结构后，计算并添加Laplacian位置编码，直接使用 L_adj_sp
        g = laplacian_positional_encoding(L_adj_sp, g, pos_enc_dim)

    return g, e, kr
