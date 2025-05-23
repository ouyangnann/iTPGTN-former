import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func




"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, pos_att):
        super().__init__()
        self.pos_att = pos_att
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_kr = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_kr = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    
    
    def propagate_attention_with_positional(self, g):
    # 集成图位置核到注意力分数
            # Compute attention score using positionally encoded kernels
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
   
        # Incorporate graph positional kernel into the attention scores
        g.apply_edges(lambda edges: {'score': edges.data['score'] + edges.data['kr']})
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Softmax normalization modulated by Kr
      
        g.apply_edges(out_edge_features('score'))
        g.apply_edges(exp('score'))
        # Send weighted values to target nodes and sum up contributions
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
        
       
        
    def forward(self, g, h, e, kr):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        with torch.no_grad():
            g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
            g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
            g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
            
            if self.pos_att:
                proj_kr = self.proj_kr(kr)
                g.edata['kr'] = proj_kr.view(-1, self.num_heads, self.out_dim)
                self.propagate_attention_with_positional(g)
            else:
                self.propagate_attention(g)
                
            
            h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
            e_out = g.edata['e_out']
            
        return h_out, e_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, pos_att = True,dropout=0.2, layer_norm=True, batch_norm=True, residual=True, use_bias=False, gck = False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        #self.prelu = nn.PReLU()
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias, pos_att)
        self.gck = gck
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
       
       
    def gaussian_kernel(self, x, mu=0, sigma=1):
    # 高斯核函数实现
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)   
    
    def forward(self, g, h, e, kr=None):
        if self.gck:
            with g.local_scope():
                g.ndata['h'] = h
                path_embeddings = []
                for length in range(1, 3):  # 枚举最大路径长度
                    g.update_all(dgl.function.copy_u('h', 'm'),
                                 dgl.function.sum('m', 'h'))
                    path_embeddings.append(self.gaussian_kernel(g.ndata['h']))
                h = torch.sum(torch.stack(path_embeddings), dim=0)
        else:
            pass
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e, kr)
        
    
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)
   
        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)
        
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection
        
        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.gelu(self.FFN_h_layer2(h))

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = F.gelu(self.FFN_e_layer2(e))

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)