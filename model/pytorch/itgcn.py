import numpy as np
import torch
import torch.nn as nn
import math
from model.pytorch.graph_transformer_edge_layer import GraphTransformerLayer
import torch.nn.functional as F
import dgl

from model.pytorch.dcrnn_model import EncoderModel
from lib import utils
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from PIL import Image
import matplotlib.pyplot as plt
from iTransformer import iTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.input_dim = int(model_kwargs.get('input_dim', 2))
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.num_node = int(model_kwargs.get('num_nodes'))
        self.model_dim = int(model_kwargs.get('model_dim', 64))
        self.dec_dim = int(model_kwargs.get('dec_dim', 32))
        self.num_heads = int(model_kwargs.get('num_heads', 8))
        self.num_encoder_layers = int(model_kwargs.get('num_encoder_layers', 2))
        self.batch_size = int(model_kwargs.get('batch_size', 128))
        self.num_decoder_layers = int(model_kwargs.get('num_decoder_layers', 2))  
        self.dropout = float(model_kwargs.get('dropout', 0.5))
        self.l1_decay = float(model_kwargs.get('l1_decay', 1.0e-5))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.horizon = int(model_kwargs.get('horizon'))  # for the decoder
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', True))
        # Add additional parameters required by GTNModel
       
        self.g_heads = int(model_kwargs.get('g_heads', 2))
        self.g_dim = int(model_kwargs.get('g_dim', 32))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 2))
        self.readout = model_kwargs.get('readout', 'max')
        self.layer_norm = model_kwargs.get('layer_norm', True)
        self.use_bias = model_kwargs.get('use_bias', True)
        self.batch_norm = model_kwargs.get('batch_norm', False)
        self.residual = model_kwargs.get('residual', True)
        self.edge_feat = model_kwargs.get('edge_feat', True)
        self.g_threshold = model_kwargs.get('g_threshold', True)
        self.pos_att = model_kwargs.get('pos_att', False)
        self.gck = model_kwargs.get('gck', False)
   
class iTGCNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, cuda,**model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self._logger = logger
        self.encoder = iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len,
                dim = self.model_dim,                         
                depth = self.num_encoder_layers,                      
                heads = self.num_heads,                    
                dim_head = self.model_dim//2,                   
                pred_length = self.seq_len,    
                num_tokens_per_variate = 1,     
                use_reversible_instance_norm = False
            ) 
        self.decoder =  iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len,                  
                dim = self.dec_dim,                        
                depth = self.num_decoder_layers,             
                heads = self.num_heads,                      
                dim_head = self.dec_dim//2,               
                pred_length = self.horizon,    
                num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
                use_reversible_instance_norm = False
            ) 

        self.GClayers =  EncoderModel(**model_kwargs) 
 
        # Input embedding layer
        
        #self.positional_encoding_g = PositionalEncoding(d_model=self.g_dim)
        #self.input_embedding = nn.Linear(self.input_dim*self.num_node,self.model_dim)
        
        self.init_emb_layers()
        



    def _reset_parameters(self):
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    
    def calculate_diffusion_kernel(self, L, gamma,beta, p):
    # 计算 e^{-beta * L}
        diffusion_kernel = torch.matrix_exp(-beta * L)
        random_walk_kernel = torch.matrix_power(torch.eye(L.shape[0]) - gamma * L, p)
    
        # Convert to sparse format if not already sparse
        #diffusion_kernel_sparse = sp.csr_matrix(diffusion_kernel.numpy())
        #random_walk_kernel_sparse = sp.csr_matrix(random_walk_kernel.numpy())
        return diffusion_kernel, random_walk_kernel

  
    def create_graph_structure(self, support,  diffusion_kernel, random_walk_kernel):

        support = support.tocoo()
        support.setdiag(0)
        support.eliminate_zeros()
        g = dgl.from_scipy(support)
   
        edge_features = torch.from_numpy(support.data).float().view(-1, 1)
    
        g.edata['e'] = edge_features
        
        src, dst = g.edges()

        dk_values = diffusion_kernel[src, dst]
        rwk_values = random_walk_kernel[src, dst]

    
        g.edata['Kr'] = dk_values.float().view(-1, 1)
       
        return g, edge_features, g.edata['Kr']

    def _calculate_supports(self, adj_mx, threshold):

        supports = []
        adj_mx[adj_mx < threshold] = 0      
        L_adj_sp= utils.calculate_normalized_laplacian(adj_mx)
        supports.append(L_adj_sp.astype(np.float32))
        L = torch.tensor(L_adj_sp.astype(np.float32).todense()) 
        diffusion_kernel, random_walk_kernel = self.calculate_diffusion_kernel(L, beta=1,gamma=0.5, p=2)
        
        for support in supports:
            g,e,kr = self.create_graph_structure(L_adj_sp, diffusion_kernel, random_walk_kernel)
        #_laplacian_positional_encoding

        return g,e,kr

   
    def init_for_max(self, graph):
        g, e, Kr = self._calculate_supports(graph,self.g_threshold)
        self.h = torch.ones(self.num_node, 1).to(self.device)
        self.e_len, _ = e.size()
        self.e = e.to(self.device)
        self.kr = Kr.to(self.device)
        self.g = g.to(self.device)
        self.batch_g = dgl.batch([g for _ in range(self.batch_size)])
        self.batch_g =  self.batch_g.to(self.device)
        self.batch_num_nodes = self.batch_g.batch_num_nodes()
        self._reset_parameters()
        
        #self.src_indices = (torch.arange(self.seq_len+3, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0) / self.seq_len).repeat(batch_size, 1, 1).to(self.device)
        #self.h_indices = (torch.arange(self.num_node+2, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0) / self.num_node).repeat(batch_size, 1, 1).to(self.device)
        #self.e_indices = (torch.arange(self.e_len+2, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0) / self.e_len).repeat(batch_size, 1, 1).to(self.device)
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(self))
        )
        self._logger.info(
            "Model parameters: model_dim: {}, g_dim: {}, dec_dim: {},num_encoder: {}, num_g: {}"
            .format(self.model_dim,self.g_dim, self.dec_dim,self.num_encoder_layers, self.num_rnn_layers)
        )


    def init_emb_layers(self):
      
        self.g_conv=nn.Sequential(
            nn.Conv1d(in_channels=self.seq_len, out_channels=self.g_dim, kernel_size = 3, stride=1, padding=1) ,
            nn.BatchNorm1d(self.g_dim))
        
        
        self.gs_conv=nn.Sequential(
            nn.Conv1d(in_channels=self.num_node, out_channels=self.num_node, kernel_size = 3, stride=2, padding=1) ,
            nn.BatchNorm1d(self.num_node),
        )
        self.gs_lin=nn.Linear(self.g_dim//2,self.seq_len)
        
        self.e_lin = nn.Linear(self.num_node*self.seq_len,self.g_dim)
        self.kr_lin = nn.Linear(self.num_node*self.seq_len,self.g_dim)
        
        self.rate_g = int(math.log(self.dec_dim/8, 2))

        self.g_upemb_ = nn.Sequential(
            nn.ConvTranspose1d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8), 
            
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            
        )
        
        self.g_upemb_2  =  nn.Sequential( 
            nn.Conv1d(in_channels=self.g_dim, out_channels=self.seq_len,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.seq_len))  
        
        self.g_upemb = nn.Sequential(*[nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.seq_len, out_channels=self.seq_len, kernel_size=3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(self.seq_len), 
            
            nn.Conv1d(in_channels=self.seq_len, out_channels=self.seq_len,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.seq_len), 
            
            )for i in range(self.rate_g)])
        
        self.g_upemb_lin = nn.Linear(self.model_dim,self.num_node)     

        self.hg_bn = nn.LayerNorm(self.g_dim)
     


    def forward(self, src, graph, batches_seen=None, tgt=None, h_lap_pos_enc=None, h_wl_pos_enc=None, src_mask=None, tgt_mask=None):
        
        if batches_seen == 0:
            self.init_for_max(graph)
            
        src_g = src
        src_ = self.encoder(src)
    
        src = src_[self.seq_len] #batch,len,num_node

        current_hidden_state = None
        
            
        for t in range(self.seq_len):
            src_g,  current_hidden_state= self.GClayers(src[:, t, :], graph, current_hidden_state)
        
        src_g = src_g.reshape(-1,self.g_dim)
        
        #print(src_g_.max(),src_g_.min(),batch_e_.min(),batch_e_.max())
        self.batch_g.ndata['h'] = src_g


        src_g = src_g.view(-1,self.num_node,self.g_dim)

        src_g = self.gs_conv(src_g)
        src_g = self.gs_lin(src_g)
        #print(src.max(),src.min(),src_g.max(),src_g.min())

        src = src + src_g.view(-1,self.seq_len,self.num_node)

        g_mean = dgl.mean_nodes(self.batch_g, 'h')
        g_max = dgl.max_nodes(self.batch_g, 'h')
        g_sum = dgl.sum_nodes(self.batch_g, 'h')
        

        hg = torch.cat((g_mean.unsqueeze(1), g_max.unsqueeze(1), g_sum.unsqueeze(1)), dim=1)
        hg = self.hg_bn(hg)

        hg = self.g_upemb_(hg)
        hg = self.g_upemb_2(hg.permute(0,2,1))
        hg = self.g_upemb(hg)
        hg = self.g_upemb_lin(hg)

        memory = src + hg 

        decoder_output = self.decoder(memory)
        output = decoder_output[self.horizon]
        #output = self.output_layer(output)
        return output
