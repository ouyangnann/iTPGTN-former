import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from lib import utils
from scipy.sparse.linalg import eigsh
from PIL import Image
import matplotlib.pyplot as plt
from iTransformer import iTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.input_dim = int(model_kwargs.get('input_dim'))
        self.output_dim = int(model_kwargs.get('output_dim'))
        self.num_node = int(model_kwargs.get('num_nodes'))
        self.model_dim = int(model_kwargs.get('model_dim'))
        self.dec_dim = int(model_kwargs.get('dec_dim'))
        self.num_heads = int(model_kwargs.get('num_heads'))
        self.num_encoder_layers = int(model_kwargs.get('num_encoder_layers'))
        self.batch_size = int(model_kwargs.get('batch_size'))
        self.num_decoder_layers = int(model_kwargs.get('num_decoder_layers'))  # 解码器层数
        self.dropout = float(model_kwargs.get('dropout', 0.1))
        self.l1_decay = float(model_kwargs.get('l1_decay', 1.0e-5))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.horizon = int(model_kwargs.get('horizon'))  # for the decoder
        # Add additional parameters required by GTNModel

class iTNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, cuda,**model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self._logger = logger
        self.encoder = iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len,                  # or the lookback length in the paper
                dim = self.model_dim,                          # model dimensions
                depth = self.num_encoder_layers,                          # depth
                heads = self.num_heads,                          # attention heads
                dim_head = self.model_dim//2,                      # head dimension
                pred_length = self.seq_len,     # can be one prediction, or many
                num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
                use_reversible_instance_norm = False
            ) 
        self.decoder =  iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len,                  # or the lookback length in the paper
                dim = self.dec_dim,                          # model dimensions
                depth = self.num_decoder_layers,                          # depth
                heads = self.num_heads,                          # attention heads
                dim_head = self.dec_dim//2,                      # head dimension
                pred_length = self.horizon,     # can be one prediction, or many
                num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
                use_reversible_instance_norm = False
            ) 
    
    def _reset_parameters(self):
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

   
    def init_for_max(self, graph):
    
        self._reset_parameters()
        
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(self))
        )
        self._logger.info(
            "Model parameters: model_dim: {},  dec_dim: {},num_encoder: {}"
            .format(self.model_dim, self.dec_dim,self.num_encoder_layers)
        )


    def forward(self, src, graph, batches_seen=None, tgt=None, h_lap_pos_enc=None, h_wl_pos_enc=None, src_mask=None, tgt_mask=None):

        if batches_seen == 0:
            self.init_for_max(graph)
        src_g = src
        src_ = self.encoder(src)
    
        src = src_[self.seq_len] #batch,len,num_node

        memory = src

        decoder_output = self.decoder(memory)
        output = decoder_output[self.horizon]
    
        return output