import math
import torch
import torch.nn as nn
from base import MMModel, MultiModalConfig

class TETFNConfig(MultiModalConfig):
    def __init__(self, num_labels, rank):
        super(TETFNConfig, self).__init__(num_labels)
        self.rank = rank
        self.output_dim = num_labels
 

class TextEnhancedTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, drop_out = 0.1, attn_mask = False):
        
        pass
    

class TETFN(MMModel):
    def __init__(self, config:TETFNConfig):
        super(TETFN, self).__init__(config)