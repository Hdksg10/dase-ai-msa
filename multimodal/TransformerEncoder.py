import math
import torch
import torch.nn as nn

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    

def generate_square_subsequent_mask(sz) -> torch.Tensor:
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out = 0.1, attn_mask = False):
        super(TransformerEncoderLayer, self).__init__()
        if attn_mask:
            self.mask = generate_square_subsequent_mask(embed_dim)
        else:
            self.mask = None
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, drop_out, batch_first=True)
        
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x, x_k, x_v):
        # print(self.self_attn(x, x_k, x_v, attn_mask=self.mask)[0].shape)
        x = x + self.self_attn(x, x_k, x_v, attn_mask=self.mask)[0]
        x = self.norm1(x)
        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, drop_out = 0.1, attn_mask = False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, drop_out, attn_mask) for _ in range(layers)])
        self.pos = PositionalEncoding(embed_dim, drop_out)

    def forward(self, x, x_k, x_v):
        """_summary_

        Args:
            x (_type_): _description_
            x_k (_type_): _description_
            x_v (_type_): _description_

        Returns:
            shape(batch_size, seq_len, embed_dim)
        """
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, x_k, x_v)
        return x
    

if __name__ == '__main__':
    # test TransformerEncoder
    embed_dim = 512
    num_heads = 8
    layers = 1
    drop_out = 0.1
    attn_mask = False
    batch_size = 8
    seq_len = 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    x_k = torch.randn(batch_size, seq_len, embed_dim)
    x_v = torch.randn(batch_size, seq_len, embed_dim)
    model = TransformerEncoder(embed_dim, num_heads, layers, drop_out, attn_mask)
    y = model(x, x_k, x_v)
    print(y.shape)
    
    pass