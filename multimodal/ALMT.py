# implement of paper "Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis"
# published in EMNLP 2023
# https://arxiv.org/abs/2310.05804
import torch
import torch.nn as nn
from .base import MMModel, MultiModalConfig
from .TransformerEncoder import TransformerEncoder

class ALMTConfig(MultiModalConfig):
    def __init__(self, num_labels, drop_out = 0.1,
                 embed_dim = 768, 
                 embed_trm_layers = 1,
                 embed_trm_heads = 8,
                 embed_trm_token_len = 8,
                 ahl_layers = 3,
                 ahl_heads = 2,
                 fusion_layers = 3,
                 fusion_trm_heads = 8
                 ):
        super(ALMTConfig, self).__init__(num_labels)
        self.num_labels = num_labels
        self.drop_out = drop_out
        self.embed_dim = embed_dim
        self.embed_trm_layers = embed_trm_layers
        self.embed_trm_heads = embed_trm_heads
        self.embed_trm_token_len = embed_trm_token_len
        self.ahl_layers = ahl_layers
        self.ahl_heads = ahl_heads
        self.fusion_layers = fusion_layers
        self.fusion_trm_heads = fusion_trm_heads

class AHL_layer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out = 0.1):
        super(AHL_layer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, drop_out, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, text_embed, visual_embed, hyper_embed):
        # print(text_embed)
        # print(visual_embed)
        # print(hyper_embed)
        text_embed = self.norm1(text_embed)
        visual_embed = self.norm2(visual_embed)
        hyper_embed = self.norm3(hyper_embed)
        out = self.attn(text_embed, visual_embed, visual_embed, attn_mask=None)[0]
        hyper_embed = out + hyper_embed
        return hyper_embed
        
class HyperModalityLearning(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, drop_out = 0.1):
        super(HyperModalityLearning, self).__init__()
        self.ahl_layers = nn.ModuleList([AHL_layer(embed_dim, num_heads, drop_out) for _ in range(layers)])
        self.trm_layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, layers = 1, drop_out=0.1) for _ in range(layers-1)])
        
    def forward(self, text_embed, visual_embed):
        hyper_embed = torch.randn(text_embed.shape).to(text_embed.device)
        for i in range(len(self.ahl_layers)):
            # print(hyper_embed)
            hyper_embed = self.ahl_layers[i](text_embed, visual_embed, hyper_embed)
            if i < len(self.ahl_layers) - 1:
                text_embed = self.trm_layers[i](text_embed, text_embed, text_embed)
        return hyper_embed, text_embed
    
class CrossModalityFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, drop_out = 0.1):
        super(CrossModalityFusion, self).__init__()
        self.transformer = TransformerEncoder(embed_dim, num_heads, layers, drop_out)
        self.extra_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, text_embed, hyper_embed):
        b, n_h, _ = hyper_embed.shape
        b, n_t, _ = text_embed.shape
        extra_token = self.extra_token.expand(b, 1, -1)
        text_embed = torch.cat((extra_token, text_embed), dim=1)
        hyper_embed = torch.cat((extra_token, hyper_embed), dim=1)
        fusion = self.transformer(text_embed, hyper_embed, hyper_embed)
        return fusion
    
class ALMT(MMModel):
    def __init__(self, config:ALMTConfig):
        super(ALMT, self).__init__(config)
        self.config = config
        self.classifier_dim = config.embed_dim
        self.token_len = config.embed_trm_token_len
        self.token = nn.Parameter(torch.randn(1, self.token_len, config.embed_dim))
        self.text_trm = TransformerEncoder(config.embed_dim, config.embed_trm_heads, config.embed_trm_layers, config.drop_out)
        self.image_trm = TransformerEncoder(config.embed_dim, config.embed_trm_heads, config.embed_trm_layers, config.drop_out)
        
        self.hml = HyperModalityLearning(config.embed_dim, config.ahl_heads, config.ahl_layers, config.drop_out)
        self.fusion = CrossModalityFusion(config.embed_dim, config.fusion_trm_heads, config.fusion_layers, config.drop_out)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, self.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.classifier_dim, self.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.classifier_dim, config.num_labels)
        )              
    
    def forward(self, input_id, attention_mask, token_type_ids, image):
        # embedding
        visual_embed = self.vit(image)
        text_embed = self.bert(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        visual_embed = visual_embed['last_hidden_state']
        text_embed = text_embed['last_hidden_state']
        
        text_embed = torch.cat((self.token.expand(text_embed.shape[0], -1, -1), text_embed), dim=1)
        visual_embed = torch.cat((self.token.expand(visual_embed.shape[0], -1, -1), visual_embed), dim=1)
        text_embed = self.text_trm(text_embed, text_embed, text_embed)[:, :self.token_len, :]
        visual_embed = self.image_trm(visual_embed, visual_embed, visual_embed)[:, :self.token_len, :]        
        
        hyper_embed, text_embed = self.hml(text_embed, visual_embed)
        fusion = self.fusion(text_embed, hyper_embed)
        fusion = fusion[:, 0, :]
        output = self.classifier(fusion)
        return output
    
if __name__ == '__main__':
    config = ALMTConfig(3)
    almt_model = ALMT(config)
    from sadatasets import TrainDataset, TestDataset
    from utils import MultiModalTrainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    trainer = MultiModalTrainer(almt_model, train_dataset, test_dataset = test_dataset, compute_metrics=accuracy_score, batch_size=32, num_epochs=10)
    trainer.train()