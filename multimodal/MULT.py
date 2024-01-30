# implement of paper "Multimodal Transformer for Unaligned Multimodal Language Sequences"
# https://arxiv.org/abs/1906.00295

import torch
import torch.nn as nn
from .base import MMModel, MultiModalConfig
from .TransformerEncoder import TransformerEncoder
class MULTConfig(MultiModalConfig):
    def __init__(self, num_labels, drop_out = 0.1,
                 embed_dim = 768, num_heads = 8, layers = 3,
                 attn_mask = False, conv1d_kernel_size = 3,):
        super(MULTConfig, self).__init__(num_labels)
        self.output_dim = num_labels
        self.drop_out = drop_out
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_mask = attn_mask
        self.conv1d_kernel_size = conv1d_kernel_size
        

class MULT(MMModel):
    def __init__(self, config:MULTConfig):
        super(MULT, self).__init__(config)
        self.config = config
        # project layers
        self.text_project = nn.Conv1d(config.text_embed_dim, config.embed_dim, config.conv1d_kernel_size)
        self.visual_project = nn.Conv1d(config.visual_embed_dim, config.embed_dim, config.conv1d_kernel_size)
        
        # cross-modal attention
        self.text2visual = TransformerEncoder(config.embed_dim, config.num_heads, config.layers, config.drop_out, config.attn_mask)
        self.visual2text = TransformerEncoder(config.embed_dim, config.num_heads, config.layers, config.drop_out, config.attn_mask)
        # self attention
        self.text2text = TransformerEncoder(config.embed_dim, config.num_heads, 3, config.drop_out, config.attn_mask)
        self.visual2visual = TransformerEncoder(config.embed_dim, config.num_heads, 3, config.drop_out, config.attn_mask)
        
        self.classifier_dim = config.embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, self.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.classifier_dim, self.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.classifier_dim, config.num_labels)
        )
        pass
    
    def forward(self, input_id, attention_mask, token_type_ids, image):
        # embedding
        visual_embed = self.vit(image)
        text_embed = self.bert(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # visual_embed = visual_embed['last_hidden_state'][:, 0, :]
        # text_embed = text_embed['last_hidden_state'][:, 0, :]
        visual_embed = visual_embed['last_hidden_state']
        text_embed = text_embed['last_hidden_state']
        # modal projection if unaligned
        if self.config.text_embed_dim != self.config.embed_dim:
            text_embed = self.text_project(text_embed)
        
        if self.config.visual_embed_dim != self.config.embed_dim:
            visual_embed = self.visual_project(visual_embed)
        
        # cross-modal attention
        visual2text = self.visual2text(text_embed, visual_embed, visual_embed)
        text2visual = self.text2visual(visual_embed, text_embed, text_embed)
        # self attention
        text2text = self.text2text(visual2text, visual2text, visual2text)
        visual2visual = self.visual2visual(text2visual, text2visual, text2visual)
        
        last_text = text2text[:, -1, :]
        last_visual = visual2visual[:, -1, :]
        combined = torch.cat((last_text, last_visual), dim=1)
        output = self.classifier(combined)
        return output
    
    # def get_network(self, self_type='l', layers=-1):
    #     if self_type == 'v':
    #         embed_dim = self.config.text_embed_dim
    #     elif self_type == 'v':
    #         embed_dim = self.config.visual_embed_dim

if __name__ == '__main__':
    config = MULTConfig(3)
    mult_model = MULT(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    from sadatasets import TrainDataset, TestDataset
    from utils import MultiModalTrainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    trainer = MultiModalTrainer(mult_model, train_dataset, test_dataset = test_dataset, compute_metrics=accuracy_score, batch_size=32, num_epochs=20, lr=5e-4)
    trainer.train()
    pass