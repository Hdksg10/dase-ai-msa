# implement of paper "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors"
# https://arxiv.org/abs/1806.00064

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .base import MMModel, MultiModalConfig

class LMFConfig(MultiModalConfig):
    def __init__(self, num_labels, rank = 10):
        super(LMFConfig, self).__init__(num_labels)
        self.rank = rank
        self.output_dim = num_labels

class LMF(MMModel):
    def __init__(self, config:LMFConfig):
        super(LMF, self).__init__(config)
        self.output_dim = config.output_dim
        self.classifier = None
        self.post_fusion_dropout = nn.Dropout(p=config.drop_out)
        self.image_factor = nn.Parameter(torch.Tensor(config.rank, config.visual_embed_dim + 1, config.output_dim))
        self.text_factor = nn.Parameter(torch.Tensor(config.rank, config.text_embed_dim + 1, config.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, config.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, config.output_dim))
        xavier_normal_(self.image_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    
    def forward(self, input_id, attention_mask, token_type_ids, image):
        visual_embed = self.vit(image)
        text_embed = self.bert(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        visual_embed = visual_embed['last_hidden_state'][:, 0, :]
        text_embed = text_embed['last_hidden_state'][:, 0, :]
        
        batch_size = visual_embed.shape[0]
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(visual_embed).to(image.device)
        _visual_embed = torch.cat((visual_embed, add_one), dim=1)
        _text_embed = torch.cat((text_embed, add_one), dim=1)
        
        fusion_image = torch.matmul(_visual_embed, self.image_factor)
        fusion_text = torch.matmul(_text_embed, self.text_factor)
        fusion_zy = fusion_image * fusion_text
        
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        
        return output

if __name__ == '__main__':
    lmf_config = LMFConfig(3, 20)
    lmf_model = LMF(lmf_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    from sadatasets import TrainDataset, TestDataset
    from utils import MultiModalTrainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    trainer = MultiModalTrainer(lmf_model, train_dataset, test_dataset = test_dataset, compute_metrics=accuracy_score, batch_size=32, num_epochs=5, lr=5e-3)
    trainer.train()