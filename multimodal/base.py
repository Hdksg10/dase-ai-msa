from transformers import BertModel, BertConfig
from transformers import BeitModel, BeitConfig, ViTConfig, ViTModel


from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MultiModalConfig(PretrainedConfig):
    def __init__(self, num_labels, drop_out=0.1):
        self.num_labels = num_labels
        self.bert_config = BertConfig()
        # self.beit_config = AutoConfig.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.beit_config = ViTConfig()
        self.drop_out = drop_out
        # set embed size from config
        self.text_embed_dim = self.bert_config.hidden_size
        self.visual_embed_dim = self.beit_config.hidden_size
        self.hiden_dim = self.text_embed_dim + self.visual_embed_dim
        
class MMModel(PreTrainedModel):
    def __init__(self, config:MultiModalConfig):
        super(MMModel, self).__init__(config)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(config.drop_out),
            nn.Linear(config.hiden_dim, config.hiden_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(config.hiden_dim, config.hiden_dim),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(config.hiden_dim, config.num_labels),
            nn.ReLU()
        )

    def forward(self, input_id, attention_mask, token_type_ids, image):
        visual_embed = self.vit(image)
        text_embed = self.bert(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        visual_embed = visual_embed['last_hidden_state'][:, 0, :]
        text_embed = text_embed['last_hidden_state'][:, 0, :]
        # print(text_embed)
        embed = torch.cat((visual_embed, text_embed), dim=1)
        output = self.classifier(embed)
        return output
        # text_embeds = self.bert(**output['text'])



if __name__ == '__main__':
    mm_config = MultiModalConfig(3)
    mm_model = MMModel(mm_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from sadatasets import TrainDataset, TestDataset
    from utils import MultiModalTrainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    trainer = MultiModalTrainer(mm_model, train_dataset, test_dataset = test_dataset, compute_metrics=accuracy_score, batch_size=32, num_epochs=20)
    trainer.train()
