import torch
import torch.nn as nn

class ConcatFusionLayer(nn.Module):
    def __init__(self, text_embed_size, visual_embed_size, output_size, fc_layers=1):
        super(ConcatFusionLayer, self).__init__()
        self.concat_size = text_embed_size + visual_embed_size
        layers = []
        for i in range(fc_layers):
            layers.append(nn.Linear(self.concat_size if i == 0 else output_size, output_size))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    text_embed_size = 768
    visual_embed_size = 768
    output_size = 1024
    fc_layers = 2
    concat = ConcatFusionLayer(text_embed_size, visual_embed_size, output_size, fc_layers)
    text_embeds = torch.randn(1, 768)
    visual_embeds = torch.randn(1, 768)
    concat_input = torch.cat((text_embeds, visual_embeds), dim=1)
    concat_output = concat(concat_input)
    print(concat_output.shape)