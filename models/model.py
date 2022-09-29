from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import AutoModel
import numpy as np

class C3PModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.code_encoder = AutoModel.from_pretrained("TerryPei/CL")
        self.comment_encoder = AutoModel.from_pretrained("TerryPei/NL")
    
    def forward(self, batch):
        code_features = self.code_encoder(input_ids=batch['code_token_ids'], 
                                          attention_mask=batch['code_mask_ids'],
                                          position_ids=batch['code_pos_ids']) 
        comment_features = self.comment_encoder(input_ids=batch['comment_token_ids'], 
                                                attention_mask=batch['comment_mask_ids'],
                                                position_ids=batch['comment_pos_ids'])

        batch_size = code_features.view(0)

        if self.config.type.lower() == 'cls':
            code_features = code_features[0][:, 0, :]
            comment_features = comment_features[0][:, 0, :]

        elif self.config.type.lower() == 'eot':
            code_features = code_features.last_hidden_state[torch.arange(batch_size), batch['code_token_ids'].argmax(dim=-1)]
            comment_features = comment_features.last_hidden_state[torch.arange(batch_size), batch['comment_token_ids'].argmax(dim=-1)]
        
        else:
            code_features = torch.argmax(code_features[0], dim=-1)
            comment_features = torch.argmax(comment_features[0], dim=-1)

        code_features = code_features / code_features.norm(dim=-1, keepdim=True)
        comment_features = comment_features / comment_features.norm(dim=-1, keepdim=True)

        return code_features, comment_features

class Config():
    def __init__(self):
        super().__init__()
        self.type = 'EOT' # 'CLS'



if __name__ == '__main__':
    config = Config()
    model = C3PModel(config)
    print(model)