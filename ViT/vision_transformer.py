import torch 
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipTextModel, SiglipModel

class FeedForwardLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.feed_forward(x)
        
        return self.dropout(x)
    
    

class Attention(nn.Module):
    def __init__(self, model_dim, n_heads=8, dropout=0.0, project_dropout=0.0):
        super().__init__()
        self.num_heads = n_heads
        self.scale = 1.0 / model_dim**0.5
        
        self.qkv = nn.Linear(model_dim, model_dim*3, bias=False)
        self.attend_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(project_dropout)
        )
        
    def forward(self, x):
        a, b, c = x.shape
        qkv = self.qkv(x).reshape(a, b, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        dot_prod = (q @ k.transpose(-2, -1)) * self.scale
        attention_score_nodrop = dot_prod.softmax(dim=1)
        attention_score = self.attend_dropout(attention_score_nodrop)

        x = (attention_score @ v).transpose(1, 2).reshape(a, b, c)
        x = self.output_layer(x)
        
        return x
        
class ImagePatching(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, img):
        image_patches = self.patch_embed(img).flatten(2).tra
        return image_patches
    

