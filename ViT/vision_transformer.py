import torch 
import torch.nn as nn
from einops import repeat

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
    

class Encoder_block(nn.Module):
    def __init__(self, model_dim, n_heads=8, ff_ratio=4, drop_rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attention = Attention(model_dim, n_heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForwardLayer(model_dim, model_dim * ff_ratio, model_dim, dropout=drop_rate)
        
    def forward(self, x):
        x1 = self.ln1(x)
        x2 = x + self.attention(x1)
        x = x + self.ln2(self.feed_forward(x2))
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, depth, model_dim, n_heads=8, ff_ratio=4, drop_rate=0.0):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [Encoder_block(model_dim, n_heads, ff_ratio, drop_rate) for _ in range(depth)]
        )
        
    def forward(self, x):
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
        
        return x
    
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=100, embed_dim=768, depth=12, n_heads=12, ff_ratio=4, drop_rate=0.0):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        self.image_patches = ImagePatching(in_channels, embed_dim, self.patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.class_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.2)
        nn.init.trunc_normal_(self.class_embed, std=0.2)
        
        self.drop = nn.Dropout(p=drop_rate)
        self.transformer = Transformer(depth, embed_dim, n_heads, ff_ratio, drop_rate)
        self.normalize = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, num_classes)
        self.latent = nn.Identity()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, v):
        if isinstance(v, nn.Linear):
            nn.init.trunc_normal_(v.weight, std=0.2)
            if isinstance(v, nn.Linear) and v.bias is not None:
                nn.init.constant_(v.bias, 0)
                
        elif isinstance(v, nn.LayerNorm):
            nn.init.constant_(v.bias, 0)
            nn.init.constant_(v.weight, 1.0)
            
    
    def forward(self, x):
        b, c, _ = x.shape
        # class_token = self.class_embed.expand(b, -1, -1)
        class_tokens = repeat(self.class_embed, '1 1 d -> b 1 d', b=b)
        
        x = self.image_patches(x)
        x = torch.cat((class_tokens, x), dim=1)
        x += self.pos_embedding[:, : (c + 1)]
        x = self.transformer(self.drop(x))
        x = self.normalize(x)
        x = self.latent(x)
        
        x_out = self.output_layer(x[:, 0])
        
        return x_out