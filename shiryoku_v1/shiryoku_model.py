import torch
import torch.nn as nn  
from einops import rearrange                                                  

from image_encoder import ConvNetEncoder
from shiryoku_v1.text_model import TextRNNDecoder
from mixtral_of_experts import MixtralOfExpertsLayer
from ViT.vision_transformer import VisionTransformer
from config import Config
from utils_functions import *


class ModalityFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ConvNetEncoder()
        self.text_embedding = TextRNNDecoder()

    def forward(self, x):
        visual_tokens = self.vision_encoder(x)
        text_tokens = self.text_embedding(x)
        
        token_fusion = torch.cat((visual_tokens, text_tokens), dim=1) 
        
        return token_fusion
    

