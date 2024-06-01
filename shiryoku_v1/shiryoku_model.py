import torch.nn as nn                                                
from image_encoder import ResConvNet
from shiryoku_v1.text_model import TextRNNDecoder
from shiryoku_v1.dataset_prep import vocab_size
from config import Config
from utils_functions import *


class ImageTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ResConvNet(embed_size=Config.embed_size)
        self.text_decoder = TextRNNDecoder(
            vocab_size=vocab_size,
            embed_dim=Config.embed_size,
            hidden_size=Config.hidden_size,
        )

    def forward(self, image, text, lengths):
        visual_tokens = self.vision_encoder(image)
        decoder_output = self.text_decoder(visual_tokens, text, lengths)

        return decoder_output
