import cv2
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import Tokenizer

tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

def read_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return image


def tokenize_text(text_input):
    text_tokens = tokenizer(text_input, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
    
    return text_tokens


