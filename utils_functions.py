import cv2
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from datasets import load_dataset, Image

tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

def read_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return image


def tokenize_text(text_input):
    text_tokens = tokenizer.encode(text_input, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
    
    return text_tokens

def get_dataset(data_split):
    docci_dataset = load_dataset("google/docci")

    data = docci_dataset[data_split]  # type: ignore

    image = data["image"] 
    descriptions = data["description"] 

    images = [Image(x) for x in image]
    img_captions = [caption for caption in descriptions]
    
    return images, img_captions
