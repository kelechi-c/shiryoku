from altair import Config
from networkx import dispersion
import torch
from matplotlib import pyplot as plt
from matplotlib import image as pyimg
import cv2
import numpy as np
from einops import rearrange
from PIL import Image as pillow_image
from shiryoku_v1.shiryoku_model import ImageTextModel
from utils_functions import display_image, read_img
from config import Config
from shiryoku_v1.dataset_prep import idx2word

shiryoku_model = ImageTextModel()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = 'shiryoku.pth'


shiryoku_model.load_state_dict(torch.load(model_path))
shiryoku_model.eval()
def sample_run(image_file, model, device):
    image = pillow_image.open(image_file)
    image = read_img(image)
    model.eval()

    model = model.to(device)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    image = image.unsqueeze(0)
    image = rearrange(image, "b h w c -> b c h w")

    image_tensor = image.to(device)

    model_prediction = model.vision_encoder(image_tensor)

    model_prediction = model_prediction.squeeze(1)

    model_prediction = model_prediction.to(device)
    print(model_prediction.shape)

    sampled_ids = model.text_decoder.sample(model_prediction)

    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break

    caption = " ".join(sampled_caption)

    print(caption)
    print(f"Image file: {image_file}")
    
    display_image(image_file)
    


sample_run('sample_image.jpg', shiryoku_model, device)
