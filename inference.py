import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image as pillow_image
from shiryoku_v1.shiryoku_model import ImageTextModel
from utils_functions import read_img
from shiryoku_v1.dataset_prep import captions_vocab

shiryoku_model = ImageTextModel()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = 'shiryoku.pth'

shiryoku_model.load_state_dict(torch.load(model_path))
shiryoku_model.eval()

def sample_run(image, model, device):
    image = read_img(image)
    model.eval()
    
    model = model.to(device)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    image_tensor = image.to(device)
    
    model_prediction = model.vision_encoder(image_tensor)
    model_prediction = model_prediction.unsqueeze(0)
    model_prediction = model_prediction.to(device)
    
    sampled_ids = model.text_decoder.sample(model_prediction)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = captions_vocab[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    caption = " ".join(sampled_caption)
    
    print(caption)
    image = pillow_image.open(image)
    plt.imshow(np.asarray(image))


sample_run('sample_image.jpg', shiryoku_model, device)