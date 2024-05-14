import cv2
from transformers import AutoTokenizer
from datasets import load_dataset, Image
import requests
import PIL.Image as pillow_image
from nltk import tokenize

tokenizer = tokenize.
# tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

def read_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return image

def get_image_from_url(url):
    url_content = requests.get(url, stream=True).raw
    image = pillow_image.open(url_content)
    
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

def get_moondream_dataset():
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset['train'] # type: ignore
    
    image_urls = md_data['url'] # type: ignore
    descriptions = md_data['moondream2_caption'] # type: ignore
    
    images = [get_image_from_url(img_url) for img_url in image_urls]
    captions = [caption for caption in descriptions]
    
    return images, captions