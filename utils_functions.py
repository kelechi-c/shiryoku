import cv2
import nltk
from datasets import load_dataset
import requests
import PIL.Image as pillow_image
from nltk import tokenize
from torchvision import transforms
import re
from collections import Counter
from tqdm.auto import tqdm
from config import Config
import numpy as np
from matplotlib import pyplot as plt


# For image data
def image_transforms():
    trasnformed_image = transforms.Compose(
        [
            transforms.RandomCrop(Config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return trasnformed_image


def read_img(image_data):
    if isinstance(image_data, str):
        # Assume image_data is a file path
        image = cv2.imread(image_data)
    elif isinstance(image_data, bytes):
        # Assume image_data is binary data
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image_data, pillow_image.Image):
        # Assume image_data is a PIL Image object
        image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        
    if image is None:
        raise ValueError("Could not read the image data.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
    return image


def load_image(url):
    try:
        # Attempt to load image from URL
        url_content = requests.get(url, stream=True).raw
        image = pillow_image.open(url_content)
        return image

    except Exception as e:

        print(f"Error loading image: {url}, {e}")
        return None
    

def display_image(image):
    plt.figure(figsize=(50, 50))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# For text data
def tokenize_text(text_input):
    text_tokens = tokenize.word_tokenize(str(text_input).lower())

    return text_tokens


def preprocess_text(text):
    text = re.sub(r"[^a-z0-9\s]", "", text)  
    tokens = tokenize_text(text)  
    tokens = [
        t for t in tokens if t not in nltk.corpus.stopwords.words("english")
    ] 
    return tokens


def create_vocabulary(text_dataset):
    all_tokens = []
    for text in text_dataset:
        tokens = preprocess_text(text)
        all_tokens.extend(tokens)

    vocab_counter = Counter(all_tokens)
    vocab = [
        word for word, count in vocab_counter.most_common() if count >= 1
    ]
    vocab = ["<pad>", "<start>", "<end>", "<unk>"] + vocab

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    return word_to_idx, idx_to_word


def load_moondream_dataset():
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset['train'][:1000000] # type: ignore

    image_urls = md_data["url"] # type: ignore
    descriptions = md_data["moondream2_caption"] # type: ignore
    
    for url, desc in tqdm(zip(image_urls, descriptions)):
        try:
            image = load_image(url)
            caption = desc.lower()

            if image is not None:
                yield (image, caption)

        except:
            print(f"Error loading image/caption")
            
            continue
