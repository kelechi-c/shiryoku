import cv2
import os
import nltk
from datasets import load_dataset
from base64 import b64decode
from io import BytesIO
import PIL.Image as pillow_image
from nltk import tokenize
from torchvision import transforms
import re
from collections import Counter
from tqdm.auto import tqdm
from config import Config
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# For image data
def image_transforms():
    transformed_image = transforms.Compose(
        [
            transforms.RandomCrop(Config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return transformed_image


def read_img(image_data):
    image = cv2.imread(image_data)
        
    if image is None:
        raise ValueError("Could not read the image data.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
    return image


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

# Data from prepared dataset


def load_images_from_directory(csv_file):
    image_folder = ''
    data = pd.read_csv(csv_file).drop_duplicates(keep='first')
    paths = data['image_path']
    
    captions = data['caption']

    image_paths = [os.path.join(image_folder, file) for file in paths]
    
    return image_paths, captions


def load_mit_dataset():

    ds_name = "coco"  # change the dataset name here
    dataset = load_dataset("MMInstruction/M3IT", ds_name)
    train_set = dataset["train"] # type: ignore
    validation_set = dataset["validation"] # type: ignore

    return train_set, validation_set


def cv_decode_image(image64_str):
    image = BytesIO(b64decode(image64_str))
    image = pillow_image.open(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))
    image = np.array(image, dtype=np.float32) / 255.0
    
    return image


def retrieve_data(dataset):

    for k in tqdm(range(len(dataset))):

        _, _, image, _ = dataset[k]

        captext = dataset[k]["outputs"]

        yield (image, captext)
