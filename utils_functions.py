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


def read_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return image


def get_image_from_url(url):
    try:
        url_content = requests.get(url, stream=True).raw
        image = pillow_image.open(url_content)
        return image
    except Exception as e:
        print(e)


def load_image(url, output_path=None):
    try:
        # Attempt to load image from URL
        url_content = requests.get(url, stream=True).raw
        image = pillow_image.open(url_content)
        return image

    except Exception as e:

        print(f"Error loading image: {url}, {e}")
        return None


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


def fetch_moondream_dataset():
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset['train'][:1000000] # type: ignore

    image_urls = md_data["url"] # type: ignore
    descriptions = md_data["moondream2_caption"] # type: ignore

    for url, desc in zip(image_urls, descriptions):
        yield url, desc


def load_image_captions():
    data = []
    for image_url, desc in tqdm(fetch_moondream_dataset()):
        try:
            image = load_image(image_url)
            caption = desc.lower()
            if image is not None:
                data.append((image, caption))
        except Exception as e:
            print(f"Error loading image/caption: {image_url} + {e}")
            continue

    return data
