import cv2
import nltk
from transformers import AutoTokenizer
from datasets import load_dataset, Image
import requests
import PIL.Image as pillow_image
from nltk import tokenize
from torchvision import transforms
import re
from io import BytesIO
from collections import Counter
from config import Config

# tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

# For image data
def image_transforms(image_file):
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

        # response = requests.get(data)
        # image = pillow_image.open(BytesIO(response.content))
    except:
        try:
            # Attempt to load image from raw data
            image = requests.get(url, stream=True).raw
            img = pillow_image.open(image)
            img = img.convert('RGB')  # Ensure RGB mode for consistent conversion
            if not output_path:
                output_path = f"{image.rsplit('.', 1)[0]}.jpeg"
                
            img.save(output_path, 'JPEG')
            return img
        
        except OSError:
            print(f"Error converting image: {image}")
            return None
        
        except:
            print(f"Error loading image: {url}")
            return None
        
    print(f'success loading {url}')
    return image    


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


# for fetching the required datasets
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
