import torch
import re
from collections import Counter
from nltk import tokenize
from torch.utils.data import Dataset, DataLoader, random_split
from utils_functions import read_img, tokenize_text, load_image_captions, image_transforms
from config import Config
from utils_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images, captions = zip(*load_image_captions())

print(f'Images: {len(images)}')
print(f'Captions: {len(captions)}')

captions_vocab = create_vocabulary(captions)

captions_vocab = captions_vocab[0]


class ImageCaptionData(Dataset):
    def __init__(self, images, captions, transforms=None, device=device):
        super().__init__()
        self.images = images
        self.captions = captions
        self.transform = transforms
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # image = read_img(self.images[idx])
        try:
            image = read_img(self.images[idx])
        except Exception as e:
            print(f"Error reading image at index {idx}: {e}")
            return None, None, None
        
        
        if self.transform:
            image = self.transform(image)
            
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        
        caption = tokenize_text(self.captions[idx])
        
        caption = torch.tensor(caption).to(self.device)
        
        length = len(caption)
        
        return image, caption, length



dataset = ImageCaptionData(images=images, captions=captions)

train_size = 0.90 * len(dataset)
val_size = len(dataset) - train_size

train_data, valid_data = random_split(dataset, (train_size, val_size))

train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=Config.batch_size, shuffle=False)
