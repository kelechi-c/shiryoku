import torch
import re
from collections import Counter
from nltk import tokenize
from torch.utils.data import Dataset, DataLoader, random_split
from utils_functions import read_img, tokenize_text, get_moondream_dataset, image_transforms
from config import Config
from utils_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images, captions = get_moondream_dataset()

captions_vocab = create_vocabulary(captions)

class ImageCaptionData(Dataset):
    def __init__(self, images, captions, transforms, device):
        super().__init__()
        self.images = images
        self.captions = captions
        self.transform = transforms
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = read_img(self.images[idx])
        if self.transform:
            image = self.transform(image)
            
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        
        caption = tokenize_text(self.captions[idx])
        
        caption = torch.tensor(caption).to(self.device)

        return image, caption


dataset = ImageCaptionData(images=images, captions=captions_vocab, transforms=image_transforms, device=device)

train_size = 0.95 * len(dataset)
val_size = len(dataset) - train_size

train_data, valid_data = random_split(dataset, (train_size, val_size))

train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=Config.batch_size, shuffle=False)
