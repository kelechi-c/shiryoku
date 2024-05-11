from altair import Data
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, Image
from utils_functions import read_img, tokenize_text, get_dataset
from config import Config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_images, train_captions = get_dataset('train')
valid_images, valid_captions = get_dataset('test')


class ImageCaptionData(Dataset):
    def __init__(self, images, captions, device):
        super().__init__()
        self.images = images
        self.captions = captions
        self.device = device
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = read_img(self.images[idx])
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        caption = tokenize_text(self.captions[idx])
        caption = torch.tensor(caption).to(self.device)
        
        return image, caption


train_dataset = ImageCaptionData(images=train_images, captions=train_captions, device=device)
valid_dataset = ImageCaptionData(images=valid_images, captions=valid_captions, device=device)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)
