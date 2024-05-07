import torch
from torch.utils.data import DataLoader, Dataset, random_split
import tokenizers

class CaptionDataset(Dataset):
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return self
    
    def __getitem__(self, index):
        return 