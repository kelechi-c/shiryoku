import torch.optim as optim
import torch
from transformers import SiglipTextModel
from torch.utils.data import DataLoader
from dataset_prep import ImageCaptionData

def training_loop(model, train_data, lossfn, optimizer, config, epochs):
    model.train()
    
    
