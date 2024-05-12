import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from shiryoku_v1.dataset_prep import ImageCaptionData

def training_loop(model, train_data, lossfn, optimizer, config, epochs):
    model.train()


def validate_step(model, train_data, lossfn, optimizer, config):
    return 
    
