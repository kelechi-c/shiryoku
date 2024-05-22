from logging import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from shiryoku_v1.shiryoku_model import ImageTextModel
from shiryoku_v1.dataset_prep import train_loader, valid_loader
from utils_functions import *
from config import Config, wandb_config
from tqdm.auto import tqdm
import os

import wandb
wandb.login()

wandb.init(project="shiryoku_vision", config=wandb_config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

shiryoku_model = ImageTextModel()
shiryoku_model = shiryoku_model.to(device)

wandb.watch(shiryoku_model, log_freq=100)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=shiryoku_model.parameters(), lr=Config.lr)
epochs = Config.num_epochs

os.mkdir(Config.model_output_path)

def train_step(train_loader, model):
    total_correct = 0
    total_samples = 0
    
    
    for _, (images, captions, lengths) in tqdm(enumerate(train_loader)):
        
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        model_outputs = model(images, captions, lengths)
        
        _, predicted = torch.max(model_outputs, 1)

        # Update the running total of correct predictions and samples
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        
        accuracy = 100 * total_correct / total_samples
        train_loss = criterion(model_outputs, targets)
        
        model.zero_grad()
        train_loss.backwards()
        optimizer.step()
    
    return accuracy, train_loss

def validation_step(model, valid_loader):
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval() 
    
    with torch.no_grad():  
        for _, (images, captions, lengths) in tqdm(enumerate(valid_loader)):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Update the running total of correct predictions and samples
            model_outputs = model(images, captions, lengths)
            
            _, predicted = torch.max(model_outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            val_accuracy = 100 * total_correct / total_samples
            
            loss = criterion(model_outputs, targets)
            val_loss += loss.item()
    
    return val_accuracy, val_loss


def training_loop(model, train_loader, valid_loader, epochs=epochs):
    model.train()
    for epoch in tqdm(range(epochs)):
        print(f'Training epoch {epoch}')
        train_acc, train_loss = train_step(train_loader, model)
        valid_acc, valid_loss = validation_step(model, valid_loader)

        print(
            f"Epoch {epoch} of {epochs}, train_accuracy: {train_acc:.2f}, train_loss: {train_loss.item():.4f}, valid_accuracy: {valid_acc:.2f}, val_loss: {train_loss.item():.2f}"
        )

        torch.save(model.state_dict(), os.path.join(os.getcwd(), Config.model_output_path, f'caption_model_{epoch}.pth'))

        wandb.log({"accuracy": train_acc, "loss": train_loss, "valid_accuracy": valid_acc, "val_loss": valid_loss})
        print(f"Epoch {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, accuracy: {train_acc:.2f}, train_loss: {train_loss.item():.4f},valid_accuracy: {valid_acc:.2f}, valid_loss: {valid_loss:.4f}"
    )
    
    torch.save(model.state_dict(), os.path.join(os.getcwd(), Config.model_output_path, f'{Config.model_filename}'))


training_loop(shiryoku_model, train_loader, valid_loader)
