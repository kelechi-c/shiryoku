from re import T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from shiryoku_v1.dataset_prep import train_loader, valid_loader
from utils_functions import *
from shiryoku_v1.image_encoder import ConvNetEncoder, PretrainedConvNet
from shiryoku_v1.text_model import TextRNNDecoder
from config import Config
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = PretrainedConvNet(embed_size=Config.embed_size)
decoder = TextRNNDecoder(vocab_size=Config.vocab_size, embed_dim=Config.embed_size, hidden_size=Config.hidden_size)
criterion = nn.CrossEntropyLoss()
parameters = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batch_norm.parameters())
optimizer = optim.Adam(params=parameters, lr=Config.lr)
epochs = Config.num_epochs

def training_loop(train_loader, lossfn, optimizer, epochs=epochs):
    for epoch in tqdm(range(epochs)):
        print(f'Training epoch {epoch}')
        for _, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = lossfn(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backwards()
            optimizer.step()

            print(f'Epoch {epoch} of {epochs}, loss: {loss.item():.4f}')
            
        print(f"Epoch {epoch} complete")

def validate_step(model, valid_loader, lossfn, optimizer, config):
    return 
