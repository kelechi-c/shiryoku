import torch
import math
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, random_split
from utils_functions import read_img, tokenize_text
from config import Config
from utils_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_file = "/kaggle/input/md-data-500/moondream2.csv"

train_data = load_mit_dataset()

image_strings, captions = zip(*retrieve_data(train_data))


print(f'Images: {len(image_strings)}')
print(f'Captions: {len(captions)}')

captions_vocab = create_vocabulary(captions)

caption_vocab = captions_vocab[0]
idx2word = captions_vocab[1]

vocab_size = len(caption_vocab)


unk_token = "_"
unk_idx = caption_vocab.get(unk_token, len(caption_vocab))


def tokenize_caption(caption, vocab):
    caption_tokens = tokenize_text(caption)
    caption_indices = []
    for token in caption_tokens:
        caption_indices.append(vocab.get(token, unk_idx))

    return caption_indices


class ImageCaptionData(Dataset):
    def __init__(self, images, captions, captions_vocab=caption_vocab, transforms=None, device=device):
        super().__init__()
        self.images = images
        self.captions = captions
        self.transform = transforms
        self.device = device
        self.vocab = captions_vocab
        self.max_caption_length = max(len(tokenize_caption(caption, captions_vocab)) for caption in captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images
        try:
            image = cv_decode_image(self.images[idx])
        except Exception as e:
            print(f"Error reading image at index {idx}: {e}")

        if self.transform:
            image = self.transform(image)

        torch.from_numpy(image)
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        image = rearrange(image, "h w c -> c h w")

        # caption
        vocab = self.vocab
        caption = []
        caption_tokens = tokenize_text(self.captions[idx])
        caption.append(vocab('<start>'))
        caption.append([vocab(token) for token in caption_tokens])
        caption.append(vocab('<end>'))
        caption = torch.tensor(caption).to(self.device)

        length = len(caption)

        return image, caption, length


dataset = ImageCaptionData(images=image_strings, captions=captions)

train_size = math.floor(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_data, valid_data = random_split(dataset, (train_size, val_size))

train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=Config.batch_size, shuffle=False)
