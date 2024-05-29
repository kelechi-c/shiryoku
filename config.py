class Config:
    num_epochs = 30
    model_output_path = 'shiryoku_icm'
    model_filename = 'shiryoku_vision.pth'
    lr = 0.01
    momentum = 0.9
    batch_size = 64
    top_k = 2
    num_experts = 4
    image_size = 128
    train_size = 0.95
    embed_size = 512
    hidden_size = 1024
    
wandb_config = {
    'num_epochs': 30,
    'lr': 0.01
}