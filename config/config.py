import torch

CONFIG = {
    "seq_length": 10, 
    "hidden_size": 64,
    "num_layers": 2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
