import torch

MODEL_ID = 'CNN_BiLSTM_AM'

RANDOM_SEED = 42

# Model parameters for CNNBiLSTMAM
SEQUENCE_LENGTH = 5  # Number of time steps to look back
# SEQUENCE_LENGTH = 5
BATCH_SIZE = 64
# BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
PATIENCE = 50  # Early stopping patience

# CNN parameters
CNN_FILTERS = [64, 128]
# CNN_FILTERS = [64, 128]
CNN_KERNEL_SIZES = [3, 5]
CNN_POOL_SIZE = 2

# LSTM parameters
LSTM_UNITS = 256
# LSTM_UNITS = 256
BIDIRECTIONAL = True

# Attention parameters
ATTENTION_UNITS = 128
# ATTENTION_UNITS = 128

SAVED_WEIGHTS = "cnn_bilstm_am_model.h5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device for PyTorch computations