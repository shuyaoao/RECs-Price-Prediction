"""CNN-BiLSTM-AM model architecture for time series forecasting using PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
from india_wind_solar import config

class AttentionLayer(nn.Module):
    """Attention mechanism for BiLSTM output."""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(units * 2, units)  # BiLSTM output has double the units
        self.V = nn.Linear(units, 1)
        
    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch_size, seq_len, hidden_size*2]
        
        # Calculate attention scores
        score = torch.tanh(self.W(encoder_outputs))  # [batch_size, seq_len, units]
        attention_weights = torch.softmax(self.V(score), dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to encoder outputs
        context_vector = attention_weights * encoder_outputs  # [batch_size, seq_len, hidden_size*2]
        context_vector = torch.sum(context_vector, dim=1)  # [batch_size, hidden_size*2]
        
        return context_vector


class CNNBiLSTMAM(nn.Module):
    """CNN-BiLSTM with Attention Mechanism for time series forecasting."""
    
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH):
        super(CNNBiLSTMAM, self).__init__()
        
        self.cnn_branches = nn.ModuleList()
        for i, (filters, kernel_size) in enumerate(zip(config.CNN_FILTERS, config.CNN_KERNEL_SIZES)):
            branch = nn.Sequential(
                nn.Conv1d(1, filters, kernel_size, padding='same'),
                nn.ReLU(),
                nn.MaxPool1d(config.CNN_POOL_SIZE),
                nn.BatchNorm1d(filters)
            )
            self.cnn_branches.append(branch)
        
        # Calculate the size after CNN branches
        self.cnn_output_size = sum(filters for filters in config.CNN_FILTERS)
        self.sequence_length_after_pooling = sequence_length // config.CNN_POOL_SIZE
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=config.LSTM_UNITS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT_RATE
        )
        
        # Attention layer
        self.attention = AttentionLayer(config.LSTM_UNITS)
        
        # Output layer
        self.output_layer = nn.Linear(config.LSTM_UNITS * 2, 1)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, 1]
        
        # Reshape for Conv1d which expects [batch, channels, length]
        x = x.permute(0, 2, 1)  # [batch_size, 1, sequence_length]
        
        # Apply CNN branches
        cnn_outputs = []
        for branch in self.cnn_branches:
            output = branch(x)  # [batch_size, filters, sequence_length/pool_size]
            cnn_outputs.append(output)
        
        # Concatenate CNN outputs if multiple branches
        if len(cnn_outputs) > 1:
            merged = torch.cat(cnn_outputs, dim=1)  # Concatenate along filter dimension
        else:
            merged = cnn_outputs[0]
        
        # Reshape for LSTM which expects [batch, seq_len, features]
        merged = merged.permute(0, 2, 1)  # [batch_size, sequence_length/pool_size, total_filters]
        
        # Apply BiLSTM
        bilstm_output, _ = self.bilstm(merged)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply attention
        context_vector = self.attention(bilstm_output)  # [batch_size, hidden_size*2]
        
        # Output layer
        output = self.output_layer(context_vector)  # [batch_size, 1]
        
        return output


def build_model(sequence_length=config.SEQUENCE_LENGTH):
    """
    Build the CNN-BiLSTM-AM model.
    
    Args:
        sequence_length: Length of input time series sequence
        
    Returns:
        PyTorch model and optimizer
    """
    model = CNNBiLSTMAM(sequence_length)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    return model, optimizer, criterion


def build_simple_model(sequence_length=config.SEQUENCE_LENGTH):
    """
    Build a simpler CNN-BiLSTM-AM model for testing purposes.
    
    Args:
        sequence_length: Length of input time series sequence
        
    Returns:
        PyTorch model and optimizer
    """
    class SimpleCNNBiLSTMAM(nn.Module):
        def __init__(self, sequence_length):
            super(SimpleCNNBiLSTMAM, self).__init__()
            
            # CNN layer
            self.conv = nn.Conv1d(1, 64, kernel_size=3, padding='same')
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            
            # Calculate the size after pooling
            self.sequence_length_after_pooling = sequence_length // 2
            
            # BiLSTM layer
            self.bilstm = nn.LSTM(
                input_size=64,
                hidden_size=64,
                batch_first=True,
                bidirectional=True
            )
            
            # Attention layer
            self.attention = AttentionLayer(64)
            
            # Output layer
            self.output_layer = nn.Linear(64 * 2, 1)
            
        def forward(self, x):
            # x shape: [batch_size, sequence_length, 1]
            
            # Reshape for Conv1d which expects [batch, channels, length]
            x = x.permute(0, 2, 1)  # [batch_size, 1, sequence_length]
            
            # CNN layers
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            
            # Reshape for LSTM
            x = x.permute(0, 2, 1)  # [batch_size, seq_len, features]
            
            # BiLSTM
            lstm_out, _ = self.bilstm(x)
            
            # Attention
            context = self.attention(lstm_out)
            
            # Output
            output = self.output_layer(context)
            
            return output
    
    model = SimpleCNNBiLSTMAM(sequence_length)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    return model, optimizer, criterion