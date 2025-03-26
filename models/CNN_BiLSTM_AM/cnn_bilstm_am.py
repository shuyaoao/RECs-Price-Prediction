"""CNN-BiLSTM-AM model architecture for time series forecasting using PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from models.dataloader import CNN_BiLSTM_AM_DataLoader
from model_config import CNN_BiLSTM_AM_config
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils import plot_training_history
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import plot_predictions
import pandas as pd


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
    
    def __init__(self, data_config, model_config):
        super(CNNBiLSTMAM, self).__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.dataloader = CNN_BiLSTM_AM_DataLoader(self.data_config, self.model_config)
        
        self.cnn_branches = nn.ModuleList()
        for i, (filters, kernel_size) in enumerate(zip(self.model_config.CNN_FILTERS, self.model_config.CNN_KERNEL_SIZES)):
            branch = nn.Sequential(
                nn.Conv1d(1, filters, kernel_size, padding='same'),
                nn.ReLU(),
                nn.MaxPool1d(self.model_config.CNN_POOL_SIZE),
                nn.BatchNorm1d(filters)
            )
            self.cnn_branches.append(branch)
        
        # Calculate the size after CNN branches
        self.cnn_output_size = sum(filters for filters in self.model_config.CNN_FILTERS)
        self.sequence_length_after_pooling = self.model_config.SEQUENCE_LENGTH // self.model_config.CNN_POOL_SIZE
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=self.model_config.LSTM_UNITS,
            batch_first=True,
            bidirectional=True,
            dropout=self.model_config.DROPOUT_RATE
        )
        
        # Attention layer
        self.attention = AttentionLayer(self.model_config.LSTM_UNITS)
        
        # Output layer
        self.output_layer = nn.Linear(self.model_config.LSTM_UNITS * 2, 1)
        
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
    
    def set_seed(self):
        """Set seed for reproducibility."""
        random.seed(self.model_config.RANDOM_SEED)
        np.random.seed(self.model_config.RANDOM_SEED)
        torch.manual_seed(self.model_config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.model_config.RANDOM_SEED)
            torch.cuda.manual_seed_all(self.model_config.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self):
        """Train the CNN-BiLSTM-AM model."""
        # Set seed for reproducibility
        self.set_seed()
        
        # Set device
        device = torch.device(self.model_config.DEVICE)
        print(f"Using device: {device}")
        
        # Use the DataLoader from dataloader.py if available
        try:
            print("Loading data using DataLoader...")
            train_loader, val_loader, _, scaler, train_dates, val_dates, _ = self.dataloader.get_data_loaders()
            
            # Move validation data to device (for full batch evaluation)
            X_val, y_val = next(iter(DataLoader(val_loader.dataset, batch_size=len(val_loader.dataset))))
            X_val, y_val = X_val.to(device), y_val.to(device)
            
        except (NameError, AttributeError):
            # Fallback to direct data loading if get_data_loaders is not available
            print("Preparing data directly...")
            X_train, y_train, X_val, y_val, _, _, _, _, _, _ = self.dataloader.prepare_data_for_model()
            
            # Convert numpy arrays to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.model_config.BATCH_SIZE, 
                shuffle=True,
                num_workers=self.model_config.NUM_WORKERS,
                pin_memory=True  # Speeds up host to device transfers
            )
        
        print(f"Training batches: {len(train_loader)}")
        
        # Create model, optimizer, and loss function
        print("Building model...")
        model = self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.model_config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Create save directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID)
        os.makedirs(model_dir, exist_ok=True)
        
        # Training parameters
        epochs = self.model_config.EPOCHS
        patience = self.model_config.PATIENCE
        best_val_loss = float('inf')
        wait = 0
        best_model = None
        
        # For tracking progress
        train_losses = []
        val_losses = []
        learning_rates = []
        current_lr = self.model_config.LEARNING_RATE
        
        # Train model
        print("Training model...")
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
                val_losses.append(val_loss)
            
            # Learning rate tracking
            learning_rates.append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                best_model = model.state_dict()
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(model_dir, self.model_config.SAVED_WEIGHTS))
                print(f"Model saved at epoch {epoch+1}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Reduce learning rate if no improvement
                if wait % (patience // 2) == 0:
                    current_lr *= 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    print(f"Learning rate reduced to {current_lr}")
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
    
        # Plot training history
        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
        os.makedirs(results_dir, exist_ok=True)
        
        history = {
            'loss': train_losses,
            'val_loss': val_losses,
            'lr': learning_rates
        }
        
        plot_training_history(
            history,
            save_path=os.path.join(results_dir, 'training_history.png')
        )
        
        print(f"Model saved to {os.path.join(model_dir, self.model_config.MODEL_ID)}")
        
        return
    
    def evaluate(self):
        """Evaluate the CNN-BiLSTM-AM model on test data."""
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("Preparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_dates, val_dates, test_dates = self.dataloader.prepare_data_for_model()
        
        # Convert test data to PyTorch tensor
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        print(f"Test data shape: {X_test.shape}")
        
        # Load model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID, self.model_config.SAVED_WEIGHTS)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        model = self.to(device)
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_predictions = test_predictions.cpu().numpy()
        
        # Reshape predictions
        test_predictions = test_predictions.flatten()
        
        # Inverse transform to get original scale
        test_predictions_rescaled = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
        mask = y_test_rescaled != 0
        mape = 100 * np.mean(np.abs((y_test_rescaled[mask] - test_predictions_rescaled[mask]) / y_test_rescaled[mask]))
        
        # Print metrics
        print("\nTest Set Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot predictions
        test_dates_array = np.array(test_dates)[self.model_config.SEQUENCE_LENGTH:]
        
        plot_predictions(
            y_test,
            test_predictions,
            test_dates_array,
            scaler=scaler,
            title='CNN-BiLSTM-AM Test Predictions',
            save_path=os.path.join(results_dir, 'test_predictions.png')
        )
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE'],
            'Value': [mse, rmse, mae, mape]
        })
        metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)
        print(f"Metrics saved to {os.path.join(results_dir, 'test_metrics.csv')}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'Date': test_dates_array,
            'Actual': y_test_rescaled,
            'Predicted': test_predictions_rescaled
        })
        predictions_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(results_dir, 'test_predictions.csv')}")
        
        return mse, rmse, mae, mape, predictions_df