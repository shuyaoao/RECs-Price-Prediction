"""Training script for CNN-BiLSTM-AM model using PyTorch."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from india_wind_solar import config
from india_wind_solar.CNN_BiLSTM_AM.data.dataloader import prepare_data_for_model, get_data_loaders
from india_wind_solar.CNN_BiLSTM_AM.models.cnn_bilstm_am import build_model
from india_wind_solar.CNN_BiLSTM_AM.utils.visualisation import plot_training_history

def set_seed(seed=config.RANDOM_SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train():
    """Train the CNN-BiLSTM-AM model."""
    # Set seed for reproducibility
    set_seed()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Use the DataLoader from dataloader.py if available
    try:
        print("Loading data using DataLoader...")
        train_loader, val_loader, _, scaler, train_dates, val_dates, _ = get_data_loaders()
        
        # Move validation data to device (for full batch evaluation)
        X_val, y_val = next(iter(DataLoader(val_loader.dataset, batch_size=len(val_loader.dataset))))
        X_val, y_val = X_val.to(device), y_val.to(device)
        
    except (NameError, AttributeError):
        # Fallback to direct data loading if get_data_loaders is not available
        print("Preparing data directly...")
        X_train, y_train, X_val, y_val, _, _, _, _, _, _ = prepare_data_for_model()
        
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.FloatTensor(y_val).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True  # Speeds up host to device transfers
        )
    
    print(f"Training batches: {len(train_loader)}")
    
    # Create model, optimizer, and loss function
    print("Building model...")
    model, optimizer, criterion = build_model()
    model = model.to(device)
    
    # Create save directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_SAVE_DIR)
    os.makedirs(model_dir, exist_ok=True)
    
    # Training parameters
    epochs = config.EPOCHS
    patience = config.PATIENCE
    best_val_loss = float('inf')
    wait = 0
    best_model = None
    
    # For tracking progress
    train_losses = []
    val_losses = []
    learning_rates = []
    current_lr = config.LEARNING_RATE
    
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
            }, os.path.join(model_dir, config.MODEL_NAME))
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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RESULTS_DIR)
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
    
    print(f"Model saved to {os.path.join(model_dir, config.MODEL_NAME)}")
    
    return model, history

if __name__ == "__main__":
    train()