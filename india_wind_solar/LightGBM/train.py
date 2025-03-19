"""Training script for LightGBM model."""
import os
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from india_wind_solar import config
from india_wind_solar.LightGBM.data.dataloader import prepare_data_for_model
from india_wind_solar.LightGBM.models.lightgbm import build_model, save_model
from india_wind_solar.LightGBM.utils.visualisation import plot_feature_importance, plot_predictions, plot_training_vs_validation_loss

def set_seed(seed=config.RANDOM_SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def train():
    """Train the LightGBM model."""
    # Set seed for reproducibility
    set_seed()
    
    print("Preparing data...")
    X_train, y_train, X_val, y_val, _, _, scaler, train_dates, val_dates, _ = prepare_data_for_model()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create model
    print("Building model...")
    model = build_model()
    
    # Create save directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/saved")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create evaluation set
    eval_set = [(X_train, y_train), (X_val, y_val)]
    eval_names = ['train', 'valid']
    
    # For tracking losses
    train_losses = []
    val_losses = []
    
    # Create callback to track losses
    def callback(env):
        iteration = env.iteration
        train_loss = env.evaluation_result_list[0][2]
        val_loss = env.evaluation_result_list[1][2]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_metric='rmse',
        # early_stopping_rounds=20,
        # verbose=True,
        callbacks=[callback]
    )
    
    # Save model
    model_path = os.path.join(model_dir, config.LIGHTGBM_MODEL_NAME)
    save_model(model, model_path)
    
    # Make predictions on validation set
    val_predictions = model.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Plot training vs validation loss
    plot_training_vs_validation_loss(
        train_losses,
        val_losses,
        title='LightGBM Training vs Validation Loss',
        save_path=os.path.join(results_dir, 'training_vs_validation_loss.png')
    )
    
    # Plot feature importance
    plot_feature_importance(
        model,
        save_path=os.path.join(results_dir, 'feature_importance.png')
    )
    
    # Plot validation predictions
    val_dates_array = np.array(val_dates)[config.SEQUENCE_LENGTH:]
    plot_predictions(
        y_val,
        val_predictions,
        dates=val_dates_array,
        scaler=scaler,
        title='LightGBM Validation Predictions',
        save_path=os.path.join(results_dir, 'validation_predictions.png')
    )
    
    print(f"Model saved to {model_path}")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    train()