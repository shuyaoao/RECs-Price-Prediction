"""Training script for XGBoost model."""
import os
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from india_wind_solar import config
from india_wind_solar.XGBoost.data.dataloader import prepare_data_for_model
from india_wind_solar.XGBoost.models.xgboost import build_model, save_model
from india_wind_solar.XGBoost.utils.visualisation import plot_training_history, plot_predictions

def set_seed(seed=config.RANDOM_SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def train():
    """Train the XGBoost model."""
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
    
    # Create evaluation set for early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        # eval_metric='rmse',
        # early_stopping_rounds=20,
        verbose=True
    )
    
    # Save model
    model_path = os.path.join(model_dir, config.XGBOOST_MODEL_NAME)
    save_model(model, model_path)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    print(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Make predictions on validation set
    val_predictions = model.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Plot training history
    plot_training_history(
        model,
        X_val,
        y_val,
        save_path=os.path.join(results_dir, 'training_history.png')
    )
    
    # Plot validation predictions
    val_dates_array = np.array(val_dates)[config.SEQUENCE_LENGTH:]
    plot_predictions(
        y_val,
        val_predictions,
        dates=val_dates_array,
        scaler=scaler,
        title='XGBoost Validation Predictions',
        save_path=os.path.join(results_dir, 'validation_predictions.png')
    )
    
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train()