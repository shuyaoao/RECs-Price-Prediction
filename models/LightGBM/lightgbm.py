"""LightGBM model for time series forecasting."""
import numpy as np
import lightgbm as lgb
import os
import pickle
from models.dataloader import TreeBased_DataLoader
import pandas as pd
import random
from math import sqrt
from utils import plot_training_vs_validation_loss, plot_feature_importance, plot_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LightGBM():
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.dataloader = TreeBased_DataLoader(self.data_config, self.model_config)
        self.model = lgb.LGBMRegressor(**self.model_config.LIGHTGBM_PARAMS)


    def set_seed(self):
        """Set seed for reproducibility."""
        seed = self.model_config.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)

    def train(self):
        """Train the LightGBM model."""
        # Set seed for reproducibility
        self.set_seed()
        
        print("Preparing data...")
        X_train, y_train, X_val, y_val, _, _, scaler, train_dates, val_dates, _ = self.dataloader.prepare_data_for_model()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Create model
        print("Building model...")
        model = self.model
        
        # Create save directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
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
        model_path = os.path.join(model_dir, self.model_config.SAVED_WEIGHTS)
        self.save_model(model_path)
        
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
        val_dates_array = np.array(val_dates)[self.model_config.SEQUENCE_LENGTH:]
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

    def save_model(self, path):
        """
        Save LightGBM model to disk using pickle.
        
        Args:
            model: Trained LightGBM model
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model in pickle format
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved in pickle format to {path}")

    def load_model(self, path):
        """
        Load LightGBM model from disk using pickle.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded LightGBM model
        """
        print(f"Loading model from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load model from pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model

    def evaluate(self):
        """Evaluate the LightGBM model on test data."""
        print("Preparing data...")
        _, _, _, _, X_test, y_test, scaler, _, _, test_dates = self.dataloader.prepare_data_for_model()
        
        print(f"Test data shape: {X_test.shape}")
        
        # Load model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID, self.model_config.SAVED_WEIGHTS)

        try:
            print(f"Loading model from: {model_path}")
            model = self.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Make predictions
        print("Making predictions...")
        test_predictions = model.predict(X_test)
        print("Predictions made")
        # Inverse transform to get original scale
        test_predictions_rescaled = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
        
        # Calculate metrics
        mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
        r2 = r2_score(y_test_rescaled, test_predictions_rescaled)
        
        # Calculate MAPE
        # Avoid division by zero
        mask = y_test_rescaled != 0
        mape = 100 * np.mean(np.abs((y_test_rescaled[mask] - test_predictions_rescaled[mask]) / y_test_rescaled[mask]))
        
        # Print metrics
        print("\nTest Set Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot predictions
        test_dates_array = np.array(test_dates)[self.model_config.SEQUENCE_LENGTH:]
        
        plot_predictions(
            y_test,
            test_predictions,
            dates=test_dates_array,
            scaler=scaler,
            title='LightGBM Test Predictions',
            save_path=os.path.join(results_dir, 'test_predictions.png')
        )
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²'],
            'Value': [mse, rmse, mae, mape, r2]
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
        
        return mse, rmse, mae, mape, r2, predictions_df
