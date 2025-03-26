"""XGBoost model for time series forecasting."""
import numpy as np
import xgboost as xgb
import pickle
import os
import random
from models.dataloader import TreeBased_DataLoader
from utils import plot_training_history, plot_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from math import sqrt

class XGBoost():
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.model = xgb.XGBRegressor(**self.model_config.XGBOOST_PARAMS)
        self.dataloader = TreeBased_DataLoader(self.data_config, self.model_config)

    def set_seed(self):
        """Set seed for reproducibility."""
        seed = self.model_config.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)

    def train(self):
        """Train the XGBoost model."""
        # Set seed for reproducibility
        self.set_seed()
        
        print("Preparing data...")
        X_train, y_train, X_val, y_val, _, _, scaler, train_dates, val_dates, _ = self.dataloader.prepare_data_for_model()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Create save directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create evaluation set for early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        print("Training model...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            # eval_metric='rmse',
            # early_stopping_rounds=20,
            verbose=True
        )
        
        # Save model
        model_path = os.path.join(model_dir, self.model_config.SAVED_WEIGHTS)
        self.save_model(model_path)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
        os.makedirs(results_dir, exist_ok=True)
        
        # Make predictions on validation set
        val_predictions = self.model.predict(X_val)
        val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
        print(f"Validation RMSE: {val_rmse:.6f}")
        
        # # Plot training history
        # plot_training_history(
        #     self.model,
        #     X_val,
        #     y_val,
        #     save_path=os.path.join(results_dir, 'training_history.png')
        # )
        
        # Plot validation predictions
        val_dates_array = np.array(val_dates)[self.model_config.SEQUENCE_LENGTH:]
        plot_predictions(
            y_val,
            val_predictions,
            dates=val_dates_array,
            scaler=scaler,
            title='XGBoost Validation Predictions',
            save_path=os.path.join(results_dir, 'validation_predictions.png')
        )
        
        print(f"Model saved to {model_path}")
        
        return self.model

    def save_model(self, path):
        """
        Save XGBoost model to disk.
        
        Args:
            model: Trained XGBoost model
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def evaluate(self):
        """Evaluate the XGBoost model on test data."""
        print("Preparing data...")
        _, _, _, _, X_test, y_test, scaler, _, _, test_dates = self.dataloader.prepare_data_for_model()
        
        print(f"Test data shape: {X_test.shape}")
        
        # Load model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID, self.model_config.SAVED_WEIGHTS)
        model = self.load_model(model_path)
        print("Loaded Model")
        
        # Make predictions
        test_predictions = model.predict(X_test)
        
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
            title='XGBoost Test Predictions',
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

    def load_model(self, path):
        """
        Load XGBoost model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded XGBoost model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model