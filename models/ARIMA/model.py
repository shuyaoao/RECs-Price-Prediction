import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import os
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import import_config

class TimeSeriesARIMA:
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

    def load_data(self):
        """Load the time series dataset and apply train-test split."""
        try:
            df = pd.read_excel(self.data_config.DATASET_PATH, engine="openpyxl")
            df['Published Date(in GMT)'] = pd.to_datetime(df['Published Date(in GMT)'])
            df = df[df["Market Activity"].str.lower() == self.data_config.MARKET_ACTIVITY_TYPE.lower()]
            df = df.sort_values('Published Date(in GMT)').set_index('Published Date(in GMT)')

            price_series = df['Price (USD/MWh)'].resample('D').mean().ffill()

            train_data = price_series[self.data_config.TRAIN_START:self.data_config.VALIDATION_END]
            test_data = price_series[self.data_config.TEST_START:self.data_config.TEST_END]

            print(f"Data shapes - Train: {train_data.shape}, Test: {test_data.shape}")

            return train_data, test_data

        except Exception as e:
            raise Exception(f"Error in load_data: {str(e)}")

    def check_stationarity(self, series):
        """Perform the Augmented Dickey-Fuller (ADF) test for stationarity."""
        result = adfuller(series)
        return result[1]  # p-value

    def make_stationary(self, series):
        """Apply differencing if data is not stationary."""
        p_value = self.check_stationarity(series)
        if p_value >= 0.05:
            series = series.diff().dropna()
            print("Data was not stationary, differencing applied.")
        else:
            print("Data is already stationary.")
        return series

    def select_arima_order(self, series):
        """Use Auto-ARIMA to determine the best (p, d, q) order, enforcing differencing."""
        model = auto_arima(series, d=1, seasonal=False, trace=True, stepwise=True)
        print(f"Selected ARIMA order: {model.order}")
        return model.order

    def train_arima(self, series, order):
        """Train ARIMA model using the best (p, d, q) order."""
        model = ARIMA(series, order=order)
        result = model.fit()
        print("ARIMA model training complete.")
        return result
    
    def train(self):
        train_data, _ = self.load_data()
        print(train_data.index)

        # Ensure stationarity
        print("Ensuring Stationarity")
        train_series = self.make_stationary(train_data)
        print("Ensured Stationarity")

        # Select best (p, d, q) order
        print("Selecting best order")
        best_order = self.select_arima_order(train_series)
        print(f"Best ARIMA Order: {best_order}")

        # Train ARIMA on training data
        arima_model = ARIMA(train_series, order=best_order)
        model_fit = arima_model.fit()

        # Rolling Forecast Validation
        history = list(train_series)
        predictions = []

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_pkl_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID, self.model_config.SAVED_WEIGHTS)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_pkl_path), exist_ok=True)

        # Save the model
        with open(model_pkl_path, "wb") as model_file:
            pickle.dump(model_fit, model_file)

        print(f"✅ ARIMA model saved as: {model_pkl_path}")

    def evaluate(self):
        _, test_data = self.load_data()
        print("loaded data")

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.MODEL_ID, self.model_config.SAVED_WEIGHTS)

        arima_result = ARIMAResults.load(model_path)
        forecast_steps = len(test_data)
        test_forecast = arima_result.forecast(steps=forecast_steps)

        test_forecast = pd.Series(test_forecast, index=test_data.index)

        results_dir = os.path.join(base_dir, self.data_config.PLOTS_DIR, self.model_config.MODEL_ID)
        os.makedirs(results_dir, exist_ok=True)

        # Calculate test error metrics
        test_mse = mean_squared_error(test_data, test_forecast)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(test_data, test_forecast)
        test_mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100

        # Print results
        print(f"\n📊 Test Metrics:")
        print(f" - MSE:  {test_mse:.4f}")
        print(f" - RMSE: {test_rmse:.4f}")
        print(f" - MAE:  {test_mae:.4f}")
        print(f" - MAPE: {test_mape:.2f}%")

        metrics = {
            "MSE": test_mse,
            "RMSE": test_rmse,
            "MAE": test_mae,
            "MAPE": test_mape
        }

        results_df = pd.DataFrame([metrics])
        results_path = os.path.join(results_dir, "test_metrics.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Metrics saved to: {results_path}")

        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label="Test Data", color="blue", marker='o')
        plt.plot(test_forecast.index, test_forecast.values, label="Test Forecast", 
                color="red", linestyle="dashed", marker='x')
        plt.xlabel("Date")
        plt.ylabel("Price (USD/MWh)")
        plt.title("ARIMA Model: Test Set Predictions vs Actual Values")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, "test_predictions.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"\n✅ Plot saved as: {plot_path}")

