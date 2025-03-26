import os
import importlib

class Evaluator:
    def __init__(self, model_id, data_id):
        self.model_id = model_id
        self.data_id = data_id
        self.data_config, self.model_config = self._load_config()
        self.model_class = self._load_model()
        self.model_instance = self.model_class(self.data_config, self.model_config)

    def _load_config(self):
        data_config_path = os.path.join("data_config", f"{self.data_id}_config.py")
        model_config_path = os.path.join("model_config", f"{self.model_id}_config.py")
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(f"Data Configuration file not found: {data_config_path}")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model Configuration file not found: {model_config_path}")
        
        data_spec = importlib.util.spec_from_file_location("config", data_config_path)
        data_config = importlib.util.module_from_spec(data_spec)
        data_spec.loader.exec_module(data_config)

        model_spec = importlib.util.spec_from_file_location("config", model_config_path)
        model_config = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_config)
        return data_config, model_config
    
    def _load_model(self):
        if self.model_id == "ARIMA":
            from models.ARIMA.model import TimeSeriesARIMA
            return TimeSeriesARIMA
        elif self.model_id == "CNN_BiLSTM_AM":
            from models.CNN_BiLSTM_AM.cnn_bilstm_am import CNNBiLSTMAM
            return CNNBiLSTMAM
        elif self.model_id == "XGBoost":
            from models.XGBoost.xgboost import XGBoost
            return XGBoost
        elif self.model_id == "LightGBM":
            from models.LightGBM.lightgbm import LightGBM
            return LightGBM
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")

    def evaluate(self):
        self.model_instance.evaluate()
        # mse = mean_squared_error(test_data, forecast_series)
        # rmse = np.sqrt(mse)
        # mae = mean_absolute_error(test_data, forecast_series)
        # mape = np.mean(np.abs((test_data - forecast_series) / test_data)) * 100

        # print("\n📊 Evaluation Metrics:")
        # print(f"- MSE:  {mse:.4f}")
        # print(f"- RMSE: {rmse:.4f}")
        # print(f"- MAE:  {mae:.4f}")
        # print(f"- MAPE: {mape:.2f}%")

if __name__ == "__main__":
    for model in ["CNN_BiLSTM_AM"]:
        evaluator = Evaluator(model, "thailand_biomass")
        evaluator.evaluate()