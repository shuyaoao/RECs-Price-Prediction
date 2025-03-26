import os
import importlib
import pickle

class Trainer:
    def __init__(self, model_id, data_id):
        self.model_id = model_id
        self.data_id = data_id
        self.data_config, self.model_config = self._load_config()
        self.model_class = self._load_model_class()
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
    
    def _load_model_class(self):
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

    def train(self):
        if self.model_id == "ARIMA" or self.model_id == "XGBoost" or self.model_id == "LightGBM":
            self.model_instance.train()
        elif self.model_id == "CNN_BiLSTM_AM":
            self.model_instance.fit()
        
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(base_dir, self.data_config.MODEL_SAVE_DIR, self.model_config.SAVED_WEIGHTS)
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # with open(model_path, "wb") as f:
        #     pickle.dump(self.model_instance.model, f)
        # print(f"✅ Model saved at: {model_path}")

if __name__ == "__main__":
    for model in ["CNN_BiLSTM_AM"]:
        trainer = Trainer(model, "thailand_biomass")
        trainer.train()