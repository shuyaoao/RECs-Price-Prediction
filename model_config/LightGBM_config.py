MODEL_ID = 'LightGBM'

RANDOM_SEED = 42

SEQUENCE_LENGTH = 10

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 200
}

# LightGBM model name
SAVED_WEIGHTS = "lightgbm_model.pkl"