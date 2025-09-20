import os
from dotenv import load_dotenv
from logging_custom.logger import Logger
import xgboost as xgb


load_dotenv()
MODEL_PATH = os.getenv("XGBOOST_MODEL_PATH")
SCALER_PATH = os.getenv("STD_SCALER_PATH")

class XGBoostRecommender:
    def __init__(self):
        self.logger = Logger("XGBoostRecommender").get_logger()
        self.model = self.load_model(MODEL_PATH)
        self.scaler = self.load_scaler(SCALER_PATH)
        self.logger.info("XGBoost Recommender initialized.")
    def load_model(self, model_path):
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading XGBoost model from {model_path}")
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        else:
            self.logger.error(f"Model path {model_path} does not exist.")
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
    def load_scaler(self, scaler_path):
        if scaler_path and os.path.exists(scaler_path):
            self.logger.info(f"Loading scaler from {scaler_path}")
            import joblib
            scaler = joblib.load(scaler_path)
            return scaler
        else:
            self.logger.error(f"Scaler path {scaler_path} does not exist.")
            raise FileNotFoundError(f"Scaler path {scaler_path} does not exist.")
    
    def preprocess(self, X):
        self.logger.info("Preprocessing input features.")
        return self.scaler.transform(X)
    
    def predict(self, X, movie_ids, seen_movies):
        self.logger.info("Making predictions.")
        preds = self.model.predict(X)
        self.logger.info(f"Prediction complete. Sample view: {preds[:5]}. Shape: {preds.shape}")
        preds = sorted(zip(movie_ids, preds), key=lambda x: x[1], reverse=True)
        # exclude seen movies
        preds = [(movie_id, pred) for movie_id, pred in preds if movie_id not in seen_movies]

        return preds