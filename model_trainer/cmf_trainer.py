import os
import pickle
from dotenv import load_dotenv
from itertools import product
from cmfrec import CMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error as mse
from logging_custom.logger import Logger
from utils.ratings_helper import RatingsHelper

load_dotenv()
CMF_MODEL_PATH = os.getenv('CMF_MODEL_PATH')
CMF_USER_KNN = os.getenv('CMF_USER_KNN')
CMF_ITEM_KNN = os.getenv('CMF_ITEM_KNN')

class CMFTrainer:
    def __init__(self):
        self.logger = Logger("CMFTrainer").get_logger()
        ratings_helper = RatingsHelper()
        self.ratings_df = ratings_helper.ratings_df
        self.ratings_df = self.ratings_df.drop(columns=['timestamp'])
        self.ratings_df.columns = ['UserId', 'ItemId', 'Rating']
        self.logger.info(f"Initialized ratings data: {self.ratings_df.shape}")
        self.params_dist = {
            'k': [5, 10, 25, 40, 60],
            'lambda_': [0.001, 0.01, 0.1, 1, 10]
        }
        self.logger.info(f"Parameters to test: {self.params_dist}")
        self.best_param = None
        self.best_score = float('inf')
        self.best_model = None


    def _save_kneighbors(self, cmf_model):
        self.logger.info(f"Saving NNeighbors model for user and item..")
        n_neighbors = 20
        metric = 'cosine'
        algorithm = 'brute'
        self.logger.info(f"Assigned NNeighbors stats: n_neighbors: {n_neighbors}, metric: {metric}, algorithm: {algorithm}")

        nn_user = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        nn_movie = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        self.logger.info(f"Initialized Nearest Neighbors")

        self.logger.info(f"Fitting NN models")
        nn_user.fit(cmf_model.A_)
        nn_movie.fit(cmf_model.B_)
        self.logger.info(f"Saving NN models")
        with open("models/cmfrec_model/cmf_user_kneighbors.pkl", 'wb') as f:
            pickle.dump(nn_user, f)
            self.logger.info(f"Saving user NN models")
        with open("models/cmfrec_model/cmf_item_kneighbors.pkl", 'wb') as f:
            pickle.dump(nn_movie, f)
            self.logger.info(f"Saving movie NN models")

    def search_best_param(self):
        self.logger.info("Searching for best parameters.")
        param_names = list(self.params_dist.keys())
        param_values = list(self.params_dist.values())

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            self.logger.info(f"Testing parameters: {params}")
            model = CMF(k=params['k'], lambda_=params['lambda_'])
            model.fit(self.ratings_df)
            preds = model.predict(self.ratings_df['UserId'].values, self.ratings_df['ItemId'].values)
            score = mse(self.ratings_df['Rating'].values, preds)
            self.logger.info(f"MSE for parameters {params}: {score}")
            if score < self.best_score:
                self.best_score = score
                self.best_param = params
                self.best_model = model
        self.logger.info(f"****Best parameters: {self.best_param} with MSE: {self.best_score}****")
        if self.best_model is not None:
            self.logger.info("Saving best model configuration in output path")
            os.makedirs(os.path.dirname(CMF_MODEL_PATH), exist_ok=True)
            with open("models/cmfrec_model/cmf_full.pkl", 'wb') as file:
                pickle.dump(self.best_model, file)
            self.logger.info(f"Best model saved at {self.best_model}. Saving NN models for user and item simmilarities")
            self._save_kneighbors(self.best_model)
            return True
        


