import os
import pandas as pd
from logging_custom.logger import Logger

class DataFrameManager:
    def __init__(self, file_path: str):
        self.logger = Logger("DataFrameManager").get_logger()
        self.file_path = file_path

    def load_dataframe(self) -> pd.DataFrame:
        """Load a DataFrame from the file path."""
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            self.logger.info(f"DataFrame loaded from {self.file_path}")
        else:
            df = pd.DataFrame()
            self.logger.warning(f"File {self.file_path} does not exist. Initialized empty DataFrame.")
        return df

    def save_dataframe(self, df: pd.DataFrame, file_path: str):
        """Save a DataFrame to the specified file path."""
        df.to_csv(file_path, index=False)
        self.logger.info(f"DataFrame saved to {file_path}")

    def delete_dataframe(self, df: pd.DataFrame):
        """Delete a DataFrame from memory."""
        del df
        self.logger.info("DataFrame deleted from memory")