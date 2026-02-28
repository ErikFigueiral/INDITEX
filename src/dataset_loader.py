import os
import pandas as pd
from .config import DATA_DIR

def load_dataset(dataset_name="product_dataset"):
    csv_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    if dataset_name == "product_dataset":
        return df[["product_asset_id", "product_image_url"]]

    raise ValueError("Unsupported dataset")