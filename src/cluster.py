import pandas as pd
from sklearn.cluster import KMeans
from .config import EMB_DIR

def cluster_products(n_clusters=50):

    df = pd.read_csv(f"{EMB_DIR}/product_embeddings.csv")

    ids = df["product_asset_id"]
    X = df.drop(columns=["product_asset_id"]).values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    df_out = pd.DataFrame({
        "product_asset_id": ids,
        "cluster_id": labels
    })

    df_out.to_csv(f"{EMB_DIR}/product_clusters.csv", index=False)

    print("Clusters saved.")