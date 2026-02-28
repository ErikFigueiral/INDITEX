import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.image_processing import download_image, preprocess_image

EMB_PATH = "out/embeddings/product_embeddings.csv"
IMG_SIZE = 224


def load_embeddings():
    df = pd.read_csv(EMB_PATH)
    X = df["embedding"].apply(json.loads).tolist()
    X = np.array(X)
    ids = df["product_asset_id"].values
    return ids, X


def cluster_embeddings(X, n_clusters=20):
    print(f"Clustering with k={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    return labels


def visualize_cluster_samples(ids, labels, cluster_id, n_show=10):
    idx = np.where(labels == cluster_id)[0]

    if len(idx) == 0:
        print("Cluster vacío")
        return

    selected = idx[:n_show]

    plt.figure(figsize=(15, 3))

    for i, index in enumerate(selected):
        try:
            product_id = ids[index]
            # reconstruir url si la necesitas desde CSV original
            # aquí solo mostramos ID
            plt.subplot(1, n_show, i + 1)
            plt.text(0.5, 0.5, product_id, ha="center")
            plt.axis("off")
        except:
            continue

    plt.suptitle(f"Cluster {cluster_id}")
    plt.show()


def visualize_cluster_images(ids, labels, df_products, cluster_id, n_show=10):
    idx = np.where(labels == cluster_id)[0]

    if len(idx) == 0:
        print("Cluster vacío")
        return

    selected = idx[:n_show]

    plt.figure(figsize=(15, 3))

    for i, index in enumerate(selected):
        product_id = ids[index]

        row = df_products[df_products["product_asset_id"] == product_id]
        if row.empty:
            continue

        url = row.iloc[0]["product_image_url"]

        try:
            img = download_image(url)
            img = preprocess_image(img, IMG_SIZE)

            plt.subplot(1, n_show, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

        except:
            continue

    plt.suptitle(f"Cluster {cluster_id}")
    plt.show()


def main():

    # Cargar embeddings
    ids, X = load_embeddings()

    # Cambia aquí número de clusters
    labels = cluster_embeddings(X, n_clusters=20)

    # Guardar resultado
    df_out = pd.DataFrame({
        "product_asset_id": ids,
        "cluster_id": labels
    })
    df_out.to_csv("out/embeddings/product_clusters.csv", index=False)
    print("Clusters guardados")

    # PCA visual
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=5)
    plt.title("PCA colored by cluster")
    plt.show()

    # Cargar dataset original para recuperar URLs
    df_products = pd.read_csv("data/product_dataset.csv")

    # Mostrar 3 clusters aleatorios
    for cluster_id in range(3):
        visualize_cluster_images(ids, labels, df_products, cluster_id, n_show=10)


if __name__ == "__main__":
    main()