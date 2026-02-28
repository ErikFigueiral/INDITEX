import argparse
from src.dataset_loader import load_dataset
from src.embedder import generate_embeddings
from src.cluster import cluster_products

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--FULL", action="store_true")
    parser.add_argument("--10", action="store_true")
    parser.add_argument("--cluster", action="store_true")

    args = parser.parse_args()

    df = load_dataset(args.dataset)

    if args.FULL:
        limit = None
    elif args.__dict__["10"]:
        limit = 10
    else:
        limit = None

    generate_embeddings(
        df,
        id_col="product_asset_id",
        url_col="product_image_url",
        limit=limit
    )

    if args.cluster:
        cluster_products()

if __name__ == "__main__":
    main()