import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from .config import EMB_DIR
from .image_processing import download_image, preprocess_image

IMG_SIZE = 224


def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.math.l2_normalize(x, axis=1)

    return tf.keras.Model(inp, x)


def generate_embeddings(df, id_col, url_col, limit=None, show_debug=False):

    if limit:
        df = df.head(limit)

    model = build_model()

    embeddings = []
    ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        try:
            img = download_image(row[url_col])
            img = preprocess_image(img, IMG_SIZE)

            arr = np.expand_dims(img, axis=0).astype(np.float32)
            emb = model(arr, training=False).numpy()[0]

            embeddings.append(emb)
            ids.append(row[id_col])

        except Exception as e:
            print("Error:", e)
            continue

    if len(embeddings) == 0:
        print("No embeddings generated.")
        return

    df_out = pd.DataFrame(embeddings)
    df_out.insert(0, id_col, ids)

    out_path = f"{EMB_DIR}/product_embeddings.csv"
    df_out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", df_out.shape)