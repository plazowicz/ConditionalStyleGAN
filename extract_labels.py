import tensorflow as tf
import numpy as np
import os
import pickle
import argparse

from sklearn.cluster import KMeans
from horology import timed


def mini_batches_generator(dir_path, batch_size=32):
    images_names = os.listdir(dir_path)
    for i in range(0, len(images_names), batch_size):
        batch_names = images_names[i:i + batch_size]
        images = []
        for img_name in batch_names:
            img_path = os.path.join(dir_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
        x = np.array(images, dtype=np.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        yield x, batch_names


@timed
def compute_embeddings(dir_path, embeddings_path):
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    generator = mini_batches_generator(dir_path)

    image_embeddings_dict = {}

    for x, batch_names in generator:
        features = model.predict(x)
        for i, img_name in enumerate(batch_names):
            image_embeddings_dict[img_name] = features[i]

    with open(embeddings_path, 'wb') as f:
        pickle.dump(image_embeddings_dict, f, pickle.HIGHEST_PROTOCOL)

    return image_embeddings_dict


@timed
def cluster_embeddings(embeddings):
    kmeans = KMeans(n_clusters=10, verbose=1, n_jobs=-1)
    kmeans.fit(embeddings)
    return kmeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-path', required=True, type=str)
    parser.add_argument('--embeddings-path', required=True, type=str)
    parser.add_argument("--labels-path", required=True, type=str)
    args = parser.parse_args()

    if os.path.exists(args.embeddings_path):
        with open(args.embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
    else:
        embeddings_dict = compute_embeddings(args.dir_path, args.embeddings_path)

    file_names = sorted(embeddings_dict.keys())

    embeddings = []

    for fn in file_names:
        embeddings.append(embeddings_dict[fn])

    embeddings = np.array(embeddings)

    kmeans = cluster_embeddings(embeddings)

    labels_dict = {
        "Filenames": file_names,
        "Labels": kmeans.labels_
    }

    with open(args.labels_path, 'wb') as f:
        pickle.dump(labels_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
