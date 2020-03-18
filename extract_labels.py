import tensorflow as tf
import numpy as np
import os
import pickle
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-path', required=True, type=str)
    parser.add_argument('--embeddings-path', required=True, type=str)
    args = parser.parse_args()

    compute_embeddings(args.dir_path, args.embeddings_path)


if __name__ == '__main__':
    main()
