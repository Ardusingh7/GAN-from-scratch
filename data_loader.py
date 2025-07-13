import tensorflow as tf
from config import IMAGE_SIZE, BATCH_SIZE, BUFFER_SIZE

def load_dataset():
    (train_images, _), _ = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)