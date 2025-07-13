import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from config import LATENT_DIM

def generate_and_save_images(model, epoch, test_input=None):
    if test_input is None:
        test_input = tf.random.normal([16, LATENT_DIM])

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2.0, cmap='gray')
        plt.axis('off')

    if not os.path.exists('images'):
        os.makedirs('images')

    plt.savefig(f'images/image_at_epoch_{epoch:04d}.png')
    plt.close()