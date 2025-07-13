from data_loader import load_dataset
from model import build_generator, build_discriminator
from train import train
import tensorflow as tf
from config import EPOCHS

def main():
    dataset = load_dataset()
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    train(generator, discriminator, dataset, EPOCHS, generator_optimizer, discriminator_optimizer)

if __name__ == "__main__":
    main()