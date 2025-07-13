# GAN from Scratch 🧠✨

This repository contains a from-scratch implementation of a **Generative Adversarial Network (GAN)** using **PyTorch**. The GAN learns to generate synthetic data by training two neural networks — a generator and a discriminator — in a competitive setting.

## 🔍 Overview

This project demonstrates the fundamental working of GANs:
- A **Generator** learns to produce data similar to the training distribution.
- A **Discriminator** learns to distinguish real data from generated (fake) data.
- Both models are trained in tandem using adversarial loss.

The code is written for clarity and educational purposes, making it a great starting point for anyone learning about GANs.

## 📁 Project Structure

GAN-from-scratch/
├── gan_train.py # Main training script
├── models.py # Generator and Discriminator model definitions
├── utils.py # Utility functions (plotting, saving outputs)
├── outputs/ # Saved generated samples

Install dependencies:

pip install -r requirements.txt
