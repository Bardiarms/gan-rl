# 🎨 GAN–RL for Cartoon Face Generation

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![GAN](https://img.shields.io/badge/Model-GAN-orange.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning-green.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

This repository contains the implementation of a **Generative Adversarial Network (GAN)** combined with **Reinforcement Learning (RL)** for generating diverse and high-quality cartoon face images.  
The project was developed as part of **Homework 3** and follows a complete pipeline including GAN training, quantitative evaluation, and RL-guided latent optimization.

---

## 📌 Project Overview

The project consists of two main stages:

### 1️⃣ GAN Training  
A Deep Convolutional GAN (DCGAN-style) is trained on the **CartoonSet** dataset to generate cartoon face images.

### 2️⃣ Reinforcement Learning for Latent Selection  
After GAN training, the generator is frozen and treated as an environment.  
An RL agent learns to **select latent vectors** that maximize a reward derived from the discriminator, leading to improved image quality and diversity.

---

## 🧠 Key Ideas

- GAN learns the data distribution of cartoon faces
- Discriminator acts as a perceptual quality signal
- RL agent learns *where to sample* in latent space
- Quantitative metrics (FID, LPIPS, entropy) validate improvements
- RL-guided samples outperform purely random sampling

---
