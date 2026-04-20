## Anime_Face_Generation_DCGAN

## Overview
This repository contains a PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed to generate realistic anime faces. The model is trained on the Anime Face Dataset from Kaggle.

## Instructions for Running the Code
1. **Prerequisites**: Ensure you have Python installed along with PyTorch, Torchvision, Matplotlib, Numpy. 
   ```bash
   pip install torch torchvision matplotlib Numpy

2. Dataset Setup: Download the Anime Face Dataset from Kaggle "https://www.kaggle.com/splcher/animefacedataset"
   a) You can use Kaggle API Key for fast and quick downloading of dataset.
   OR
   b) Just do Manual download, Extract the zip file and start working with it.
   Recommendition:- If you are using Colab then it is better to use Kaggle API Key and Put the path of Dataset here dataroot = "PASTE_YOUR_DATASET_PATH".

3. Execution: Run the Python script from your terminal:
   ```bash
   python gan_anime_faces.py

# Detailed Project Report

1. **Approach**:- The objective of this project was to implement a Generative Adversarial Network (GAN) capable of synthesizing realistic anime faces. I selected a Deep Convolutional GAN (DCGAN) architecture due to its proven stability and performance in image generation tasks.

  i) **Data Pipeline**:- I utilized the Anime Face Dataset from Kaggle. To optimize the data ingestion pipeline within a cloud notebook environment, I bypassed manual, error-prone file uploads and integrated the Kaggle API for direct server-to-server transfer. The images were preprocessed using PyTorch’s torchvision.transforms, scaled to a uniform 64x64 resolution, center-cropped, and normalized to a [-1, 1] range to align with the Generator's Tanh activation function.

  ii) **Model Architecture**:- The project utilizes a DCGAN architecture:

  a) **Generator**: Maps a 100-dimensional latent vector (z) to a 64x64x3 RGB image. It utilizes a series of ConvTranspose2d layers paired with BatchNorm2d and ReLU activations. The final layer uses a Tanh activation to output pixel values between -1 and 1.

  b) **Discriminator**: A binary classification network that takes a 64x64x3 image and outputs a scalar probability of it being real. It consists of Conv2d layers, BatchNorm2d, and LeakyReLU activations, culminating in a Sigmoid activation function.

  iii) **Training Process**:-
  
  a) **Data Preprocessing**: Images are resized to 64x64, center-cropped, and normalized to the range [-1, 1].

  b) **Loss Function**: Binary Cross Entropy Loss (BCELoss) is used for both the generator and discriminator.

  c) **Optimizers**: Adam optimizers are used for both networks with a learning rate of 0.0002 and a Beta1 hyperparameter of 0.5 to ensure training stability.

  d) **Loop**: The discriminator is trained to maximize the probability of correctly classifying real vs. fake images. The generator is trained simultaneously to maximize the   probability that the discriminator mistakes its generated images for real ones.

2) **Challenges Faced**:- During the implementation and training phases, several technical challenges required active debugging:

  a) Data Transfer and Zip Corruption: Initial attempts to manually upload the large, compressed dataset into the cloud computing environment resulted in interrupted transfers and "zip bomb" errors due to missing bytes. This was resolved by discarding the manual upload approach and implementing a script to authenticate and pull the dataset directly via the Kaggle API.

  b) Hardware and Pipeline Bottlenecks: The initial training loops were bottlenecked by the CPU, resulting in highly inefficient epoch times. After allocating a T4 GPU, a secondary issue arose where PyTorch's DataLoader froze due to multiprocessing conflicts across multiple CPU cores (num_workers=2). This was successfully debugged and resolved by forcing the data loader to operate on the main thread (workers=0), which restored pipeline stability and allowed the GPU to achieve maximum throughput.

3) **Results and Evaluation**:- The DCGAN was successfully trained, and the evaluation indicates a well-balanced adversarial relationship.

  a) **Training Dynamics**: The plotted loss curves demonstrate a stable equilibrium between the Generator and Discriminator. The Discriminator maintained a low, consistent loss, while the Generator avoided catastrophic gradient collapse, indicating that neither network overpowered the other prematurely.

  b) **Visual Fidelity and Diversity**: The model successfully mapped the latent space to the dataset's distribution. The generated 64x64 images exhibit clear, recognizable facial geometries, including distinct eyes, hair framing, and chins. Furthermore, the outputs show significant diversity in hair color and eye style, confirming that the model successfully avoided mode collapse.

Note:- Due to computational limitations and strict time constraints, computationally heavy quantitative metrics such as Inception Score (IS) and Frechet Inception Distance (FID) were not implemented. Instead, the model was evaluated qualitatively and through training dynamics(Visual Fidelity and Diversity).

