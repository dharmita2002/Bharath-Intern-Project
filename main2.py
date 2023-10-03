import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
if not os.path.exists("mnist_images"):
    os.makedirs("mnist_images")
for i in range(10):
    plt.figure(figsize=(2,2))
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
    # Save the image in the directory as a PNG file
    plt.savefig(f"mnist_images/image_{i}.png")
print("Images saved!")