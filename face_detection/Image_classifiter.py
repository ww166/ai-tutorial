import imp
import ssl

from tensorflow.python.keras.engine import input_spec

ssl._create_default_https_context = ssl._create_unverified_context

# Code start

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

import matplotlib.pyplot as plt

# Load a pre-defined dataset
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data

print(train_labels)

plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

  
# Deffine our neural net structure

model = keras.Sequential([

    
    keras.layers.Flatten(input_shape=(28, 28)),
  ])