import cv2 #opencv-python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import decode_predictions, ResNet50
import h5py
import datetime
import os
import shutil
import pathlib
import time
from IPython.display import display, Image, Markdown
import pickle
# import pydot
# import graphviz


def md(input):
    display(Markdown(input))

def step(input):
    return md(f"✅ *{input}*")


def save_model(model, model_path):
  """
  Saves a given model in a models directory and appends a suffix (str)
  for clarity and reuse.
  """
  # Create model directory with current time
#   modeldir = os.path.join("models",
#                           datetime.datetime.now().strftime("%d-%m-%y-%Hh%Mm"))
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  shutil.copy(model_path,'models/last_model.h5')

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


def plot_dog(row):
    breeds = list(row.proba.keys())
    scores = list(row.proba.values())
    img = mpimg.imread(row.path)
    plt.figure(figsize=(20,5))
    Grid_plot = plt.GridSpec(1, 2, wspace = 0.15)
    plt.subplot(Grid_plot[0, 0])
    imgplot = plt.imshow(img);
    plt.subplot(Grid_plot[0, 1:])
    clrs = ['green' if x == row.breed else 'grey' for x in breeds]
    sns.barplot(y=breeds, x=scores, orient='h', palette=clrs)
    
    
def plot_history(history):
    metrics = (('accuracy', 'val_accuracy'), ('loss', 'val_loss'))
    for metric in metrics:
        plt.plot(history[metric[0]])
        plt.plot(history[metric[1]])
        plt.title('model {}'.format(metric[0]))
        plt.ylabel(metric[0])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
# def draw_model(model, model_name):
#     plot_model(model,
#                show_shapes=True,
#                show_layer_names=True,
#                to_file='models/{}.jpg'.format(model_name))        