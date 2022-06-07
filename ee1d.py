import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print("tensorflow version", tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

## Define image properties:
imgDir = "..\ps7\data"
targetWidth, targetHeight = 500, 200
imageSize = (targetWidth, targetHeight)
channels = 1  # color channels

## define other constants, including command line argument defaults
epochs = 10
plot = True  # show plots?

# sanity checks
import __main__ as main
if hasattr(main, "__file__"):
    # run as file
    print("parsing command line arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
                        help = "directory to read images from",
                        default = imgDir)
    parser.add_argument("--epochs", "-e",
                        help = "how many epochs",
                        default= epochs)
    parser.add_argument("--plot", "-p",
                        action = "store_true",
                        help = "plot a few wrong/correct results")
    args = parser.parse_args()
    imgDir = args.dir
    epochs = int(args.epochs)
    plot = args.plot
else:
    # run as notebook
    print("run interactively from", os.getcwd())
    imageDir = os.path.join(os.path.expanduser("~"),
                            "data", "images", "text", "language-text-images")
print("Load images from", imgDir)
print("epochs:", epochs)

# Prepare training data
filenames = os.listdir(os.path.join(imgDir, "train"))
print(len(filenames), "images found")
trainingResults = pd.DataFrame({
    'filename':filenames,
    'category':pd.Series(filenames).str[-10:-8]
})

### Separation of languages for training
language_1 = 'EN'
language_2 = 'ZN'
trainingResultsFirst = trainingResults[(trainingResults['category'] == language_1)]
trainingResultsSecond = trainingResults[(trainingResults['category'] == language_2)]
trainingResultsFirst = trainingResultsFirst.iloc[:1000]
trainingResultsSecond = trainingResultsSecond.iloc[:1000]
trainingResultsSep = pd.concat([trainingResultsFirst,trainingResultsSecond])
###

print(trainingResultsSep.shape)