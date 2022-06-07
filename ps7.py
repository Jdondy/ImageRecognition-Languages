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
imgDir = "..\ps7\changeD"
targetWidth, targetHeight = 200, 200
imageSize = (targetWidth, targetHeight)
channels = 1  # color channels



## define other constants, including command line argument defaults
epochs = 10
plot = True  # show plots?

# sanity checks
import __main__ as main
if hasattr(main, "__file__"):
    # run as filecd 
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
language_3 = 'TH'
language_4 = 'RU'
trainingResultsFirst = trainingResults[(trainingResults['category'] == language_1)]
trainingResultsSecond = trainingResults[(trainingResults['category'] == language_2)]
trainingResultsThird = trainingResults[(trainingResults['category'] == language_3)]
trainingResultsFourth = trainingResults[(trainingResults['category'] == language_4)]
trainingResultsFirst = trainingResultsFirst.iloc[:2000]
trainingResultsSecond = trainingResultsSecond.iloc[:2000]
trainingResultsThird = trainingResultsThird.iloc[:2000]
trainingResultsFourth = trainingResultsFourth.iloc[:2000]
trainingResultsSep = pd.concat([trainingResultsFirst,trainingResultsSecond,trainingResultsThird, trainingResultsFourth])
#trainingResultsSep = trainingResults
###

print("example data files:")
print(trainingResultsSep.sample(5))
nCategories = trainingResultsSep.category.nunique()
print("categories:\n", trainingResultsSep.category.value_counts())
## Create model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM


# sequential (not recursive) model (one input, one output)
model=Sequential()

model.add(Conv2D(64,
                 kernel_size=3,
                 strides=3,
                 activation='relu',
                 kernel_initializer = tf.keras.initializers.Orthogonal(),
                 input_shape=(targetWidth, targetHeight, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.8))

model.add(Conv2D(64,
                 kernel_size=3,
                 kernel_initializer = tf.keras.initializers.Orthogonal(),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(620,  
                kernel_initializer = tf.keras.initializers.Orthogonal(),
                activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(100, activation='softmax'))

model.add(Dense(nCategories,
                kernel_initializer = tf.keras.initializers.Orthogonal(),
                activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

## Training and validation data generator:
trainingGenerator = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
).\
    flow_from_dataframe(trainingResultsSep,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=True)
label_map = trainingGenerator.class_indices
## Model Training:
history = model.fit(
    trainingGenerator,
    epochs=epochs
)

## Validation data preparation:
validationDir = os.path.join(imgDir, "validation")
fNames = os.listdir(validationDir)
print(len(fNames), "validation images")
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': pd.Series(fNames).str[-10:-8]
})

### Separation of languages for validation
validationResultsFirst = validationResults[(validationResults['category'] == language_1)]
validationResultsSecond = validationResults[(validationResults['category'] == language_2)]
validationResultsThird = validationResults[(validationResults['category'] == language_3)]
validationResultsFourth = validationResults[(validationResults['category'] == language_4)]
validationResultsFirst = validationResultsFirst.iloc[:2000]
validationResultsSecond = validationResultsSecond.iloc[:2000]
validationResultsThird = validationResultsThird.iloc[:2000]
validationResultsFourth = validationResultsThird.iloc[:2000]
validationResultsSep = pd.concat([validationResultsFirst,validationResultsSecond, validationResultsThird, validationResultsFourth])
#validationResultsSep = validationResults
###

print(validationResultsSep.shape[0], "validation files read from", validationDir)
validationGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(validationResultsSep,
                        os.path.join(imgDir, "validation"),
                        x_col='filename',
                        class_mode = None,
                        target_size = imageSize,
                        shuffle = False,
                        # do _not_ randomize the order!
                        # this would clash with the file name order!
                        color_mode="grayscale"
    )


## Make categorical prediction:
print(" --- Predicting on validation data ---")
phat = model.predict(validationGenerator)
print("Predicted probability array shape:", phat.shape)
print("Example:\n", phat[:5])

## Convert labels to categories:
validationResultsSep['predicted'] = pd.Series(np.argmax(phat, axis=-1), index=validationResultsSep.index)
print(validationResultsSep.head())
labelMap = {v: k for k, v in label_map.items()}
validationResultsSep["predicted"] = validationResultsSep.predicted.replace(labelMap)
print("confusion matrix (validation)")
print(pd.crosstab(validationResultsSep.category, validationResultsSep.predicted))
print("Validation accuracy", np.mean(validationResultsSep.category == validationResultsSep.predicted))

## Print and plot misclassified results
wrongResults = validationResultsSep[validationResultsSep.predicted != validationResultsSep.category]
rows = np.random.choice(wrongResults.index, min(4, wrongResults.shape[0]), replace=False)
print("Example wrong results (validation data)")
print(wrongResults.sample(min(10, wrongResults.shape[0])))
if plot:
    plt.figure(figsize=(12, 12))
    index = 1
    for row in rows:
        filename = wrongResults.loc[row, 'filename']
        predicted = wrongResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    # now show correct results
    index = 5
    correctResults = validationResultsSep[validationResultsSep.predicted == validationResultsSep.category]
    rows = np.random.choice(correctResults.index,
                            min(4, correctResults.shape[0]), replace=False)
    for row in rows:
        filename = correctResults.loc[row, 'filename']
        predicted = correctResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    plt.tight_layout()
    plt.show()

## Training data predictions.
print(" --- Predicting on training data: ---")
# do another generator: the same as training, just w/o shuffle
predictTrainGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(trainingResultsSep,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=False  # do not shuffle!
    )
phat = model.predict(predictTrainGenerator)
trainingResultsSep['predicted'] = pd.Series(np.argmax(phat, axis=-1), index=trainingResultsSep.index)
trainingResultsSep["predicted"] = trainingResultsSep.predicted.replace(labelMap)
print("confusion matrix (training)")
print(pd.crosstab(trainingResultsSep.category, trainingResultsSep.predicted))
print("Train accuracy", np.mean(trainingResultsSep.category == trainingResultsSep.predicted))

