# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
## Neural Network Model

![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/b2fcd796-b12b-4056-a609-a5b705ddb6a2)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:
Build a CNN model
### STEP 3:
Compile and fit the model and then predict


## PROGRAM
### LIBRARIES
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
### DATA LOADING AND SHAPING
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
### ONE HOT ENCODING
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
### RESHAPE INPUTS
```
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
### BUILD CNN MODEL
```
ai_model = keras.Sequential()
ai_model.add(layers.Input(shape=(28,28,1)))
ai_model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
ai_model.add(layers.MaxPool2D(pool_size=(2,2)))
ai_model.add(layers.Conv2D(filters=64,kernel_size=(5,5),activation='relu'))
ai_model.add(layers.Flatten())
ai_model.add(layers.Dense(24,activation='relu'))
ai_model.add(layers.Dense(28,activation='relu'))
ai_model.add(layers.Dense(10,activation='softmax'))
```
### METRICS
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
### PREDICTING FOR OWN HANDWRITTEN INPUT
```
img = image.load_img('img1.png')

type(img)

img = image.load_img('img1.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)


print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```
## OUTPUT
### Accuracy, Validation Accuracy Vs Iteration:
![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/17907f26-6657-4dfd-834c-11dafdfd27f2)

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/a3066b09-8100-4e33-ae99-d8ffe15ce7f1)


### Classification Report

![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/2185895d-0459-47c4-a0db-61783121ca4e)


### Confusion Matrix

![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/ec75c998-6b3e-4dbc-a99e-eb408b988672)


### New Sample Data Prediction
![img1](https://github.com/balaji-21005757/mnist-classification/assets/94372294/4a406732-0ede-4149-b19f-3207de95c348)
![image](https://github.com/balaji-21005757/mnist-classification/assets/94372294/59a3629e-558e-479b-8184-f1cc46dbb544)



## RESULT
Thus to create a program for creating convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is created and compiled successfully.
