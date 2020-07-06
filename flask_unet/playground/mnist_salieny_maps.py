import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from keras import backend as K
from deepexplain.tensorflow import DeepExplain
import os

os.listdir("data")

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# Convert to binary classification task
x_bitrain = x_train[np.argwhere(y_train < 2).flatten(), :, :, :]
y_bitrain = y_train[np.argwhere(y_train < 2).flatten()]
x_bitest = x_test[np.argwhere(y_test < 2).flatten(), :, :, :]
y_bitest = y_test[np.argwhere(y_test < 2).flatten()]
num_classes = 2

# Generate modified shortcut dataset
sc = np.copy(x_bitrain[1,4:13,14:23,0])
# Modify images of 0 in training data
x_mtest = np.copy(x_bitest)
y_mtrain = y_bitrain
y_mtest = y_bitest
x_mtrain = np.copy(x_bitrain)
x_mtrain[np.argwhere(y_mtrain==0).flatten(),1:10,1:10,0] = sc

print(x_mtest.shape)
print(y_mtest.shape)

# Build model
model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Train model
batch_size = 128
epochs = 3

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_mtrain, y_mtrain,
          batch_size=500,
          epochs=epochs,
          verbose=1,
          validation_data=(x_mtest, y_mtest))

# Generate saliency maps
with DeepExplain(session=K.get_session()) as de:
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    target_tensor = fModel(input_tensor)
    
    xs = x_mtest[0:100]
    
    attributions_gradin = de.explain('saliency', target_tensor, input_tensor, xs)

for i, sal_map in enumerate(attributions_gradin):
    np.save("data/img_" + str(i), (xs[i] * 255).reshape(28, 28))
    np.save("data/" + str(i), sal_map.reshape(28, 28))