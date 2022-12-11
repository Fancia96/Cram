'''
    opis zadan w pliku: opis_zadan.txt

    Program uczy sieci neuronowe na podstawie danych ze zbioru fashion-mnist, u nas konkretnie dla swetra i płaszcza
    oraz wyświetla w konsoli proces treningu sieci neuronowych

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots_NAI5

'''

# import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model

import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 5
INIT_LR = 1e-2
BS = 32

# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()


class_1_index = np.where(trainY.reshape(-1) == 2)
print(class_1_index)
x_train_class_1 = trainX[class_1_index]
y_train_class_1 = trainY[class_1_index]

class_2_index = np.where(trainY.reshape(-1) == 4)
print(class_2_index)
x_train_class_2 = trainX[class_2_index]
y_train_class_2 = trainY[class_2_index]

trainX = np.concatenate((x_train_class_1, x_train_class_2))
trainY = np.concatenate((y_train_class_1, y_train_class_2)).reshape(-1, 1)

print('Filtered Images Shap1e: {}'.format(trainX.shape))
print('Filtered Images Shape2: {}'.format(trainY.shape))

class_1_index2 = np.where(testY.reshape(-1) == 2)
x_test_class_1 = testX[class_1_index2]
y_test_class_1 = testY[class_1_index2]

class_2_index2 = np.where(testY.reshape(-1) == 4)
x_test_class_2 = testX[class_2_index2]
y_test_class_2 = testY[class_2_index2]

testX = np.concatenate((x_test_class_1, x_test_class_2))
testY = np.concatenate((y_test_class_1, y_test_class_2)).reshape(-1, 1)


##########################################

# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
	testX = testX.reshape((testX.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
# one-hot encode the training and testing labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)
# initialize the label names
labelNames = ["pullover", "coat"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training model...")
H = model.fit(x=trainX, y=trainY,
	validation_data=(testX, testY),
	batch_size=BS, epochs=NUM_EPOCHS)

# make predictions on the test set
preds = model.predict(testX)
# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))
# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")