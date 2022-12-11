'''

    opis zadan w pliku: opis_zadan.txt

    Program uczy sieci neuronowe na podstawie danych z CIFAR10, u nas konkretnie dla ptaków i jelenii
    oraz wyświetla w konsoli proces treningu sieci neuronowych

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots_NAI5

'''

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_1_index = np.where(y_train.reshape(-1) == 2)

x_train_class_1 = x_train[class_1_index]
y_train_class_1 = y_train[class_1_index]

class_2_index = np.where(y_train.reshape(-1) == 4)

x_train_class_2 = x_train[class_2_index]
y_train_class_2 = y_train[class_2_index]

x_train = np.concatenate((x_train_class_1, x_train_class_2))
y_train = np.concatenate((y_train_class_1, y_train_class_2)).reshape(-1, 1)

print('Filtered Images Shap1e: {}'.format(x_train.shape))
print('Filtered Images Shape2: {}'.format(y_train.shape))

class_1_index2 = np.where(y_test.reshape(-1) == 2)
x_test_class_1 = x_test[class_1_index2]
y_test_class_1 = y_test[class_1_index2]

class_2_index2 = np.where(y_test.reshape(-1) == 4)
x_test_class_2 = x_test[class_2_index2]
y_test_class_2 = y_test[class_2_index2]

x_test = np.concatenate((x_test_class_1, x_test_class_2))
y_test = np.concatenate((y_test_class_1, y_test_class_2)).reshape(-1, 1)

#Split them into train & test
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

K = len(set(y_train))
print("number of classes:", K)

y_test_old = y_test

for i in range(0, len(y_test)):
    if y_test[i] == 2:
        y_test[i] = 0
    if y_test[i] == 4:
        y_test[i] = 1

for i in range(0, len(y_train)):
    if y_train[i] == 2:
        y_train[i] = 0
    if y_train[i] == 4:
        y_train[i] = 1

i = Input(shape=x_train[0].shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
r = model.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=1)

params = {'random_state': 4, 'max_depth': 4}

nsamples = x_train.shape
d2_train_dataset = x_train.reshape((nsamples))

N_samples, img_width, img_height, ch = x_test.shape
x_test = x_test.reshape(N_samples, -1 )

classifier = DecisionTreeClassifier(**params)
classifier.fit(x_test, y_test_old)

y_test_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test_old, y_test_pred)
print('Confusion matrix\n\n', cm)