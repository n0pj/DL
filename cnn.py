import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from matplotlib import pyplot as plt
import datetime

base_dir = "./data/"
labels = os.listdir(base_dir)
labels_count = len(labels)
image_size = 2 ** 6

def main():
    x_train, x_test, y_train, y_test = np.load("./numpy_data.npy")
    x_train, x_test = x_train.astype("float") / 255, x_test.astype("float") / 255
    y_train, y_test = tf.one_hot(y_train, labels_count), tf.one_hot(y_test, labels_count)
    model, model2, fig, axL, axR= train_model(x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
        
    plot_history_loss(model2, axL)
    plot_history_acc(model2, axR)
    fig.savefig('./data.png')
    plt.close()

def train_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(128, input_shape=(x_train.shape[1:])))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    
    model.add(Dense(image_size * image_size))
    model.add(Activation("relu"))
    
    model.add(Dense(labels_count))
    model.add(Activation("softmax"))

    adam = tf.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-6)

    model.compile(optimizer=adam,
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])


    log_dir="log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model2 = model.fit(x_train, y_train,
              batch_size=128, epochs=50,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])

    model.summary()
    model.save("./weight.h5")
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    
    return model, model2, fig, axL, axR

def plot_history_loss(fit, axL):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

def plot_history_acc(fit, axR):
    # Plot the loss in the history
    axR.plot(fit.history['accuracy'],label="loss for training")
    axR.plot(fit.history['val_accuracy'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss: ", score[0])
    print("Accu: ", score[1])

if __name__ == "__main__":
    main()