import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime

tf.keras.backend.clear_session()
base_dir = "./data/"
labels = os.listdir(base_dir)
labels_count = len(labels)
image_size = 2**6
dim = 3

def main():
    x_train, x_test, y_train, y_test = np.load("./numpy_data.npy")
    x_train, x_test = x_train.astype("float") / 255.0, x_test.astype("float") / 255.0
    # x_train = x_train.reshape(3264, 12288).astype('float32') / 255
    # x_test = x_test.reshape(1312, 12288).astype('float32') / 255
    y_train, y_test = tf.one_hot(y_train, labels_count), tf.one_hot(y_test, labels_count)
    model, model2, fig, loss, acc = train_model(x_train, y_train, x_test, y_test)

    evaluate_model(model, x_test, y_test)

    plot_history_loss(model2, loss)
    plot_history_acc(model2, acc)
    fig.savefig("./graph.png")
    plt.close()

def train_model(x_train, y_train, x_test, y_test):
    inputs = keras.Input(shape=(image_size,image_size,dim), name="img")
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    b1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(b1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    b2_output = layers.add([x, b1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(b2_output)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(labels_count, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet")

    
    keras.utils.plot_model(model, "model.png", show_shapes=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=["accuracy"])

    log_dir="log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train, y_train,
                        batch_size=20,
                        epochs=20,
                        validation_split=0.2,
                        callbacks=[tensorboard_callback])

    test_scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss: {}".format(test_scores[0]))
    print("Test acc: {}".format(test_scores[1]))

    model.summary()
    model.save("./result.h5")
    fig, (loss, acc) = plt.subplots(ncols=2, figsize=(10, 4))

    return model, history, fig, loss, acc

def plot_history_loss(fit, loss):
    loss.plot(fit.history['loss'],label="loss for training")
    loss.plot(fit.history['val_loss'],label="loss for validation")
    loss.set_title('model loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend(loc='upper right')

def plot_history_acc(fit, acc):
    acc.plot(fit.history['accuracy'],label="loss for training")
    acc.plot(fit.history['val_accuracy'],label="loss for validation")
    acc.set_title('model accuracy')
    acc.set_xlabel('epoch')
    acc.set_ylabel('accuracy')
    acc.legend(loc='upper right')

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss: ", score[0])
    print("Accu: ", score[1])

if __name__ == "__main__":
    main()