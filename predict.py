from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import optimizers
import keras
import numpy as np
import sys
from PIL import Image

labels = ["girl", "boy"]
number_labels = len(labels)
image_size = 100

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100, 100, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics=["accuracy"]
                    )

    model = load_model("./data_cnn2.h5")

    return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(labels[predicted], percentage))

if __name__ == "__main__":
    main()
