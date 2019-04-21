from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection
import datetime

base_dir = "./data/"
labels = os.listdir(base_dir)
number_labels = len(labels)
image_size = 2 ** 6
num_testdata = 50

x_train = []
x_test = []
y_train = []
y_test = []
for index, label in enumerate(labels):
    images_dir = "./data/" + label
    files = glob.glob(images_dir + "/*.jpg")
    print("phase: {}".format(index))

    for index_file, image_file in enumerate(files):
        img = Image.open(image_file)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)

        if index_file > num_testdata:
            for angle in range(-40, 40, 5):

                img_rotate = img.rotate(angle)
                data = np.asarray(img_rotate)
                x_test.append(data)
                y_test.append(index)

                img_trans = img.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                x_test.append(data)
                y_test.append(index)
                
        else:
            for angle in range(-40, 40, 5):

                img_rotate = img.rotate(angle)
                data = np.asarray(img_rotate)
                x_train.append(data)
                y_train.append(index)

                img_trans = img.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                x_train.append(data)
                y_train.append(index)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

xy = (x_train, x_test, y_train, y_test)
file_name = datetime.date.today()
np.save("./numpy_data.npy", xy)
print("saving data completed")