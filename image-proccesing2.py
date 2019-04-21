from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection
import datetime
#ラベルの指定
base_dir = "./data/"
labels = os.listdir(base_dir)
number_labels = len(labels)
image_size = 2 ** 6
num_testdata = 50

print(os.getcwd())

#画像を読み込む
x_train = []
x_test = []
y_train = []
y_test = []

for index, label in enumerate(labels):
    images_dir = "./data/" + label
    files = glob.glob(images_dir + "/*.jpg")

    for index_file, image_file in enumerate(files):
        img = Image.open(image_file)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)

        if index_file < num_testdata:
            x_test.append(data)
            y_test.append(index)
        else:
            for angle in range(-40, 40, 5):
                #画像の回転
                img_rotate = img.rotate(angle)
                data = np.asarray(img_rotate)
                x_train.append(data)
                y_train.append(index)

                #画像の反転
                img_trans = img.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                x_train.append(data)
                y_train.append(index)



#numpyの配列に変換
# X = np.array(X)
# Y = np.array(Y)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#配列を学習データと学習済データに分割する
#x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (x_train, x_test, y_train, y_test)
np.save("./data_aug.npy", xy)
