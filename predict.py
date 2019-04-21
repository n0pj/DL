from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import optimizers
import keras
import numpy as np
import sys
from PIL import Image

#ラベルの指定
labels = ["girl", "boy"]
number_labels = len(labels)
image_size = 100

def build_model():
    #modelをシーケンシャルに
    model = Sequential()
    #32個のフィルターを3＊3、paddingは畳み込み結果が同じサイズになるようにピクセルを左右に足す
    #50*50の画像で3（RGBカラー）
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100, 100, 3)))
    #正は通して、負はすべてゼロにする（捨てる）
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    #一番大きいサイズを取り出す
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #25%以下は捨てる、データの偏りをなくす
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    #全結合層
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    #出力層のノード（ガール、ボーイ）
    model.add(Dense(2))
    #それぞれの画像が一致している確率を変換
    model.add(Activation("softmax"))

    #10^-6学習レートを下げていく
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    #loss,損失関数、正解と推定値の誤差が小さくなるように最適化
    model.compile(loss="categorical_crossentropy",
    #optimizer,optを入れる、metrics,評価の値を記録
                    optimizer=opt,
                    metrics=["accuracy"]
                    )

    #modelの保存
    model = load_model("./data_cnn2.h5")

    return model

def main():
    #コマンドライン引数で与えられた2番めを指定
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
