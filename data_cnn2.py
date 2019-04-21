from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import optimizers
import keras
import numpy as np

#ラベルの指定
labels = ["girl", "boy"]
number_labels = len(labels)

#mainを定義
def main():
    #用意したデータを読み込む
    X_train, X_test, y_train, y_test = np.load("./data_aug.npy")
    #データの正規化
    #全部で256階調あり、最大値で割って0〜1にする
    #整数をfloatに変換
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    #one hot vector,正解は１、それ以外はゼロという行列に変換
    y_train = np_utils.to_categorical(y_train, number_labels)
    y_test = np_utils.to_categorical(y_test, number_labels)

    #モデルの学習と評価
    model = model_train(X_train, y_train)
    #model評価
    model_eval(model, X_test, y_test)

def model_train(X, y):
    #modelをシーケンシャルに
    model = Sequential()
    #32個のフィルターを3＊3、paddingは畳み込み結果が同じサイズになるようにピクセルを左右に足す
    #shape,０番目はいらないので、[1:]１番目以降のみ
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X.shape[1:]))
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

    #トレーニングの回数
    model.fit(X, y, batch_size=8, epochs=80)

    #modelの保存
    model.save("./data_cnn2.h5")

    return model

def model_eval(model, X, y):
    #verbose,途中の結果を表示
    score = model.evaluate(X, y, verbose=1)
    print("Test loss: ", score[0])
    print("Test accuracy ", score[1])

if __name__ == "__main__":
    main()
