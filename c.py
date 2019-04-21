import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n = 10
x0 = np.linspace(-2.0, 2.0, n)
a0 = 3.0
a1 = 2.0
b0 = 1.0
y0 = np.zeros((n,1))
y0[:,0] = a0*x0+a1*x0**2 + b0 + 3*np.cos(20*x0)

# plt.plot(x0,y0 )
# plt.show()
# plt.savefig("graph.png")

def make_phi(x0,n,k):    
    phi = np.array([x0**j for j in range(k)])
    return phi.T
def build_model(d_input,d_middle):
    inputs = tf.keras.Input(shape=(d_input,))  #インプットの次元を指定
    x = layers.Dense(d_middle, activation='relu')(inputs) #中間層の指定
    y = layers.Dense(1)(x) #最終層の指定
    adam = optimizers.Adam() #最適化にはAdamを使用
    model =  tf.keras.Model(inputs=inputs, outputs=y) #モデルのインプットとアウトプットを指定

    model.compile(optimizer=adam,
              loss='mean_squared_error') #modelの構築。平均二乗誤差をloss関数とした。

    return model

k = 4
phi = make_phi(x0,n,k)
d_type = tf.float32
d_input = k
d_middle = 10
model = build_model(d_input,d_middle)
history = model.fit(phi, y0, epochs=2000,verbose=1)
ytest = model.predict(phi)
plt.plot(x0,y0 )
plt.plot(x0,ytest,'o')
plt.show()
plt.savefig("graph.png")

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
plt.savefig("train.png")
model.summary()
model.get_weights()