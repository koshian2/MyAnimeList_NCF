import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Input, Dense, Dot
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
import requests
import json
import pickle

data = np.load("data/rating_light.npz")
rating, avg = data["Y"], data["avg"] #フルデータでやる場合はfloat16にしたほうがよいかも


# マスキング（True＝データがない）
na_mask = rating == 0
# レーティングを平均で引いて調整
rating -= avg
# データがあって0のところは微小な値（float32であることに注意）
rating[~na_mask & (rating==0)] = 1e-7
# データがないところは0
rating[na_mask] = 0

# 定数
n_anime, n_user = rating.shape
print(n_anime, n_user)
# 潜在特徴量の次元数
latent_units = 16

# NCFモデル
input_a = Input(shape=(n_anime, ))
x_a = Dense(64, activation="relu")(input_a)
x_a = Dense(32, activation="relu")(x_a)
x_a = Dense(latent_units, activation="tanh", name="latent_user")(x_a)

input_b = Input(shape=(n_user, ))
x_b = Dense(64, activation="relu")(input_b)
x_b = Dense(32, activation="relu")(x_b)
x_b = Dense(latent_units, activation="tanh", name="latent_anime")(x_b)
y = Dot(axes=-1)([x_a, x_b])
model = Model(inputs=[input_a, input_b], outputs=y)
print(model.summary())
exit()

# fit_generator
def fit_generator():
    global rating, n_anime, n_user
    while True:
        for i in np.random.permutation(n_user):
            yield [np.tile(rating[:,i], (n_anime, 1)), rating], rating[:,i][:, np.newaxis]

# 損失関数
def loss_function(y_true, y_pred):
    return K.square(y_true - y_pred) * K.square(K.sign(y_true))

# コンパイル
model.compile(optimizer=Adam(lr=0.0001), loss=loss_function)

# コールバック
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        slack_str = "epoch={0:03f}, loss={1:.7f}".format(epoch, logs["loss"])
        requests.post("Enter your slack url", data = json.dumps(
        {
        "text": slack_str,
        "username": "Enter your user name", 
        "icon_emoji": ":musical_score:", 
        "link_names": 1, 
        }))

# 訓練
n_epochs = 25
mycallback = MyCallback()
history = model.fit_generator(generator=fit_generator(), 
                    steps_per_epoch=n_user, epochs=n_epochs, callbacks=[mycallback]).history

model.save("data/ncf_model.h5")

# ヒストリの記録
with open("data/history.dat", "wb") as fp:
    pickle.dump(history, fp)


