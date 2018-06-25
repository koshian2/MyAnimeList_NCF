import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model, Model
import pickle

data = np.load("data/rating_light.npz")
rating, avg = data["Y"], data["avg"]

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

# 損失関数
def loss_function(y_true, y_pred):
    return K.square(y_true - y_pred) * K.square(K.sign(y_true))
# NCFモデル
ncf = load_model("data/ncf_model.h5",  custom_objects={'loss_function': loss_function})

# モデルを作り変える
latent_model = Model(inputs=ncf.input, outputs=ncf.get_layer("latent_anime").output)
# アニメ単位の特徴量計算（ユーザー単位のは使わないので適当に0を入れておく）
anime_features = latent_model.predict([np.zeros((n_anime, n_anime)), rating])

np.savez("data/anime_features", anime_features=anime_features)

# ヒストリのプロット
with open("data/history.dat", "rb") as fp:
    history = pickle.load(fp)
plt.plot(np.arange(len(history["loss"]))+1, history["loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
