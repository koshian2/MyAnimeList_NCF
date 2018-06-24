import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# アニメのマスターデータ
anime = pd.read_csv("data/anime.csv")
print(anime.head())

# アニメ数
print(np.max(anime["anime_id"])) # 34527

# レーティングのデータ
# 重くてgithubに上げられなかったのでKaggleからDLしてください
# https://www.kaggle.com/CooperUnion/anime-recommendations-database
rating = pd.read_csv("data/rating.csv")
print(rating.head())
print(rating.shape) # (7813737, 3)

# rating=-1を除外する（見たことがあるがレーティングしていないケース）
rating = rating[rating.rating > 0]
print(rating.shape) # (6337241, 3)

# 作品単位の平均を計算する
anime_avg = rating.groupby("anime_id")[["rating"]].mean()
# レーティングがついているアニメが9927個しかない（作品単位の平均>0は保証される）
print(len(anime_avg)) # 9927

# ユーザー単位の平均を計算する
user_avg = rating.groupby("user_id")[["rating"]].mean()
# 1回以上レーティングをした人は69600人（ユーザー単位の平均>0も保証される）
print(len(user_avg)) # 69600

# このままフルで学習させるとCPUで1epochに5日かかるのでデータを軽くする。データが少ないものは切る。
# 計算資源が無限に使えるならそのままのデータでどうぞ
# 作品単位のレビュー数を集計
anime_num = rating.groupby("anime_id")[["rating"]].count()
# 25%クォンタイル=9、中央値＝57
print(anime_num.describe())
# レビューが50未満の作品を無視する
anime_avg = anime_avg.loc[anime_num.index.values[anime_num.rating >= 50]]
print(len(anime_avg)) # 5172

# ユーザーも同様にデータが少ないものは切る
# ユーザー単位のレビュー数
# 作品単位のレビュー数を集計
user_num = rating.groupby("user_id")[["rating"]].count()
# 25%Qt=13、中央値＝45、75%Qt=114
print(user_num.describe())
# レビューが100未満のユーザーを無視する（かなりばっさり切る）
user_avg = user_avg.loc[user_num.index.values[user_num.rating >= 100]]
print(len(user_avg)) # 19949

# レーティングのデータのフィルタリング
rating = rating[rating.anime_id.isin(anime_avg.index.values)]
rating = rating[rating.user_id.isin(user_avg.index.values)]
print(rating.shape) # (4704219, 3)

# 定数
n_anime, n_user = len(anime_avg), len(user_avg)

# anime_id -> indexのマッピング
map_anime_id = np.zeros((n_anime, 2), dtype=int)
map_anime_id[:, 0] = np.arange(n_anime) # Yのindex
map_anime_id[:, 1] = anime_avg.index.values # anime_id
# user_id -> indexのマッピング
map_user_id = np.zeros((n_user, 2), dtype=int)
map_user_id[:, 0] = np.arange(n_user) # Yのindex
map_user_id[:, 1] = user_avg.index.values # user_id
print(map_anime_id[:10, :])
print(map_user_id[:10, :])

# レーティングを行列に変換
Y = np.zeros((n_anime, n_user), dtype="float32")
for i in rating.index.values:
    # user_id, anime_id ≠ インデックスなのに注意
    idx_anime = np.min(map_anime_id[np.where(rating["anime_id"][i] == map_anime_id[:, 1]), 0])
    idx_user = np.min(map_user_id[np.where(rating["user_id"][i] == map_user_id[:, 1]), 0])
    Y[idx_anime, idx_user] = rating["rating"][i]

# Numpy配列として保存
np.savez_compressed("data/rating_light", Y=Y, avg=anime_avg.values, 
                    map_anime_id=map_anime_id, map_user_id=map_user_id)
