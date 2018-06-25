from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

anime = pd.read_csv("data/anime.csv")
data = np.load("data/rating_light.npz")
anime_features = np.load("data/anime_features.npz")["anime_features"]

# アニメのタイトルを部分一致で検索
def anime_lookup(title, outputs=True):
    filter = anime[anime.name.str.contains(title)]
    if outputs: print(filter)
# ratingの行列 -> anime_idのインデックスのマッピング
def find_anime_id(anime_index):
    if anime_index < 0 or anime_index >= data["map_anime_id"].shape[0]:
        return None
    else:
        return data["map_anime_id"][anime_index, 1]

# anime_id -> ratingの行列のインデックスのマッピング
def find_anime_index(anime_id):
    filter = data["map_anime_id"][data["map_anime_id"][:,1]==anime_id, :]
    if filter.shape[0] == 0:
        return None
    else:
        return filter[0,0]


## k-means
kmeans = KMeans(n_clusters=10, random_state=114514).fit(anime_features)
#行数
for i in range(10):
    indexes = np.arange(len(kmeans.labels_))[kmeans.labels_ == i]
    print("☆Group", i+1, " / N =", len(indexes))
    anime_ids = [find_anime_id(idx) for idx in indexes]
    filter = anime[anime.anime_id.isin(anime_ids)].sort_values("members", ascending=False).take(np.arange(10))
    print(filter.name)
    print()


# ジャンル=hentaiのものを絞り込む
hentai = anime[~anime.genre.isnull() & (anime.genre.str.contains("Hentai") )]
hentai_indices = []
for i in hentai.anime_id.values:
    h = find_anime_index(i)
    if h != None:
        hentai_indices.append(h)
# グループを調べる
group4_flag = kmeans.labels_[hentai_indices] == 3
print("ジャンル＝Hentai :", len(hentai_indices), "件、うちグループ4 : ", np.sum(group4_flag))

# 誤判定
misflag_hentai_indices = np.arange(len(kmeans.labels_))[(kmeans.labels_ == 3) & ~np.isin(np.arange(len(kmeans.labels_)), hentai_indices, )]
misflag_hentai_id = [find_anime_id(i) for i in misflag_hentai_indices]
print(anime[anime.anime_id.isin(misflag_hentai_id)].sort_values("members", ascending=False).name)

## plot
pca = PCA(n_components=2).fit_transform(anime_features)
for i in range(10):
    indexes = np.arange(len(kmeans.labels_))[kmeans.labels_ == i]
    plt.plot(pca[indexes, 0], pca[indexes, 1], ".",
            label="Group "+str(i+1))
plt.plot(pca[hentai_indices, 0], pca[hentai_indices, 1], ".", color="black", alpha=0.3, label="Hentai")
plt.legend(loc="upper left")
plt.show()

# Hentaiアニメのベクトル
hentai_vec = np.mean(anime_features[np.array(hentai_indices), :], axis=0)
# 距離
hentai_dis = cosine_similarity(hentai_vec.reshape(1, -1), anime_features)
# GroupでグルーピングしたHentai距離
hentai_df = pd.DataFrame({"group_id":kmeans.labels_+1, "hentai_dis":np.ravel(hentai_dis)})
hentai_df_statics = hentai_df.groupby("group_id").describe()
print(hentai_df_statics)
