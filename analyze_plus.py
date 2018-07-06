import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# アニメのタイトルを部分一致で検索
def anime_lookup(title, outputs=True):
    filter = anime[anime.name.str.contains(title)]
    if outputs: print(filter)

# anime_id -> ratingの行列のインデックスのマッピング
def find_anime_index(anime_id):
    filter = data["map_anime_id"][data["map_anime_id"][:,1]==anime_id, :]
    if filter.shape[0] == 0:
        return None
    else:
        return filter[0,0]

# ratingの行列 -> anime_idのインデックスのマッピング
def find_anime_id(anime_index):
    if anime_index < 0 or anime_index >= data["map_anime_id"].shape[0]:
        return None
    else:
        return data["map_anime_id"][anime_index, 1]

# anime_idからアニメタイトルを取得
def get_anime_name(anime_id):
    filter = anime[anime.anime_id == anime_id].name.values
    return filter[0]

# 類似度トップを表示
# コサイン類似度
def print_similar(anime_id, rank=10):
    index = find_anime_index(anime_id)
    if index == None: return
    cosine = cosine_similarity(anime_features[index, :].reshape(1, -1), anime_features)
    # 類似度トップ10
    similar_indices = np.argsort(np.ravel(cosine))[::-1][:(rank+1)]
    cnt = 1
    print("--", get_anime_name(anime_id), "と似ているアニメ --")
    for idx in similar_indices:
        aid = find_anime_id(idx)
        if aid == anime_id: continue
        print("第", cnt, "位 :", get_anime_name(aid), "(id =", aid, ")", "類似度 =", cosine[0, idx])
        cnt += 1
    print()

def print_similar_vec(vec, rank=10):
    cosine = cosine_similarity(vec.reshape(1, -1), anime_features)
    # 類似度トップ10
    similar_indices = np.argsort(np.ravel(cosine))[::-1][:(rank+1)]
    cnt = 1
    for idx in similar_indices:
        aid = find_anime_id(idx)
        print("第", cnt, "位 :", get_anime_name(aid), "(id =", aid, ")", "類似度 =", cosine[0, idx])
        cnt += 1
    print()

if __name__ == "__main__":
    # データの読み込み
    anime = pd.read_csv("data/anime.csv")
    data = np.load("data/rating_light.npz")
    anime_features = np.load("data/anime_features.npz")["anime_features"]

    ## 似ているアニメ
    # 初代ガンダムを探す
    anime_lookup("Mobile Suit")
    # 初代ガンダムはanime_id = 80
    print_similar(80)

    # きんいろモザイク
    anime_lookup("Kiniro Mosaic")
    # anime_id = 16732（1期）
    print_similar(16732)
    # anime_id = 23269（2期）
    print_similar(23269)

    # ラブライブ
    anime_lookup("Love Live")
    print_similar(15051)

    ## 足し算、引き算
    # ラブライブ＋ガンダム
    combined = anime_features[find_anime_index(15051), :] + anime_features[find_anime_index(80), :]
    print("ラブライブ＋ガンダムに似ているアニメ")
    print_similar_vec(combined)

    # ごちうさ(21273)、School Days(2476)
    anime_lookup("Usagi")
    anime_lookup("School Days")
    print_similar(21273)
    print_similar(2476)
    # ごちうさ(21273)＋0.25×School Days(2476)
    # 1でかけるとSchool Daysが強すぎるので希釈
    combined = anime_features[find_anime_index(21273), :] + 0.25*anime_features[find_anime_index(2476), :]
    print("ごちうさ＋0.25*School Daysに似ているアニメ")
    print_similar_vec(combined)

    # 天空の城ラピュタ（513）+鬼父（6893）-パパのいうことを聞きなさい（11179）
    anime_lookup("Oni Chichi")
    anime_lookup("Papa no")
    anime_lookup("Laputa")
    # Hentai成分が強すぎるので0.5かけて希釈
    combined = anime_features[find_anime_index(513), :] + 0.5*(anime_features[find_anime_index(6893), :] - anime_features[find_anime_index(11179), :]) 
    print("天空の城ラピュタ+0.5×（鬼父－パパのいうことを聞きなさい）に似ているアニメ")
    print_similar_vec(combined)

    # ガルパン(14131)
    anime_lookup("Girls und")
    print_similar(14131)
    # ガルパン(14131)+Free(18507)-けいおん(5680)
    combined = anime_features[find_anime_index(14131), :] + anime_features[find_anime_index(18507), :] - anime_features[find_anime_index(5680), :] 
    print("ガルパン+Free-けいおんに似ているアニメ")
    print_similar_vec(combined)

    # Tiger&Bunny(9941), Free1期(18507), けいおん1期(5680)
    print_similar(9941)
    print_similar(18507)
    print_similar(5680)
    # Tiger&Bunny + けいおん - Free
    combined = anime_features[find_anime_index(9941), :] - anime_features[find_anime_index(18507), :] + anime_features[find_anime_index(5680), :] 
    print("TigerBunny－Free＋けいおんに似ているアニメ")
    print_similar_vec(combined)
