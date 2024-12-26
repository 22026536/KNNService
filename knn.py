from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
import joblib
import os

app = FastAPI()

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_user_rating = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))

# Xử lý dữ liệu UserRating
df_user_rating["Rating"] = df_user_rating["Rating"].apply(lambda x: 1 if x >= 7 else (-1 if x <= 6 else 0))

# Tạo ma trận animes_users (cho cả hai API)
animes_users = df_user_rating.pivot(index="Anime_id", columns="User_id", values="Rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

# Kiểm tra xem mô hình đã lưu có tồn tại không, nếu không sẽ huấn luyện lại
if os.path.exists("model_anime.pkl"):
    model_anime = joblib.load("model_anime.pkl")
else:
    model_anime = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    model_anime.fit(mat_anime)
    joblib.dump(model_anime, "model_anime.pkl")

if os.path.exists("model_user.pkl"):
    model_user = joblib.load("model_user.pkl")
else:
    model_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    model_user.fit(mat_anime.T)  # Ma trận chuyển vị để tìm người dùng tương tự.
    joblib.dump(model_user, "model_user.pkl")

# API gợi ý anime (ví dụ đã có)
@app.post("/knn/anime")
async def recommend_by_anime(request: Request):
    data = await request.json()
    anime_id = data.get("anime_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if anime_id not in animes_users.index:
        return {"error": "Anime ID không tồn tại"}

    # Tìm các anime tương tự
    idx = animes_users.index.get_loc(anime_id)
    distances, indices = model_anime.kneighbors(mat_anime[idx], n_neighbors=n + 1)

    # Gợi ý các anime
    recommendations = []
    for i in indices.flatten():
        if i != idx:  # Loại bỏ anime hiện tại
            anime_data = df_anime[df_anime['Anime_id'] == animes_users.index[i]].iloc[0].to_dict()
            recommendations.append(anime_data)
    return recommendations
