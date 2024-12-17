from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId
from collections import Counter

# Khởi tạo app
app = FastAPI()

# Middleware CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép origin cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_favorites = pd.DataFrame(list(db["UserFavorites"].find()))
df_user_rating = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))

# Chuyển đổi ObjectId
def convert_id(df, column):
    df[column] = df[column].astype(str)
convert_id(df_favorites, '_id')
convert_id(df_anime, '_id')

# Gợi ý anime theo anime_id
user_anime_matrix = pd.DataFrame([
    (user["User_id"], anime_id, 1)
    for user in df_favorites.to_dict(orient="records")
    for anime_id in user["favorites"]
], columns=["User_id", "Anime_id", "Rating"])

animes_users_1 = user_anime_matrix.pivot(index="Anime_id", columns="User_id", values="Rating").fillna(0)
mat_anime_1 = csr_matrix(animes_users_1.values)
model_1 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20).fit(mat_anime_1)

def recommender_by_id(anime_id, mat_anime, n):
    if anime_id not in animes_users_1.index:
        return {"error": "Anime ID không tồn tại"}
    idx = animes_users_1.index.get_loc(anime_id)
    distances, indices = model_1.kneighbors(mat_anime[idx], n_neighbors=n)
    return [df_anime[df_anime['Anime_id'] == animes_users_1.index[i]].iloc[0].to_dict()
            for i in indices.flatten() if i != idx]

# API 1: Gợi ý anime theo anime_id
@app.post("/recommend_by_anime_id")
async def recommend_by_anime(request: Request):
    data = await request.json()
    anime_id = str(data.get("anime_id"))
    n = data.get("n", 10)
    if not anime_id:
        return {"error": "Vui lòng cung cấp animeId"}
    return recommender_by_id(anime_id, mat_anime_1, n)

# Gợi ý anime theo user_id
df_user_rating["Rating"] = df_user_rating["Rating"].apply(lambda x: 1 if x >= 7 else (-1 if x <= 6 else 0))
animes_users_2 = df_user_rating.pivot(index="User_id", columns="Anime_id", values="Rating").fillna(0)
mat_anime_2 = csr_matrix(animes_users_2.values)
model_2 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10).fit(mat_anime_2)

@app.post("/recommend_by_user_id")
async def recommend_by_user(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)
    if user_id not in animes_users_2.index:
        return {"error": f"User ID {user_id} không tồn tại trong dữ liệu"}
    
    user_idx = animes_users_2.index.get_loc(user_id)
    distances, indices = model_2.kneighbors(mat_anime_2[user_idx], n_neighbors=len(animes_users_2))
    anime_counter = Counter()
    user_anime = set(animes_users_2.iloc[user_idx][animes_users_2.iloc[user_idx] != 0].index)

    for i in indices.flatten():
        if i != user_idx:
            similar_user = animes_users_2.iloc[i]
            for anime_id, rating in similar_user.items():
                if rating == 1:
                    anime_counter[anime_id] += 1
    anime_counter = {anime_id: count for anime_id, count in anime_counter.items() if anime_id not in user_anime}
    sorted_anime = sorted(anime_counter.items(), key=lambda x: x[1], reverse=True)

    recommendations = []
    for anime_id, _ in sorted_anime[:n]:
        anime_data = df_anime[df_anime['Anime_id'] == anime_id].iloc[0].to_dict()
        recommendations.append(anime_data)
    return recommendations

# Chạy server
import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
