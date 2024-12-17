from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Khởi tạo app
app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anime-fawn-five.vercel.app"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_user_rating = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))
duplicates = df_user_rating[df_user_rating.duplicated(subset=["User_id", "Anime_id"], keep=False)]
df_user_rating = df_user_rating.drop_duplicates(subset=["User_id", "Anime_id"], keep="first")

# Xử lý dữ liệu UserRating
df_user_rating["Rating"] = df_user_rating["Rating"].apply(lambda x: 1 if x >= 7 else (-1 if x <= 6 else 0))

# Tạo ma trận animes_users (cho cả hai API)
animes_users = df_user_rating.pivot(index="Anime_id", columns="User_id", values="Rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

# Huấn luyện mô hình KNN cho Anime tương tự
model_anime = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_anime.fit(mat_anime)

# Huấn luyện mô hình KNN cho người dùng tương tự
model_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_user.fit(mat_anime.T)  # Ma trận chuyển vị để tìm người dùng tương tự

# Hàm chuyển đổi ObjectId thành JSON serializable
def jsonable(data):
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

import numpy as np

# Hàm tính khoảng cách cosine giữa hai vector
def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # Xử lý trường hợp vector rỗng
        return 1  # Khoảng cách lớn nhất
    return 1 - dot_product / (norm_vec1 * norm_vec2)  # Cosine distance (1 - Cosine similarity)


##################################
# API 1: Gợi ý anime dựa trên Anime_id
##################################
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
    return jsonable(recommendations)


##################################
# API 2: Gợi ý anime dựa trên User_id
##################################
@app.post("/knn")
async def recommend_by_user(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if user_id not in animes_users.columns:
        return {"error": f"User ID {user_id} không tồn tại trong dữ liệu."}

    # Tìm các người dùng tương tự
    user_idx = animes_users.columns.get_loc(user_id)
    distances, indices = model_user.kneighbors(mat_anime.T[user_idx], n_neighbors=len(animes_users.columns))

    # Đếm tần suất các anime từ người dùng tương tự
    anime_counter = Counter()

    for i in indices.flatten():
        if i != user_idx:  # Loại bỏ chính người dùng
            similar_user = animes_users.iloc[:, i]
            for anime_id, rating in similar_user.items():
                if rating == 1:  # Chỉ xét anime có rating >= 7
                    anime_counter[anime_id] += 1

    # Loại bỏ anime mà người dùng hiện tại đã xem
    user_anime = set(animes_users.loc[:, user_id][animes_users.loc[:, user_id] != 0].index)
    anime_counter = {anime_id: count for anime_id, count in anime_counter.items() if anime_id not in user_anime}

    # Sắp xếp và lấy top `n` anime
    sorted_anime = sorted(anime_counter.items(), key=lambda x: x[1], reverse=True)
    recommendations = []
    for anime_id, _ in sorted_anime[:n]:
        anime_data = df_anime[df_anime['Anime_id'] == anime_id].iloc[0].to_dict()
        recommendations.append(anime_data)

    return jsonable(recommendations)


# Chạy server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4003))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("knn:app", host="0.0.0.0", port=port, reload=True)
