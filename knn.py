from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId

# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Cho phép origin cụ thể
    allow_credentials=True,
    allow_methods="*",
    allow_headers="*",
)

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_user_rating = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))

# Xử lý dữ liệu UserRating
df_user_rating["Rating"] = df_user_rating["Rating"].apply(lambda x: 1 if x >= 7 else (-1 if x <= 6 else 0))

# Tạo ma trận animes_users
animes_users = df_user_rating.pivot(index="User_id", columns="Anime_id", values="Rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

# Bước 2: Huấn luyện mô hình KNN để tìm các người dùng tương tự
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model.fit(mat_anime)


# Pivot bảng để tạo ma trận sparse
animes_users2 = df_user_rating.pivot(index="Anime_id", columns="User_id", values="Rating").fillna(0)
mat_anime2 = csr_matrix(animes_users2.values)

# Huấn luyện mô hình KNN
model2 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model2.fit(mat_anime2)

# Hàm gợi ý anime theo anime_id
def recommender_by_id(anime_id, mat_anime2, n):
    if anime_id not in animes_users2.index:
        return {"error": "Anime ID không tồn tại"}

    idx = animes_users2.index.get_loc(anime_id)
    distances, indices = model.kneighbors(mat_anime2[idx], n_neighbors=n)
    recommendations = []

    for i in indices.flatten():
        if i != idx:  # Loại bỏ anime hiện tại
            anime_data = df_anime[df_anime['Anime_id'] == animes_users2.index[i]].iloc[0].to_dict()
            recommendations.append(anime_data)
    return recommendations

def jsonable(data):
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

# API POST để gợi ý anime dựa vào người dùng
from collections import Counter

@app.post("/knn")
async def recommend(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if user_id not in animes_users.index:
        return {"error": f"User ID {user_id} không tồn tại trong dữ liệu"}

    # Tìm người dùng tương tự
    user_idx = animes_users.index.get_loc(user_id)
    distances, indices = model.kneighbors(mat_anime[user_idx], n_neighbors=len(animes_users))

    # Đếm tần suất anime từ người dùng gần nhất
    anime_counter = Counter()

    for i in indices.flatten():
        if i != user_idx:  # Loại bỏ chính người dùng
            similar_user = animes_users.iloc[i]
            # Đếm các anime mà người dùng này đã đánh giá
            for anime_id, rating in similar_user.items():
                if rating == 1:  # Chỉ xét các anime có rating >= 7 (hoặc tương đương)
                    anime_counter[anime_id] += 1

    # Loại bỏ anime mà người dùng hiện tại đã xem
    user_anime = set(animes_users.iloc[user_idx][animes_users.iloc[user_idx] != 0].index)
    anime_counter = {anime_id: count for anime_id, count in anime_counter.items() if anime_id not in user_anime}

    # Sắp xếp anime theo tần suất giảm dần
    sorted_anime = sorted(anime_counter.items(), key=lambda x: x[1], reverse=True)

    # Lấy thông tin anime từ df_anime
    recommendations = []
    for anime_id, _ in sorted_anime[:n]:  # Chỉ lấy tối đa `n` anime
        anime_data = df_anime[df_anime['Anime_id'] == anime_id].iloc[0].to_dict()
        recommendations.append(anime_data)

    return jsonable(recommendations)

# API POST để gợi ý anime dựa vào anime vừa xem
@app.post("/knn/anime")
async def recommendByAnime(request: Request):
    data = await request.json()
    anime_id = data.get("anime_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if anime_id is None:
        return {"error": "Vui lòng cung cấp animeId"}

    # Gọi hàm recommender_by_id
    result = jsonable(recommender_by_id(anime_id, mat_anime, n))
    return result


import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    uvicorn.run("knn:app", host="0.0.0.0", port=port)
