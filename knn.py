from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from pymongo import MongoClient
from bson import ObjectId
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np

def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  
        return 1 
    return 1 - dot_product / (norm_vec1 * norm_vec2)  # Cosine distance (1 - Cosine similarity).
  

def find_k_nearest_neighbors(matrix, target_vector, k):
    distances = []
    for idx, vec in enumerate(matrix):
        dist = cosine_distance(target_vector, vec)
        distances.append((idx, dist))
    
    distances = sorted(distances, key=lambda x: x[1])
    return distances[:k]



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

# Hàm chuyển đổi ObjectId thành JSON serializable
def jsonable(data):
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

##################################
# API 1: Gợi ý anime dựa trên Anime_id
##################################
@app.post("/knn/anime")
async def recommend_by_anime(request: Request):
    data = await request.json()
    anime_id = data.get("anime_id")
    n = data.get("n", 10)

    if anime_id not in animes_users.index:
        return {"error": "Anime ID không tồn tại"}

    # Tìm các anime tương tự
    idx = animes_users.index.get_loc(anime_id)
    target_vector = mat_anime[idx]
    neighbors = find_k_nearest_neighbors(mat_anime, target_vector, n + 1)

    # Gợi ý các anime
    recommendations = []
    for i, _ in neighbors:
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
    n = data.get("n", 10)

    if user_id not in animes_users.columns:
        return {"error": f"User ID {user_id} không tồn tại trong dữ liệu."}

    # Tìm các người dùng tương tự
    user_idx = animes_users.columns.get_loc(user_id)
    target_vector = mat_anime[:, user_idx]
    neighbors = find_k_nearest_neighbors(mat_anime.T, target_vector,5)

    # Đếm tần suất các anime từ người dùng tương tự
    anime_counter = Counter()
    for i, _ in neighbors:
        if i != user_idx:
            similar_user = animes_users.iloc[:, i]
            for anime_id, rating in similar_user.items():
                if rating == 1:
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
