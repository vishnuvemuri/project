import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

# Load dataset
data = pd.read_csv(r"D:\location\data\museums.csv", encoding='ISO-8859-1')

# Extract latitude and longitude
data[['latitude', 'longitude']] = data['(latitude,longitude)'].str.strip('()').str.split(',', expand=True).astype(float)

# Training data
X = data[['latitude', 'longitude']].values

# Train the Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='manhattan')

model.fit(X)

# ✅ Calculate Evaluation Metrics

# **1️⃣ Mean Average Distance (MAD)**
def mean_average_distance(model, X):
    distances, _ = model.kneighbors(X)
    return np.mean(distances)

mad_score = mean_average_distance(model, X)
print(f"📊 Mean Average Distance (MAD): {mad_score:.4f}")

# **2️⃣ Recall@5**
def recall_at_k(model, X, k=5):
    recall_scores = []
    for i in range(len(X)):
        distances, indices = model.kneighbors([X[i]], n_neighbors=k)
        if i in indices[0]:  # Check if the correct museum is in top K
            recall_scores.append(1)
        else:
            recall_scores.append(0)
    return np.mean(recall_scores)

recall_k_score = recall_at_k(model, X, k=5)
print(f"📌 Recall@5 Score: {recall_k_score:.4f}")

# **3️⃣ Mean Reciprocal Rank (MRR)**
def mean_reciprocal_rank(model, X):
    reciprocal_ranks = []
    for i in range(len(X)):
        distances, indices = model.kneighbors([X[i]], n_neighbors=10)
        rank = (indices[0] == i).nonzero()[0]  # Find rank of correct point
        if len(rank) > 0:
            reciprocal_ranks.append(1 / (rank[0] + 1))
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

mrr_score = mean_reciprocal_rank(model, X)
print(f"✅ Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

# ✅ Save the trained model
with open('recommend.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🚀 Model training complete with evaluation metrics and saved as recommend.pkl.")
