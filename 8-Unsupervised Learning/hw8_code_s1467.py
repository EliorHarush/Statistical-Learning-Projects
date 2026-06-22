import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setup identifiers and paths for immediate execution
student_id = "s1467"
file_path = "hw8_corpus_s1467.csv"

print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# ==========================================
# 1. Representation & Preprocessing
# ==========================================
# Use TF-IDF, remove english stop words, and ignore extremely rare/common words
print("Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.95)
X_tfidf = vectorizer.fit_transform(df['text'])

# ==========================================
# 2. Dimension Reduction
# ==========================================
# Reduce to 50 components using Truncated SVD (Latent Semantic Analysis)
print("Performing Dimensionality Reduction (Truncated SVD)...")
svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(X_tfidf)

# ==========================================
# 3. Determine Optimal K (Number of Clusters)
# ==========================================
print("Evaluating optimal K using Silhouette Scores...")
silhouette_scores = []
K_range = range(2, 15)

best_k = 2
best_score = -1

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_lsa)
    score = silhouette_score(X_lsa, labels)
    silhouette_scores.append(score)
    
    if score > best_score:
        best_score = score
        best_k = k
        
print(f"Optimal K found: {best_k} (Silhouette Score: {best_score:.4f})")

# Generate and save Silhouette Score Plot for the presentation slides
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Score vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal K = {best_k}')
plt.legend()
plt.grid(True)

plot_filename = f'hw8_slides_plot_{student_id}.png'
plt.savefig(plot_filename)
print(f"Saved optimal K plot as {plot_filename}")

# ==========================================
# 4. Final Clustering Model
# ==========================================
print("Fitting final K-Means model...")
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df['cluster'] = final_kmeans.fit_predict(X_lsa)

# Extract top keywords per cluster to provide interpretation
terms = vectorizer.get_feature_names_out()
original_space_centroids = svd.inverse_transform(final_kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

print("\n--- Cluster Interpretations ---")
for i in range(best_k):
    top_words = [terms[ind] for ind in order_centroids[i, :5]]
    print(f"Cluster {i+1}: {', '.join(top_words)}")

# ==========================================
# 5. Format and Save Submission JSON
# ==========================================
# Note: K-means outputs 0-indexed labels; we shift to 1-indexed integers.
clusters_list = []
for _, row in df.iterrows():
    clusters_list.append({
        "document_id": row['document_id'],
        "cluster": int(row['cluster']) + 1
    })
    
submission_data = {
    "student_id": student_id,
    "K_hat": int(best_k),
    "clusters": clusters_list
}

output_filename = f'hw8_results_{student_id}.json'
with open(output_filename, 'w') as f:
    json.dump(submission_data, f, indent=4)
    
print(f"\nSuccessfully saved submission file: {output_filename}")