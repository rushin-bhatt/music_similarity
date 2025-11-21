import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import os

# Load embeddings and data from pickle file
with open('song_embeddings.pkl', 'rb') as f:
    saved_data = pickle.load(f)

embeddings = saved_data['embeddings']
data = saved_data['data']

# Load user interaction data
# Assuming user_data has columns ['user_id', 'track_id', 'rating']
# user_data = pd.read_csv('user_data.csv')  # Uncomment and adapt this line to load your user data

# Function to find n most similar songs given a track name
def recommend_similar_songs(track_name, n=5):
    if track_name not in data['track_name'].values:
        return f"Track '{track_name}' not found in the dataset.", None, None
    
    input_index = data[data['track_name'] == track_name].index[0]
    input_song_embedding = embeddings[input_index].reshape(1, -1)
    
    similarities = cosine_similarity(input_song_embedding, embeddings).flatten()
    
    sorted_indices = np.argsort(-similarities)
    similar_indices = sorted_indices[sorted_indices != input_index][:n]
    
    similar_songs = data.iloc[similar_indices][['track_id', 'artists', 'album_name', 'track_name']].copy()
    similar_songs['similarity'] = similarities[similar_indices]
    
    query_song = data.iloc[input_index][['track_id', 'artists', 'album_name', 'track_name']].copy()
    query_song = pd.DataFrame([query_song])
    query_song['similarity'] = 1.0
    result = pd.concat([query_song, similar_songs]).reset_index(drop=True)
    
    return result, input_index, similar_indices

# Function for collaborative filtering recommendations
def recommend_collaborative_songs(user_id, track_id, n=5):
    # Placeholder: Implement your collaborative filtering logic here.
    # For example, find songs that similar users liked
    # This part of the code will depend on how your user_data is structured
    # For simplicity, let's assume we return random songs
    recommended = data.sample(n=n)
    recommended['similarity'] = np.random.rand(n)
    return recommended[['track_id', 'artists', 'album_name', 'track_name', 'similarity']]

# Function to set a CJK-compatible font
def set_cjk_font():
    cjk_fonts = ['SimHei', 'Microsoft YaHei', 'MS Gothic', 'Arial Unicode MS', 'Malgun Gothic', 'Noto Sans CJK']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in cjk_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"Using font: {font}")
            return
    print("No CJK-compatible font found. Some characters may not render correctly.")

# Visualize the query and similar songs using PCA in 3D
def visualize_similar_songs(track_name, user_id=None, n=5):
    result, input_index, similar_indices = recommend_similar_songs(track_name, n)
    
    if isinstance(result, str):
        print(result)
        return
    
    collaborative_result = recommend_collaborative_songs(user_id, result['track_id'].iloc[0], n)
    
    # Combine results (content-based and collaborative)
    combined_results = pd.concat([result, collaborative_result]).drop_duplicates().reset_index(drop=True)
    
    pca = PCA(n_components=3)
    indices = combined_results.index
    song_embeddings = embeddings[indices]
    song_embeddings_3d = pca.fit_transform(song_embeddings)
    
    set_cjk_font()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(song_embeddings_3d[1:, 0], song_embeddings_3d[1:, 1], song_embeddings_3d[1:, 2], label='Similar Songs', color='blue')
    ax.scatter(song_embeddings_3d[0, 0], song_embeddings_3d[0, 1], song_embeddings_3d[0, 2], label='Query Song', color='red')
    
    for i, txt in enumerate(combined_results['track_name']):
        ax.text(song_embeddings_3d[i, 0], song_embeddings_3d[i, 1], song_embeddings_3d[i, 2], txt, fontsize=9)
    
    ax.legend()
    ax.set_title(f"Hybrid 3D Visualization of '{track_name}' and Similar Songs")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.tight_layout()
    
    plt.savefig("hybrid_songs_3d_visualization.png")
    plt.show()
    
    print("3D hybrid visualization saved as 'hybrid_songs_3d_visualization.png'")
    
    print("\nSimilar Songs:")
    for i in range(len(combined_results)):
        track_id = combined_results['track_id'].iloc[i]
        track = combined_results['track_name'].iloc[i]
        artist = combined_results['artists'].iloc[i]
        album = combined_results['album_name'].iloc[i]
        similarity = combined_results['similarity'].iloc[i]
        print(f"{i+1}. ID: {track_id} | {track} by {artist} (Album: {album}) - Similarity: {similarity:.4f}")

    return combined_results

# Example usage
if __name__ == "__main__":
    print("Sample track names:")
    print(data['track_name'].unique()[:10])
    
    track_name = input("Enter a track name: ")
    user_id = input("Enter your user ID (optional): ")  # Placeholder for user ID
    similar_songs = visualize_similar_songs(track_name, user_id=user_id, n=5)
    if similar_songs is not None:
        print(similar_songs)
