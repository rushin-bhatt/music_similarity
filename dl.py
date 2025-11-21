import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class SongSimilarityNN(nn.Module):
    def __init__(self):
        super(SongSimilarityNN, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 12)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

file_path = 'dataset.csv'
data = pd.read_csv(file_path)
feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])
X = torch.tensor(data[feature_columns].values, dtype=torch.float32)

model = SongSimilarityNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    embeddings = model(X)
    loss = criterion(embeddings, X)
    loss.backward()
    optimizer.step()
    if loss.item() < 1e-4: 
        break

torch.save(model.state_dict(), 'song_similarity_model.pth')
song_embeddings = embeddings.detach().numpy()
pickle.dump({'embeddings': song_embeddings, 'data': data}, open('song_embeddings.pkl', 'wb'))

model.load_state_dict(torch.load('song_similarity_model.pth', weights_only=True))
data_embeddings = pickle.load(open('song_embeddings.pkl', 'rb'))
data = data_embeddings['data']
song_embeddings = data_embeddings['embeddings']

def visualize_songs(track_name, track_id, n):
    song_index = data[(data['track_name'] == track_name) & (data['track_id'] == track_id)].index[0]
    song_embedding = song_embeddings[song_index].reshape(1, -1)
    
    similarities = np.linalg.norm(song_embeddings - song_embedding, axis=1)
    similar_indices = np.argsort(similarities)[:n + 1]
    similar_indices = similar_indices[1:]
    
    similar_songs = data.iloc[similar_indices]
    subset_embeddings = np.vstack((song_embedding, song_embeddings[similar_indices]))
    
    # Adjust perplexity to be less than the number of samples
    perplexity = min(30, len(subset_embeddings) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, random_state=0)
    embeddings_3d = tsne.fit_transform(subset_embeddings)
    
    print(f"Top {n} songs similar to '{track_name}':")
    for i, (name, score) in enumerate(zip(similar_songs['track_name'], similarities[similar_indices])):
        print(f"{i + 1}. {name} - Similarity Score: {score:.4f}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2], color='red', s=100, label='Query Song')
    ax.scatter(embeddings_3d[1:, 0], embeddings_3d[1:, 1], embeddings_3d[1:, 2], color='blue', s=50, label='Similar Songs')
    
    for i, txt in enumerate(similar_songs['track_name']):
        ax.text(embeddings_3d[i + 1, 0], embeddings_3d[i + 1, 1], embeddings_3d[i + 1, 2], txt, fontsize=9)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title(f"3D t-SNE Visualization of '{track_name}' and Similar Songs")
    ax.legend()
    
    os.makedirs('saved', exist_ok=True)
    plt.savefig('saved/song_similarity_3d_plot.png')
    plt.show()

track_name = input("Enter a track name: ")
track_id = input("Enter the track ID: ")
n = int(input("Enter the number of similar songs to retrieve: "))
visualize_songs(track_name, track_id, n)
