import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

file_path = 'dataset.csv'
data = pd.read_csv(file_path)

feature_columns = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature'
]

data = data.dropna(subset=feature_columns)

scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

embeddings = data[feature_columns].values

with open('song_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'data': data}, f)

print("Embeddings and data saved to 'song_embeddings.pkl'")
