import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('../datasets/spotify/Spotify Most Streamed Songs.csv', delimiter=',')

# Select numeric features for analysis (skip categorical features)
numerical_features = ['released_year', 'released_month', 'released_day', 'in_spotify_playlists',
                      'in_spotify_charts', 'streams', 'in_apple_playlists', 'in_apple_charts',
                      'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                      'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_features])

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
pca_result = pca.fit_transform(scaled_data)

# Visualize the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='Data Points')
plt.title('PCA of Music Dataset')

# Set axes to show the principal components with the largest contribution
component_names = np.array(numerical_features)[np.argmax(abs(pca.components_), axis=1)]

plt.xlabel(f'Principal Component 1 ({component_names[0]})')
plt.ylabel(f'Principal Component 2 ({component_names[1]})')

plt.legend()
plt.show()

# Print explained variance
print("Explained variance ratio: ", pca.explained_variance_ratio_)

# Print principal components
print("Principal Components:\n", pca.components_)

# Map components to original features
print("Corresponding features:\n", numerical_features)
