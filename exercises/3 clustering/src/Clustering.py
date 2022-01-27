# %% The first cell
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %% opening excel
df = pd.read_excel('../barrettII_eyes_clustering.xlsx')

print(df.head())
print(df.describe())
# %%
# N_CLUSTERS = 5
COLUMN_VALUES = ['AL', 'ACD', 'WTW', 'K1', 'K2']
MAX_CLUSTERS_TEST = 10

# %% To plot the elbow plot to check what is best

k1 = []
inertia_s1 = []

for i in range(2,21):
    k1.append(i)
    kmeans1 = KMeans(n_clusters=i).fit(df[COLUMN_VALUES])
    inertia_s1.append(kmeans1.inertia_)

# plot
plt.figure(figsize=(15,5))
plt.plot(k1,inertia_s1)
plt.title('Inércia vs. K')
plt.xlabel('Valor de K')
plt.ylabel('Inércia')
    
# %% To use the best K

MAX_CLUSTERS = 4

kmeans = KMeans(n_clusters=MAX_CLUSTERS, random_state=123, max_iter=100).fit(df[COLUMN_VALUES])

df_features = df.copy()
df_features['cluster_ids'] = kmeans.labels_
df_features.head()

# %% This is to run the porfiling on the centroids

kmeans_centroids = kmeans.cluster_centers_

print(kmeans_centroids)

print('')

centroids_df = pd.DataFrame()
for i, column in enumerate(COLUMN_VALUES):
    column_of_centroids = []
    for j in range(MAX_CLUSTERS):
        column_of_centroids.append(kmeans_centroids[j][i])
    
    centroids_df[column] = column_of_centroids

centroids_df.head().T

# %%

correct_df = df_features[df_features['Correto'] == 'S'].groupby('cluster_ids')
correct_df = correct_df.count()[['Correto']].T.rename({'Correto': 'Correto Count'})

result_df = pd.concat([centroids_df.T, correct_df])
result_df
# %%
result_df.to_csv('../clustering_olhos.csv')