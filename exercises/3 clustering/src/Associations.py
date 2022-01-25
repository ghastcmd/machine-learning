# %% The first cell
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% opening excel
df = pd.read_excel('../barrettII_eyes_clustering.xlsx')

print(df.head())
print(df.describe())
# %%
# N_CLUSTERS = 5
COLUMN_VALUES = ['AL', 'ACD', 'WTW', 'K1', 'K2']
MAX_CLUSTERS_TEST = 10
# %% Normalizing the scale of values

scaler = MinMaxScaler().fit(df[COLUMN_VALUES])
scaled_df = df.copy()
scaled_df[COLUMN_VALUES] = scaler.transform(scaled_df[COLUMN_VALUES])

# %% To plot the elbow plot to check what is best

k1 = []
inertia_s1 = []

for i in tqdm(range(2,21)):
    k1.append(i)
    kmeans1 = KMeans(n_clusters=i).fit(scaled_df[COLUMN_VALUES])
    inertia_s1.append(kmeans1.inertia_)

# plot
plt.figure(figsize=(15,5))
plt.plot(k1,inertia_s1,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Inertia (SSE) vs. K Value')
plt.xlabel('K')
plt.ylabel('Inertia score (SSE)')
    
# %% To use the best K

kmeans = KMeans(n_clusters=5, random_state=123, max_iter=100).fit(scaled_df[COLUMN_VALUES])

df_features = df.copy()
df_features['cluster_ids'] = kmeans.labels_
df_features.head()

# %% Analysing the data
df_profile_overall = df_features.describe().drop('ID', axis=1).T
df_profile_overall['Overall Dataset'] = df_profile_overall[['mean']]
df_profile_overall = df_profile_overall[['Overall Dataset']]

df_cluster_summary = df_features.groupby('cluster_ids').describe().drop('ID', axis=1).T.reset_index()
df_cluster_summary = df_cluster_summary.rename(columns={'level_0': 'column', 'level_1': 'metric'})

df_cluster_summary = df_cluster_summary[df_cluster_summary['metric'] == 'mean']
df_cluster_summary = df_cluster_summary.set_index('column')

df_profile = df_cluster_summary.join(df_profile_overall)

df_profile
# %%
prof_w_count = df_profile.copy()

count_df = df_features[df_features['Correto'] == 'S'].groupby('cluster_ids').count()
count_df = count_df[['Correto']].rename(columns={'Correto': 'Correto Count'}).T

total_count_df = df_features.groupby('cluster_ids').count()
total_count_df = total_count_df[['K2']].rename(columns={'K2': 'Total Count'}).T

total_count_df['Overall Dataset'] = total_count_df.sum(axis=1)
total_count_df['metric'] = 'count'

count_df['Overall Dataset'] = count_df.sum(axis=1)
count_df['metric'] = 'count'

prof_w_count = pd.concat([prof_w_count, count_df, total_count_df])
prof_w_count


# %%
prof_w_count
prof_w_count.to_csv('../clustering_olhos.csv')

# Reference link:
# https://medium.com/analytics-vidhya/clustering-and-profiling-customers-using-k-means-9afa4277427