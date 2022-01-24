# %% The first cell
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# %% opening excel
df = pd.read_excel('../barrettII_eyes_clustering.xlsx')

df.head()
df.describe()

# %%
# N_CLUSTERS = 5
COLUMN_VALUES = ['AL', 'ACD', 'WTW', 'K1', 'K2']
MAX_CLUSTERS_TEST = 10
# %%
for N_CLUSTERS in range(1, MAX_CLUSTERS_TEST):
    kmeans = KMeans(n_clusters=N_CLUSTERS).fit(df[COLUMN_VALUES])
    another_df = df
    
    another_df['Cluster'] = kmeans.labels_
    
    test_list = []
    for i in range(N_CLUSTERS):
        test_list.append(another_df[another_df.Cluster == i].Correto == 'S')

    OVERALL_ACC = 0
    for test in test_list:
        iter_count = test.count()
        iter_sum = test.sum()
        iter_acc = (iter_sum / iter_count) * 100
        OVERALL_ACC += iter_acc
        
        if 0:
            print('count', iter_count)
            print('correct', iter_sum)
            print(f'accuracy: {iter_acc:.2f}\n')
        
    print(OVERALL_ACC / N_CLUSTERS)
    
# %%
