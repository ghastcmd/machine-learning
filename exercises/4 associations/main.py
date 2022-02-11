#%% 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#%%

df = pd.read_csv('Grocery Products Purchase.csv')
df

# %%

all_transactions = []

for _, row in df.iterrows():
    values = list(filter(lambda x: type(x) == str, row.tolist()))
    
    all_transactions.append(values)

#%%

te = TransactionEncoder()
te_ary = te.fit(all_transactions).transform(all_transactions)
new_df = pd.DataFrame(te_ary, columns=te.columns_)

new_df
#%%
frequent_itemsets = apriori(new_df, min_support=0.01, use_colnames=True)
frequent_itemsets

#%%

res = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.45)
res
#%%

res1 = res[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
res1

#%%
res1.sort_values(by=['lift'], ascending=False)
res1
#%%
res1.to_csv('associations.csv')