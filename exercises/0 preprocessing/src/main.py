#%%
import pandas as pd

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('../datasets/diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#%%
new_feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 
                'BMI', 'DiabetesPedigreeFunction', 'Age']

new_data = data
del new_data['Insulin']
del new_data['SkinThickness']

new_data = new_data.dropna()
print(new_data)

#%%
X = new_data[new_feature_cols]
y = new_data.Outcome

#%%
from sklearn.neighbors import KNeighborsClassifier


# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('../datasets/diabetes_app.csv')
data_app = data_app[new_feature_cols]
y_pred = neigh.predict(data_app)
# %%
