#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

#%%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import numpy as np

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('../datasets/diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome
#%% Preprocessing
new_feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 
                'BMI', 'DiabetesPedigreeFunction', 'Age']

new_data = data
del new_data['Insulin']

skin_thickness_mean = new_data['SkinThickness'].mean()
new_data['SkinThickness'].fillna(value=skin_thickness_mean, inplace=True)

new_data = new_data.dropna()

#%%
print(new_data)

#%%
X = new_data[new_feature_cols]
y = new_data.Outcome

#%%

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('../datasets/diabetes_app.csv')
data_app = data_app[new_feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

DEV_KEY = "Dino"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
# %%
