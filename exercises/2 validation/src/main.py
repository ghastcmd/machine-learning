#%%
import pandas as pd
import numpy as np

data = pd.read_csv('../datasets/abalone_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['length', 'diameter', 'height', 'whole_weight',
                'shucked_weight', 'viscera_weight', 'shell_weight', 'M', 'F', 'I']
#%%
dummy_sex = pd.get_dummies(data['sex'])
data = pd.concat((data, dummy_sex), axis=1)
new_data = data.drop(['sex'], axis=1)

#%%
from sklearn.preprocessing import MinMaxScaler

X = new_data[feature_cols]

# scaler = MinMaxScaler()
# scaler.fit(X)
# scaled = scaler.fit_transform(X)
# X = pd.DataFrame(scaled, columns=X.columns)

X = np.array(X)

y = new_data.type

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

cv = KFold(5)

MLP = MLPClassifier(hidden_layer_sizes=(11, 15), random_state=1, max_iter=400)
KNN = KNeighborsClassifier(n_neighbors=3)
RFC = RandomForestClassifier(n_estimators=50, random_state=1)
LG  = LogisticRegression(random_state=1, max_iter=200)

def print_accuracy(name: str, pred, test):
    value = f'{sum(pred == test) / len(test) * 100:.2f}%'
    print(name, value)

PRED = 0.0
SAVED_MODEL = 0

def store_max_pred(pred, test, model):
    global SAVED_MODEL
    global PRED
    acc = sum(pred == test) / len(test)
    if acc > PRED:
        SAVED_MODEL = model
        PRED = acc
        

for train_index, test_index in cv.split(X):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]
    
    MLP.fit(x_train, y_train)
    KNN.fit(x_train, y_train)
    RFC.fit(x_train, y_train)
    LG.fit(x_train, y_train)
    
    pred = MLP.predict(x_test)
    print_accuracy('MLP Accuracy:', pred, y_test)
    store_max_pred(pred, y_test, MLP)
    
    pred = KNN.predict(x_test)
    print_accuracy('KNN Accuracy:', pred, y_test)
    store_max_pred(pred, y_test, KNN)
    
    pred = RFC.predict(x_test)
    print_accuracy('RFC Accuracy:', pred, y_test)
    store_max_pred(pred, y_test, RFC)
    
    pred = LG.predict(x_test)
    print_accuracy('LG Accuracy:', pred, y_test)
    store_max_pred(pred, y_test, LG)
    
    print('')

print(SAVED_MODEL)

#%%

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('../datasets/abalone_app.csv')

dummy_sex = pd.get_dummies(data_app['sex'])
data_app = pd.concat((data_app, dummy_sex), axis=1)
data_app = data_app.drop(['sex'], axis=1)
data_app = data_app[feature_cols]

# scaled = scaler.fit_transform(data_app)
# data_app = pd.DataFrame(scaled, columns=data_app.columns)

y_pred = SAVED_MODEL.predict(np.array(data_app))

# %%
# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

DEV_KEY = "Dino"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

import requests

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
# %%
