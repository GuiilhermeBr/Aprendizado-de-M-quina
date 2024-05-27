from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Muda o estilo dos plots
plt.style.use('ggplot')

# Guarda na váriavel iris os datasets iris
iris = datasets.load_iris()

# Mostra todas as linhas e colunas da iris
# print(iris)

# Mostra as chaves da variável iris
# print(iris.keys())

# Mostra todas as data da variável iris
# print(iris['data'])

# Mostra todas as targets da variavel iris, em que cada número é uma classe da flor
# print(iris['target'])

# Guarda a data e target em variáveis
X = iris.data
y = iris.target

# Guarda na variável df a dataframe de X na coluna iris usando a chave featura_names
df = pd.DataFrame(X, columns=iris.feature_names)

# Mostra as 5 primeiras informações do dataframe, no parenteses pode selecionar quantas mostrar
# print(df.head())

# Mostra todos os conjuntos dos dados em um gráfico de dispersão com scatter_matrix
# Usando como argumento c para fazer a separação e jogar os dados do target separando
# Por categoria os dados por cor
# Os outros argumentos muda o tamanho
pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='.')

# Mostrar o gráfico na tela comparando os dados da flor
# Cada bolinha no gráfico representa uma classe, a partir de um ponto central, as bolinhas
# são colocadas para definir sua classe com base nas suas coordenadas/tamanho das pétalas e sétalas
# plt.show()

# Aumenta os vizinhos de 5, padrão, para 6, que são os dados que serão utilizados
knn = KNeighborsClassifier(n_neighbors=6)

# Ajusta o modelo passando os dados que quer aprender/ajustar e os dados que quer prever como argumento
# Ajusta os dados para colocar no plano
# A data são as informações que vão ser utilizadas para colocar no gráfico e o target o que o programa quer encontrar
knn.fit(iris['data'], iris['target'])

# Mostra 150 linhas e e 4 colunas são as observações das pétalas
# print(iris['data'].shape)

# Mostra só 150 linhas sem estar organizado em colunas são observações das pétalas
# print(iris['target'].shape)

# Uma lista com lista, passando os 4 valores dessas 4 colunas anteriormente vistas no 'data'
X_new = [[0.1, 3.5, 9.4, 6.2]]

# O método predict prevê através da ajustagem de dados na variável knn usando os dados do
# X_new para prever através dos valores recebidos
prediction = knn.predict(X_new)

# Apresenta na tela a previsão realizada
# print('Prediction {}'.format(prediction))

# COMEÇO DA ATIVIDADE - VALIDAÇÃO PARA VERIFICAR A PRECISÃO DA MÁQUINA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=7, stratify=y
)

# Define quantos grupos serão utilizados
knn = KNeighborsClassifier(n_neighbors=6)

# Ajusta os dados de treino
knn.fit(X_train, y_train)

# Tenta prever os dados de teste
y_pred = knn.predict(X_test)

# Apresenta na tela a precisão da máquina ao utilizar os dados, no caso, possui 90% de chance de acerto
# A cada 100 tentativas, a máquina erra 10
print(knn.score(X_test, y_test))
