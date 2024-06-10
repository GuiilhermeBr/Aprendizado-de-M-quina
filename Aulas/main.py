from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Model Validation

plt.style.use('ggplot')
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Define 4 variaveis e adiciona a importação com argumentos, sendo eles, respectivamente
# Os dados e data e target da iris, X e y
# Percentual de dados para teste de 30% pra fazer teste e 70% para treino
# Definindo sua precisão
# random state fará com que a máquina consiga gerar números aleatórios pois ela precisa de uma base(21),
# E stratify para colocar os mesmos números de classes para fazer o teste nas mesmas quantidades
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y
)

# Define a variável knn com 8 vizinhos, aumentando a quantidade de dados que serão utilizados
knn = KNeighborsClassifier(n_neighbors=8)

# Comando fit ajusta os dados utilizando como método as variáveis de teste, 
# primeiro as features e depois o target 
knn.fit(X_train, y_train)

# Faz a previsão para tentar verificar quais dados fazem parte de fato desses conjuntos de dados
y_pred = knn.predict(X_test)
# print(f"Previsões do conjunto de teste:\n {y_pred}")

# Comparar os dados de y_pred (o que o algoritmo previu) com o y_test (conjunto de dados reais)
# Usando argumentos de teste e mostrando sua chance de acerto com base na quantidade de vizinhos
print(knn.score(X_test, y_test))





# Main

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
print('Prediction {}'.format(prediction))




# Regression

import pandas as pd # Análise de dados e cria tabelas e manipular dados
import numpy as np # Análise numérica
import matplotlib.pyplot as plt # Vizualização gráfica
from sklearn import linear_model #

# Valores que verificam, com base em diversos critério, o valor de residências
boston_csv = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# Variável le o arquivo csv
boston = pd.read_csv(boston_csv)

# Vizualização do cabeçalho do dataframe
# print(boston.head())

# Criar os vetores de recursos(feature) e alvo(target)
# Vai deletar a coluna mdev, a média dos preços
X = boston.drop("medv", axis=1).values

# Vai acessar apenas a coluna medv pegando os valores
y = boston['medv'].values

# Prevendo o preço a partir de um único recurso
# Apenas pela quantidade de quartos que ela tem
# Apenas os quartos da variável X, sendo elas todas as linhas na 5 coluna
X_rooms = X[:, 5]

# print(X)
# print(type(X_rooms))
# print(y)
# print(type(y))

# Reshape faz a mudança de lista para coluna
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)

# print(X)
# print(y)

# Valor medio vs n de quarto
# Descobre quais dados colocar na dispersão
# Label é a legenda, no caso, do eixo x e y
# plt.scatter(X_rooms, y)
# plt.ylabel("Valor da casa /1000 ($)")
# plt.xlabel("Número de quartos")
# plt.show()

# Utiliza o método fit para ajustar os dados com base nos quartos e o preço para fazer o treinamento
reg = linear_model.LinearRegression()
reg.fit(X_rooms, y)

# Tirar uma média daquele monte de bolinha e criar uma linha que as represente
# Pega o maior e menor valor dentro da lista de quartos
# O reshape -1 para pegar ao contrário, primeiro a última
# Mas como só há uma lista, então ele só pegará a utlima
# E o 1 é para usar no eixo vertical
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

# Em cima do gráfico que produziu com o scatter colocará a linha com o plot
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()