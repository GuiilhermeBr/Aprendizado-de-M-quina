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