import pandas as pd # Análise de dados e cria tabelas e manipular dados
import numpy as np # Análise numérica
import matplotlib.pyplot as plt # Vizualização gráfica

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

print(X)
print(type(X_rooms))
print(y)
print(type(y))

# Reshape faz a mudança de lista para coluna
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)

print(X)
print(y)

# Valor medio vs n de quarto
# Descobre quais dados colocar na dispersão
# Label é a legenda, no caso, do eixo x e y
plt.scatter(X_rooms, y)
plt.ylabel("Valor da casa /1000 ($)")
plt.xlabel("Número de quartos")
plt.show()