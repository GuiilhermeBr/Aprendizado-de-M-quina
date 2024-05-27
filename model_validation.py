from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Define 4 variaveis e adiciona a importação com argumentos, sendo eles, respectivamente
# Os dados e data e target da iris, X e y
# Percentual de dados para teste de 30% pra fazer teste e 70% para treino, 
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
# Usando argumentos de teste 
print(knn.score(X_test, y_test))