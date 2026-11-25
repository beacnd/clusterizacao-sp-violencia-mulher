# CÓDIGO PARA ENCONTRAR A QUANTIDADE DE CLUSTERS (K) IDEAL USANDO MÉTODO DA SILHUETA

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# carregando a base de dados padronizada
dados_padronizados = pd.read_excel("DADOS PADRONIZADOS SEADE.xlsx")

# selecionando apenas as variáveis desejáveis
variaveis = [
    "POPULAÇÃO",
    "PIB",
    "PIB PER CAPITA",
    "EST. EMP FORMAL",
    "TAXA REPROVAÇÃO",
    "TAXA ABANDONO",
    "ENFERMEIROS (mil/hab)",
]
X = dados_padronizados[variaveis]

# calculando o número de k clusters variando de 2 a 10
silhueta_score = []
for i in range(2,11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X) # passando a matriz X para o fit_predict
    score = silhouette_score(X, labels) # o silhouette_score precisa dos dados (X) e dos rótulos/clusters atribuídos (labels)
    silhueta_score.append(score)

    print(f"Para k = {i}, o Silhouette Score é: {score:.4f}")

# plotando o gráfico do método da silhueta
plt.figure(figsize=(10, 6))
plt.plot(range(2,11), silhueta_score, marker='o')
plt.title('Método da silhueta para a definição dos k clusters')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Coeficiente do método da Silhueta')
plt.show()