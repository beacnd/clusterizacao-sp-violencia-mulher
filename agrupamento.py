# CÓDIGO PARA: CALCULAR K IDEAL, EXTRAIR MUNICÍPIOS POR CLUSTER E PLOTAR DENDOGRAMA (AGRUPAMENTO HIERÁRQUICO)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Carregando a base de dados padronizada
dados = pd.read_excel("DADOS PADRONIZADOS SEADE.xlsx")

variaveis = [
    "POPULAÇÃO",
    "PIB",
    "PIB PER CAPITA",
    "EST. EMP FORMAL",
    "TAXA REPROVAÇÃO",
    "TAXA ABANDONO",
    "ENFERMEIROS (mil/hab)",
]

X = dados[variaveis].copy().values

# Calculando o número de k clusters variando de 2 a 10 usando Agrupamento Hierárquico
silhueta_score = []
for i in range(2, 11):
  modelo = AgglomerativeClustering(n_clusters=i, linkage="ward")
  labels = modelo.fit_predict(X)
  score = silhouette_score(X, labels)
  silhueta_score.append(score)
  print(f"Para k = {i}, o Silhouette Score é: {score:.4f}")

# Plotando o gráfico do método da silhueta
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhueta_score, marker="o", color="purple")
plt.title(
    "Método da Silhueta para a definição dos k clusters (Hierárquico Ward)"
)
plt.xlabel("Número de clusters (k)")
plt.ylabel("Coeficiente do método da Silhueta")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Definindo k = 5 
k_escolhido = 5
modelo_final = AgglomerativeClustering(n_clusters=k_escolhido, linkage="ward")
rotulos = modelo_final.fit_predict(X)

# Exibindo o array com os clusters atribuídos por município
print(f"\nArray com os clusters atribuídos por município (k={k_escolhido}):")
print(rotulos)

# Exibindo a quantidade de municípios por cluster
print(f"\nQuantidade de municípios por cluster (k={k_escolhido}):")
unique, counts = np.unique(rotulos, return_counts=True)
for cluster_id, qtd in zip(unique, counts):
  print(f"Cluster {cluster_id}: {qtd} municípios")

# Configuração e exibição do Dendrograma
Z = linkage(X, method="ward")
plt.figure(figsize=(14, 7))
dendrogram(Z, truncate_mode="lastp", p=25)
plt.title(
    f"Agrupamento Hierárquico (k={k_escolhido})"
)
plt.xlabel("Clusters / Municípios agrupados")
plt.ylabel("Distância de ligação (Ward)")
plt.tight_layout()
plt.show()