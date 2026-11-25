# CÓDIGO PARA APLICAR O ALGORITMO KMEANS NOS DADOS PADRONIZADOS 
# TUTORIAL: Tisfoulla Academy - Link: https://www.youtube.com/watch?v=2a2JVFS6NcY

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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

# dados que serão utilizados
dados_seade = dados_padronizados[variaveis] 

# aplicação do kmeans com k = 5
num_clusters = 5 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(dados_seade)
print(labels)

# TÉCNICA TSNE
# reduzindo dimensões com t-sne 
tsne = TSNE(n_components=2, random_state=42)
transformacao_tsne = tsne.fit_transform(dados_seade)

# plotando os resultados do t-sne
plt.figure(figsize=(10,6))

# pontos coloridos
scatter = plt.scatter(
    transformacao_tsne[:, 0],
    transformacao_tsne[:, 1],
    c=labels,
    cmap="Set1",
    alpha=0.8,
    edgecolors="k",
    linewidth=0.5,
)

# nome dos rótulos
plt.title('Visualização T-SNE da clusterização dos municípios (k=5)')
plt.xlabel('Dimensão 1 do t-SNE')
plt.ylabel('Dimensão 2 do t-SNE')

# adiciona legenda e barra de cores para identificar os clusters
plt.colorbar(scatter, label="Cluster")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# adicionando a coluna CLUSTER na base padronizada
dados_padronizados["CLUSTER"] = labels
dados_padronizados.to_excel("DADOS PADRONIZADOS SEADE.xlsx", index=False)

# carregando a base bruta original e adicionando a coluna CLUSTER
dados_brutos = pd.read_excel("DADOS SEADE.xlsx")
dados_brutos["CLUSTER"] = labels
dados_brutos.to_excel("DADOS SEADE.xlsx", index=False)

# resumo descritivo da contagem por cluster
print("\nQuantidade de municípios por cluster")
print(dados_brutos["CLUSTER"].value_counts().sort_index())