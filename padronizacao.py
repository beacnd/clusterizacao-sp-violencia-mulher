# CÓDIGO PARA A GERAÇÃO DA PLANILHA COM OS VALORES PADRONIZADOS
# StandardScaler realiza a padronização via Z-score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# importando a base de dados
dados = pd.read_excel("DADOS SEADE.xlsx")

# selecionando as variáveis desejáveis (o que será padronizado) dos identificadores
variaveis = [
    "POPULAÇÃO",
    "PIB",
    "PIB PER CAPITA",
    "EST. EMP FORMAL",
    "TAXA REPROVAÇÃO",
    "TAXA ABANDONO",
    "ENFERMEIROS (mil/hab)"
]

identificacao = dados[["MUNICÍPIOS", "Código IBGE"]]

# ETAPA 1: TRANSFORMAÇÃO LOGARÍTIMICA
# criando e aplicando transformação logarítmica (log1p = log(1 + x)) apenas nas variáveis que possuem assimetria extremas
dados_com_log = dados.copy()
variaveis_log = ["POPULAÇÃO", "PIB", "EST. EMP FORMAL"]

for col in variaveis_log:
    dados_com_log[col] = np.log1p(dados_com_log[col])

# ETAPA 2: PADRONIZAR VIA Z-SCORE APLICANDO STANDARDSCALER
# criando o padronizador, a saída será uma matriz
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(dados_com_log[variaveis])

# criando tabela 
dados_padronizados = pd.DataFrame(
    dados_padronizados,
    columns=variaveis
)

# juntar identificadores com os dados padronizados
base_final = pd.concat(
    [identificacao, dados_padronizados],
    axis=1
)

# gerar nova planilha
base_final.to_excel(
    "DADOS PADRONIZADOS SEADE.xlsx",
    index=False
)