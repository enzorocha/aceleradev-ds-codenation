#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (KBinsDiscretizer,OneHotEncoder,StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[4]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Eliminando espaços em branco
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[8]:


# Alterando os tipos de algumas colunas
countries['Pop_density'] = countries['Pop_density'].str.replace(',','.').astype('float')
countries['Coastline_ratio'] = countries['Coastline_ratio'].str.replace(',','.').astype('float')
countries['Net_migration'] = countries['Net_migration'].str.replace(',','.').astype('float')
countries['Infant_mortality'] = countries['Infant_mortality'].str.replace(',','.').astype('float')
countries['Literacy'] = countries['Literacy'].str.replace(',','.').astype('float')
countries['Phones_per_1000'] = countries['Phones_per_1000'].str.replace(',','.').astype('float')
countries['Arable'] = countries['Arable'].str.replace(',','.').astype('float')
countries['Crops'] = countries['Crops'].str.replace(',','.').astype('float')
countries['Other'] = countries['Other'].str.replace(',','.').astype('float')
countries['Climate'] = countries['Climate'].str.replace(',','.').astype('float')
countries['Birthrate'] = countries['Birthrate'].str.replace(',','.').astype('float')
countries['Deathrate'] = countries['Deathrate'].str.replace(',','.').astype('float')
countries['Agriculture'] = countries['Agriculture'].str.replace(',','.').astype('float')
countries['Industry'] = countries['Industry'].str.replace(',','.').astype('float')
countries['Service'] = countries['Service'].str.replace(',','.').astype('float')


# In[9]:


# Funcao auxiliar para identificar o intervalo
def get_interval(bin_idx, bin_edges):
  return f"{np.round(bin_edges[bin_idx], 2):.2f} ⊢ {np.round(bin_edges[bin_idx+1], 2):.2f}"


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[10]:


def q1():
    # Lista com valores unicos 
    lista = list(countries['Region'].unique())
    # Ordena a lista
    lista.sort()
    return lista


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[11]:


def q2():
    # Discretizando
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(countries[['Pop_density']])
    density_bins = discretizer.transform(countries[['Pop_density']])
    bin_edges_quantile = discretizer.bin_edges_[0]
    # Identificando os intervalos
    intervals = []
    for i in range(len(bin_edges_quantile)-1):
        # Adicionando ao array apenas a quantidade de paises em cada percentil
        intervals.append(sum(density_bins[:, 0] == i))
    # Retornando o percentil desejado
    return int(intervals[9])


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[12]:


# Define encoder
encoder = OneHotEncoder()
# Transforma / codifica a variavel categorica
region_encoded = encoder.fit_transform(countries[['Region']])
climate_encoded = encoder.fit_transform(countries[['Climate']].astype('str'))
region_elements = region_encoded.shape[1]
climate_elements = climate_encoded.shape[1]
region_elements+climate_elements


# In[13]:


def q3():
    # Define encoder
    encoder = OneHotEncoder()
    # Transforma / codifica a variavel categorica
    region_encoded = encoder.fit_transform(countries[['Region']])
    climate_encoded = encoder.fit_transform(countries[['Climate']].astype('str'))
    region_elements = region_encoded.shape[1]
    climate_elements = climate_encoded.shape[1]
    # Retorna a quantidade de novos atributos
    return region_elements+climate_elements


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[14]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[44]:


def q4():
    # Cria pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standard", StandardScaler())
    ])
    
    # Remove a primeiras duas colunas que sao object
    # permancendo somente os tipos int64 e float64
    columns_fit = np.delete(countries.columns,[0,1])
    
    # Aplicando a pipeline ao dataframe principal
    estimator = num_pipeline.fit(countries[columns_fit])
    
    # Monta dataframe com os dados de teste
    data_test = pd.DataFrame(test_country).T
    data_test.columns = countries.columns
    
    # Transforma
    data_test[columns_fit] = estimator.transform(data_test[columns_fit])
    
    return round(float(data_test['Arable']),3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[18]:


def q5():
    # Dados a serem utilizados
    data_migration = countries['Net_migration'].dropna()
    # Identificando os quartis
    q1 = data_migration.quantile(0.25)
    q3 = data_migration.quantile(0.75)
    iqr = q3 - q1
    # Identificando o intervalo normal
    non_outlier_interval_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    # Identificando os intervalos de outliers (abaixo e acima)
    outliers_abaixo = data_migration[(data_migration < non_outlier_interval_iqr[0])]
    outliers_acima = data_migration[(data_migration > non_outlier_interval_iqr[1])]
    # Retorna as quantidades e a decisao de removor os outliers 
    # apos analise via histograma
    return tuple(
        [
            int(outliers_abaixo.shape[0]),
            int(outliers_acima.shape[0]), 
            bool(False)
        ]
    )


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[19]:


def q6():
    # Dataset do sklearn
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # Vetorizando
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    # Array das palavras a serem analisadas
    words_idx = sorted([count_vectorizer.vocabulary_.get(u"phone")])
    # Dataset com a vertorizacao das palavras escolhidas
    df = pd.DataFrame(newsgroups_counts[:, words_idx].toarray(), columns=np.array(count_vectorizer.get_feature_names())[words_idx])
    return float(df['phone'].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[20]:


def q7():
    # Dataset do sklearn
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # Vetorizando
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    # Array das palavras a serem analisadas
    words_idx = sorted([count_vectorizer.vocabulary_.get(u"phone")])
    # Estatistica TF-IDF
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(newsgroups_counts)
    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)
    # Dataset com a estatistica TF-IDF das palavras escolhidas
    df = pd.DataFrame(newsgroups_tfidf[:, words_idx].toarray(), columns=np.array(count_vectorizer.get_feature_names())[words_idx])
    return round(float(df['phone'].sum()),3)


# In[ ]:




