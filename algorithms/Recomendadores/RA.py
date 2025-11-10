# -*- coding: utf-8 -*-
# Auto-generated from RA.ipynb.
# Source: algorithms/Recomendadores

# %% In[1]
# Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# warning
import  warnings
warnings.filterwarnings("ignore")

# %% In[2]
dataset=[
      ["Leche", "Huevos"],
      ["Pan", "Queso"],
      ["Leche", "Pan", "Huevos"],
      ["Pan", "Queso"],
      ["Leche","Queso","Huevos"],
      ["Pan","Queso","Huevos"],
      ["Leche","Pan","Queso","Huevos"]
]

# %% In[3]
# algoritmo apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# %% In[4]
# Convertir los datos a formato canasta
te = TransactionEncoder()
# ajustar
te_ary = te.fit(dataset).transform(dataset)
# convertir a dataframe
df = pd.DataFrame(te_ary,columns=te.columns_)
df

# %% In[5]
# Obtener los conjuntos frecuentes(apriori)
np.random.seed(2025)# Fijar semilla
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets

# %% In[6]
# Generar reglas de asociacion
reglas=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(reglas[["antecedents","consequents","support","confidence","lift"]])

# %% In[7]
#! pip install scikit-surprise
#! pip install "numpy<2"

# %% In[8]
from surprise import Dataset, Reader, SVD, accuracy

# %% In[9]
# Crear valoraciones de peliculas
ranking={
    "usuario":["A","A","A","B","B","B","C","C","C"],
    "pelicula":["pelicula1","pelicula2","pelicula3","pelicula1","pelicula2","pelicula,3","pelicula1","pelicula2","pelicula3"],
    "Valoracion":[5,4,5,1,4,5,4,3,3]}
df=pd.DataFrame(ranking)
df

# %% In[10]
# convertir en formato surprise
reader=Reader(rating_scale=(1,5))
data=Dataset.load_from_df(df[["usuario","pelicula","Valoracion"]],reader=reader)
data

# %% In[11]
# Data de entrenamiento
from surprise.model_selection import train_test_split
trainset,testset=train_test_split(data,test_size=0.25, random_state=2025)

# %% In[12]
# modelo UBCF(Filtrado colaborativo)
from surprise import KNNBasic, accuracy
opciones={
    "name":"cosine",
    "user_based":True
}
# crear modelo
model_UBCF=KNNBasic(sim_options=opciones)

# %% In[13]
# ajustemos
prediciones=model_UBCF.fit(trainset).test(testset)
accuracy.rmse(prediciones)

# %% In[14]
# Predccion de calificar un usuario
pred=model_UBCF.predict("A","pelicula3")
print(pred)

# %% In[15]
# Modelo IBCF(Similitud entre items)
opciones={
    "name":"cosine",
    "user_based":False
}
# crear modelo
model_IBCF=KNNBasic(sim_options=opciones)
# ajustemos
prediciones=model_IBCF.fit(trainset).test(testset)
accuracy.rmse(prediciones)

# %% In[16]
# Predccion de calificar un usuario
pred=model_IBCF.predict("A","pelicula3")
print(pred)
