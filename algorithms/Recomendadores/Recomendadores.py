# -*- coding: utf-8 -*-
# Auto-generated from Recomendadores.ipynb.
# Source: algorithms/Recomendadores

# %% In[1]
#!pip install scikit-surprise

# %% In[2]
#!pip install "numpy<2"

# %% In[3]
from surprise import Dataset, Reader
import pandas as pd

# Supón que tienes un dataset de calificaciones usuario-item
ratings_dict = {
    "usuario": ["A", "A", "B", "B", "C", "C", "D"],
    "item":    ["pelicula1", "pelicula2", "pelicula2", "pelicula3", "pelicula1", "pelicula3", "pelicula2"],
    "rating":  [5, 4, 5, 3, 4, 2, 4]
}

df = pd.DataFrame(ratings_dict)
df.head()

# %% In[4]

# --- Convertir a formato Surprise ---
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['usuario', 'item', 'rating']], reader)

# %% In[5]
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.25)

# %% In[6]
#Modelo UBCF (User-Based Collaborative Filtering)
from surprise import KNNBasic, accuracy

# Similaridad basada en usuarios
sim_options = {
    'name': 'cosine',   # o 'pearson'
    'user_based': True   # True = UBCF
}

modelo_ubcf = KNNBasic(sim_options=sim_options)
modelo_ubcf.fit(trainset)

# Evaluar
predicciones_ubcf = modelo_ubcf.test(testset)
accuracy.rmse(predicciones_ubcf)

# %% In[7]
# Predecir la calificación del usuario 'A' para 'pelicula3'
pred = modelo_ubcf.predict('A', 'pelicula3')
print(pred)

# %% In[8]
# Modelo IBCF (Item-Based Collaborative Filtering) Similaridad basada en ítems
sim_options = {
    'name': 'cosine',   # o 'pearson'
    'user_based': False  # False = IBCF
}

modelo_ibcf = KNNBasic(sim_options=sim_options)
modelo_ibcf.fit(trainset)

# Evaluar
predicciones_ibcf = modelo_ibcf.test(testset)
accuracy.rmse(predicciones_ibcf)

# %% In[9]
# prediccion
pred = modelo_ibcf.predict('A', 'pelicula3')
print(pred)
