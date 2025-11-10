# -*- coding: utf-8 -*-
# Auto-generated from Regla_de_asociacion.ipynb.
# Source: algorithms/Recomendadores

# %% In[1]
# Librerías
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

plt.style.use("ggplot")
# warning
import warnings
warnings.filterwarnings('ignore')

# %% In[2]
# Datos
# ==============================================================================
url = (
    "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-con-R/"
    "master/datos/datos_groceries.csv"
)
datos = pd.read_csv(url)
datos.shape

# %% In[3]
# Ítems de la transacción 14
# ==============================================================================
datos.query("id_compra==14")

# %% In[4]
# Total de transacciones e ítems (productos)
# ==============================================================================
print(f"Número total de transacciones: {datos['id_compra'].nunique()}")
print(f"Número total de ítems (productos): {datos['item'].nunique()}")

# %% In[5]
# Ítems (productos) agrupados por transacción (compra)
# ==============================================================================
datos.groupby('id_compra')['item'].apply(list)

# %% In[6]
# Distribución del número de ítems por transacción
# ==============================================================================
display(datos.groupby('id_compra')['item'].size().describe(percentiles=[.25, .5, .75, .9]))

fig, ax = plt.subplots(figsize=(7, 3))
datos.groupby('id_compra')['item'].size().plot.hist(ax=ax)
ax.set_title('Distribución del tamaño de las transacciones');
ax.set_xlabel('Número de ítems');

# %% In[7]
# Codificar las transacciones en forma de matriz binaria
# ==============================================================================
# Crear una lista de listas que contenga los ítems de cada transacción
transacciones = datos.groupby('id_compra')['item'].apply(list).to_list()

# Entrenar el objeto TransactionEncoder y transformar los datos
encoder = TransactionEncoder()
transacciones_encoded = encoder.fit(transacciones).transform(transacciones)
transacciones_encoded = pd.DataFrame(transacciones_encoded, columns=encoder.columns_)
transacciones_encoded.head(3)

# %% In[8]
# Porcentaje de transacciones en las que aparece cada producto (top 5)
# ==============================================================================
transacciones_encoded.mean(axis = 0).sort_values(ascending = False).head(5)

# %% In[9]
# Distribución del sopoerte de los items
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 2.5))
transacciones_encoded.mean(axis = 0).plot.hist(ax = ax)
ax.set_xlim(0, )
ax.set_title('Distribución del soporte de los ítems');

# %% In[10]
# Codificar las transacciones en forma de matriz binaria
# ==============================================================================
# Crear una lista de listas que contenga los artículos comprados en cada transacción
transacciones = datos.groupby('id_compra')['item'].apply(list).to_list()

# Entrenar el objeto TransactionEncoder y transformar los datos
encoder = TransactionEncoder()
transacciones_encoded = encoder.fit(transacciones).transform(transacciones)
transacciones_encoded = pd.DataFrame(transacciones_encoded, columns=encoder.columns_)
transacciones_encoded.head(3)

# %% In[11]
# Identificación de itemsets frecuentes
# ==============================================================================
soporte = 30 / transacciones_encoded.shape[0]
print(f"Soporte mínimo: {soporte}")

itemsets = apriori(transacciones_encoded, min_support=soporte, use_colnames=True)
itemsets.sort_values(by='support', ascending=False)

# %% In[12]
# Top 10 itemsets con mayor soporte
# ==============================================================================
itemsets.sort_values(by='support', ascending=False).head(10)

# %% In[13]
# Top 10 itemsets con al menos 2 items, ordenados por soporte
# ==============================================================================
itemsets['n_items'] = itemsets['itemsets'].apply(lambda x: len(x))
itemsets.query('n_items >= 2').sort_values('support', ascending=False).head(10)

# %% In[14]
# Itemsets frecuentes que contienen el ítem newspapers
# ==============================================================================
mask = itemsets['itemsets'].map(lambda x: 'newspapers' in x)
itemsets.loc[mask].sort_values(by='support', ascending=False)

# %% In[15]
# Itemsets frecuentes que contienen los items, al menos, newspapers y whole milk
# ==============================================================================
items = {'newspapers', 'whole milk'}
mask = itemsets['itemsets'].map(lambda x: x.issuperset(items))
itemsets.loc[mask].sort_values(by='support', ascending=False).reset_index()

# %% In[16]
# Identificación subsets
# ==============================================================================
itemset_a = {'other vegetables', 'whole milk', 'newspapers'}
itemset_b = {'other vegetables', 'whole milk', 'newspapers'}

print(itemset_a.issubset(itemset_b))
print(itemset_b.issuperset(itemset_a))

# %% In[17]
# Identificación de itemsets frecuentes
# ==============================================================================
soporte = 30 / transacciones_encoded.shape[0]
itemsets_frecuentes = apriori(transacciones_encoded, min_support=soporte, use_colnames=True)

# Crear reglas de asociación (confianza mínima del 70% para que una regla sea selecionada)
# ==============================================================================
confianza = 0.7
reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=confianza)

print(f"Número de reglas generadas: {len(reglas)}")
print(f"Confianza mínima: {reglas['confidence'].min()}")
print(f"Confianza máxima: {reglas['confidence'].max()}")
reglas.sort_values(by='confidence').head(5)

# %% In[18]
# Seleccionar reglas que tienen "other vegetables" en el consecuente
# ==============================================================================
mask = reglas['consequents'].map(lambda x: 'other vegetables' in x)
reglas.loc[mask]

# %% In[19]
# Antecedentes de las reglas que tienen "other vegetables" en el consecuente
# ==============================================================================
antecedents = reglas.loc[mask, 'antecedents'].to_list()
set().union(*antecedents)

# %% In[20]
# Crear una columna con los ítems que forman parte del antecedente y el consecuente
# ======================================================================================
reglas['items'] = reglas[['antecedents', 'consequents']].apply(lambda x: set().union(*x), axis=1)
reglas.head(3)

# %% In[21]
# Filtrar reglas
# ==============================================================================
seleccion_items = {'citrus fruit', 'root vegetables', 'tropical fruit',
                   'whole milk', 'other vegetables'}
mask = reglas['items'].map(lambda x: x.issubset(seleccion_items))
reglas.loc[mask]

# %% In[22]
# Identificación de itemsets (2 items) frecuentes
# ==============================================================================
soporte = 30 / transacciones_encoded.shape[0]
itemsets_frecuentes = apriori(
                          transacciones_encoded,
                          min_support  = soporte,
                          use_colnames = True,
                          max_len      = 2
                      )

# %% In[23]
# Calcular lift base
# ==============================================================================
lift = 0
reglas = association_rules(itemsets_frecuentes, metric="lift", min_threshold=lift)

print(f"Número de reglas generadas: {len(reglas)}")
print(f"Lift medio: {reglas['lift'].mean()}")
print(f"Lift mediano: {reglas['lift'].median()}")

# %% In[24]
# Crear reglas de asociación (lift mínimo de 1.6)
# ==============================================================================
lift = 1.6
reglas = association_rules(itemsets_frecuentes, metric="lift", min_threshold=lift)

print(f"Número de reglas generadas: {len(reglas)}")
print(f"Lift mínimo: {reglas['lift'].min()}")
print(f"Lift máximo: {reglas['lift'].max()}")
reglas.sort_values(by='lift').head(3)

# %% In[25]
# Filtrar por confianza mínima del 50%
# ==============================================================================
confianza = 0.5
reglas = reglas[reglas['confidence'] >= confianza]

print(f"Número de reglas generadas: {len(reglas)}")
print(f"Confianza mínima: {reglas['confidence'].min()}")
print(f"Confianza máxima: {reglas['confidence'].max()}")
reglas.sort_values(by='confidence').head(3)

# %% In[26]
# Recomendaciones 1 a 1
# ==============================================================================
antecedente = 'rice'

mask = reglas['antecedents'].map(lambda x: antecedente in x)
reglas.loc[mask]

# %% In[27]
#!pip install pyvis

# %% In[28]
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pyvis.network import Network

# --- 1. Cargar datos ---
datos = pd.read_csv(url)

# --- 2. Transformar en formato de transacciones ---
# Agrupamos los productos por cliente (id_compra)
dataset = datos.groupby('id_compra')['item'].apply(list).tolist()

# --- 3. Codificar datos ---
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# --- 4. Calcular itemsets y reglas ---
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# --- 5. Crear red interactiva ---
# Usa notebook=True para Jupyter/Colab o cdn_resources='in_line' si se muestra vacío
net = Network(height="700px", width="100%", directed=True, notebook=True)

# --- 6. Agregar nodos ---
for _, row in rules.iterrows():
    for antecedent in row['antecedents']:
        net.add_node(antecedent, label=antecedent, color="#FFD966", title=f"Soporte: {round(row['support'],2)}")
    for consequent in row['consequents']:
        net.add_node(consequent, label=consequent, color="#F4B183", title=f"Soporte: {round(row['support'],2)}")

# --- 7. Agregar aristas (reglas) ---
for _, row in rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            net.add_edge(
                antecedent,
                consequent,
                value=row['lift'],  # grosor de la línea
                title=f"Soporte: {round(row['support'],2)}<br>"
                      f"Confianza: {round(row['confidence'],2)}<br>"
                      f"Lift: {round(row['lift'],2)}"
            )

# --- 8. Configurar opciones visuales ---
net.set_options('''
var options = {
  "nodes": {
    "shape": "dot",
    "scaling": { "min": 10, "max": 50 },
    "font": { "size": 16 }
  },
  "edges": {
    "arrows": { "to": { "enabled": true } },
    "color": { "color": "gray", "highlight": "red" },
    "smooth": false
  },
  "physics": {
    "forceAtlas2Based": { "gravitationalConstant": -60, "centralGravity": 0.01 },
    "solver": "forceAtlas2Based"
  }
}
''')

# --- 9. Exportar gráfico interactivo ---
net.show("reglas_groceries_interactivo.html")
