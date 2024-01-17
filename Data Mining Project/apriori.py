from apyori import apriori
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx

store_data =  pd.read_csv('C:\CODING\CSV Files\datasetDMBI.csv',header=None)
print(store_data)

print(store_data.shape)

records = []
for i in range(0,5):
    records.append([str(store_data.values[i,j])  for j in range(0,4)])
    
association_rule = apriori(records,min_support=0.2,min_confidence = 0.6)
association_results = list(association_rule)
print(len(association_results))
print('\n'.join(map(str,association_results)))

# create the graph
G = nx.DiGraph()

# add nodes to the graph
for result in association_results:
    for item in result[0]:
        G.add_node(item)

# add edges to the graph
for result in association_results:
    for item in result[0]:
        for item2 in result[0]:
            if item != item2:
                G.add_edge(item, item2)

# draw the graph
nx.draw(G, with_labels=True)
plt.show()