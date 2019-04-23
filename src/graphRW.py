# Author: Aric Hagberg (hagberg@lanl.gov)
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_edge(1, 2, weight=7, color='red')
# nx.write_edgelist(G, 'test.edgelist', data=False)
# nx.write_edgelist(G, 'test.edgelist', data=['color'])
# nx.write_edgelist(G, 'test.edgelist', data=['color', 'weight'])


# H = nx.read_edgelist(path="test.edgelist")

# nx.draw(H)
plt.show()
