# Author: Aric Hagberg (hagberg@lanl.gov)
import matplotlib.pyplot as plt
import networkx as nx
import json

G = nx.Graph()

# nTeams = 30

data = json.load(open('weightsForPicture.json'))
# data = json.load(open('weightsForPictureDoubled.json'))
matrix = data['matrix']
array = data['array']

nTeamsToUse = 5

matrix = matrix[:nTeamsToUse] + matrix[30:30 + nTeamsToUse]


def keepOnlyMax(row):
    themax = max(row)
    for i in range(len(row)):
        if row[i] != themax:
            row[i] = 0


def useMax(matrix):
    for row in matrix:
        keepOnlyMax(row)


useMax(matrix)

numInputs = len(matrix)
innerDim = len(array)

print(numInputs, innerDim)

# exit()
dictionaries = json.load(open('data/dictionary.json'))

# teamLabels = {1: 'houston (away) ', 2: 'heat (away) ', 31: 'houston (home) ', 32: 'heat (home) '}
teamLabels = {}

for i in range(nTeamsToUse):
    teamLabels[i] = dictionaries['m_i_team'][i] + ' (away) '

for i in range(nTeamsToUse):
    teamLabels[i + nTeamsToUse] = dictionaries['m_i_team'][i] + ' (home) '

for i in range(numInputs):
    for k in range(innerDim):
        G.add_edge(i, k + numInputs, weight=matrix[i][k])

for k in range(innerDim):
    G.add_edge(k + numInputs, numInputs + innerDim, weight=array[k])

# for node in range(90):
#     G.add_edge(node, node + 1, weight=0.1 * node)

# G.add_edge(91, node + 1, weight=0.1 * node)

# for node in range(60):
#     G.add_edge(node, node + 1, weight=0.1 * node)

pos2 = {}

spaceBetweenInputNodes = 60.0 / numInputs

# input layer
for node in range(numInputs):
    pos2[node] = (1, 60 - node * spaceBetweenInputNodes)

spaceBetweenHiddenLayerNodes = 60.0 / innerDim

# hidden layer
for node in range(innerDim):
    # pos2[node + 60] = (7, 52.5 - node * 1.5)
    pos2[node + numInputs] = (4, 60 - node * spaceBetweenHiddenLayerNodes)

# output layer
pos2[numInputs + innerDim] = (5, 30)

# left corner node for alignment wtf
pos2[numInputs + innerDim + 1] = (0, 0)
pos2[numInputs + innerDim + 2] = (1, 0)
G.add_edge(numInputs + innerDim + 1, numInputs + innerDim + 2, weight=0)

# nx.draw_networkx_nodes(G, pos2)

# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
#
# pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos2, nodelist=range(numInputs + innerDim + 1), node_size=10)
nx.draw_networkx_nodes(G, pos2, nodelist=[numInputs + innerDim + 1, numInputs + innerDim + 2], node_size=0)
#
#
# pos2 = {0: (0, 0),
#        1: (1, 0),
#        2: (0, 1),
#        3: (1, 1),
#        4: (0.5, 2.0)}

# edges
# nx.draw_networkx_edges(G, pos, width='weight')

for (u, v, d) in G.edges(data=True):
    nx.draw_networkx_edges(G, pos2, edgelist=[(u, v)], width=d['weight'], label=d['weight'])

# nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')

# labels
# bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
bbox_props = dict(boxstyle="square,pad=0.1")

nx.draw_networkx_labels(G, pos2, horizontalalignment='right', labels=teamLabels, font_size=10, font_family='sans-serif')

plt.axis('off')
plt.show()
