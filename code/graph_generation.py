import networkx as nx
import pylab as plt
import json, random

def generate_graph(nodes, threshold):
    er = nx.gnp_random_graph(nodes, threshold)
    # er = nx.erdos_renyi_graph(nodes, threshold)

    nx.draw(er, with_labels = True)
    plt.savefig('./Graphs/imgs/graph6_30.png')

    data = nx.readwrite.json_graph.node_link_data(er, {'link': 'edges', 'source': 'from', 'target': 'to'})
    for i in data['edges']:
        i['weights'] = random.randint(1,5)
    graph = json.dumps(data, indent=4)

    with open('./Graphs/graph6_30.json','w') as f:
        f.write(graph)

    #print(graph)

if __name__ == "__main__":
    generate_graph(30, 0.15)    