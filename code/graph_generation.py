import networkx as nx
import pylab as plt
import json, random

def generate_graph(nodes, threshold):
    er = nx.gnp_random_graph(nodes, threshold)
    for (u,v,w) in er.edges(data=True):
        w['weight'] = random.randint(1,5)
    # er = nx.erdos_renyi_graph(nodes, threshold)
    # er = nx.grid_graph(dim=[6,6])

    nx.draw(er, with_labels = True)
    plt.savefig('../graphs/imgs/graph_temp.png')
    #plt.show()
    plt.close()

    data = nx.readwrite.json_graph.node_link_data(er, {'link': 'edges', 'source': 'from', 'target': 'to'})
    # for i in data['edges']:
    #     i['weights'] = random.randint(1,5)
    graph = json.dumps(data, indent=4)

    with open('../graphs/graph_temp.json','w') as f:
        f.write(graph)

    # print(data)
    return er

if __name__ == "__main__":
    generate_graph(20, 0.15)    