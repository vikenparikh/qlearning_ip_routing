import numpy as np
import networkx as nx
import json, math, random, copy
import matplotlib.pyplot as plt
from graph_generation import *
import pickle

qtables = {}
weights = {}
q_tbl_cnt = 0
src_node = ''
des_node = ''

# parameters
alpha = 0.1 # learning rate
drate = 1.0 # reward discount rate
err = 0.1 # error of approaching extremities
gamma = 0.9 # discount rate
Tmax = 2000 # Temperature max (Exploration limit)
Tmin = 10 # Temperature min (Exploitation limit)
T = {} # Temperature dict
Cnt = {} # Count dict - No of times node is visited

def initialize(s,d,file_name):
    global qtables, q_tbl_cnt, src_node, des_node, weights, T, Cnt, edges, graph

    # loading graph - json file
    with open('../graphs/'+file_name+'.json','r') as f:
        graph = json.load(f)
    
    # edges in graph
    edges = graph['edges']

    # extracting the weights of the edges
    weights = {}
    for i in graph['nodes']:
        l = {}
        for j in graph['edges']:
            if j['from'] == i['id']:
                l[str(j['to'])] = j['weight']
            elif j['to'] == i['id']:
                l[str(j['from'])] = j['weight']

            weights[str(i['id'])] = l

    # intializing source and destination node
    src_node = s
    des_node = d

    # print()
    # # weights
    # print("Weights")
    # print(weights)
    # print()

    # intializing the temperature dict
    for i in graph['nodes']:
        T[str(i['id'])] = Tmax
    
    # intializing the cnt dict
    for i in graph['nodes']:
        Cnt[str(i['id'])] = 0

    # creating a dict for the index of q-tables
    q_tbl_cnt = 0
    qtables = {}
    for i in graph['nodes']:
        l = {}
        for j in graph['edges']:
            if j['from'] == i['id']:
                l[str(j['to'])] = 0.001 * random.uniform(0.1,1) # 0.000001 
                q_tbl_cnt += 1
            elif j['to'] == i['id']:
                l[str(j['from'])] = 0.001 * random.uniform(0.1,1) # 0.000001 
                q_tbl_cnt += 1
            
            qtables[str(i['id'])] = l

    # setting qvalue from destination nodes to 0
    for j in qtables[d].keys():
        qtables[d][j] = 0

    # print()
    # # initialized Q values table
    # print("Initial Q-table")
    # print(qtables)
    # print()

    # initial Parameters
    # print("Initial Parameters")
    # print("Learning rate")
    # print(alpha)
    # print("Reward discount rate")
    # print(drate)
    # print("Error of approaching extremities")
    # print(err)
    # print("Discount rate")
    # print(gamma)
    # print("Temperature max Exploration limit")
    # print(Tmax)
    # print("Temperature min Exploitation limit")
    # print(Tmin)
    # print()

    # states i.e. nodes on the graph
    # l=[]
    # print("States of the graph are:")
    # for i in graph['nodes']:
    #     l.append(i)
    #     print(i)

    # print()
    # # actions from all these states are:
    # print("Actions from all these states are:")
    # k=0
    # for i in qtables.values():
    #     print("State",l[k], "->  actions", list(i))
    #     k+=1

    # print()

# function that returns the best action based on softmax function (Boltzmann)
def softmax(node, T):
    global qtables
    # list of probabilities
    p = []    
    # calculate probability for each action
    # den = sum_a(e^Q(s,a)/T)
    den = 0
    for i in qtables[node].items():
        den += math.exp(i[1] / T[node])
    # calculating the probability 
    for i in qtables[node].items():
        t_p =  (math.exp(i[1] / T[node]) ) / den
        p.append((i[0],t_p))
    # return best action based on probabilities
    mx = max(p, key=lambda item: item[1])
    return mx[0]


# qlearning function
def qlearning(n): # alpha (learning rate)
    global qtables, Cnt, T
    # count - no of iterations
    count_i = 0
    # iterating till convergence
    # number of time it converges
    cnt_converge = 0
    # set converge flag to 0
    converge_flag = 0
    while converge_flag != 1 and count_i < 1000:
        # store the old qtable
        old_qtable = dict()
        old_qtable = copy.deepcopy(qtables)
        # start from the source node
        node = src_node
        Cnt[node] += 1
        while node != des_node:
            # temperature decrement for each node
            if Cnt[node] % n/5 == 0:
                if T[node] - 1000 > 0:
                    T[node] -= 200
                elif T[node] - 190 > 0:
                    T[node] -= 190

            next_node = select_action(node,T)
            Cnt[next_node] += 1
            q_old = qtables[node][next_node]
            q = []
            for i in qtables[next_node].items():
                q.append(i[1])
            
            r = -weights[node][next_node]
            if next_node == des_node:
                r = 10
            w = alpha * (r + (gamma * max(q))-qtables[node][next_node])
            q_new = q_old + w
            qtables[node][next_node] = q_new
            node = next_node
        count_i += 1
        # check if converges
        elements_convergence = 0
        for u in qtables.keys():
            if converge_flag == 1:
                break
            for v in qtables[u].keys():
                if math.fabs(old_qtable[u][v] - qtables[u][v]) < 0.000000000005:
                    elements_convergence += 1
    
        if elements_convergence == q_tbl_cnt:
            cnt_converge += 1
        
        if cnt_converge > 4:
            converge_flag = 1

    return count_i


def select_action(node, T):
    global qtables
    mx_t = softmax(node, T)

    return mx_t

def remove_nodes(a,b,n, alpha):
    global T, graph
    qtables[a][b] = -100
    
    # intializing the temperature dict
    for i in graph['nodes']:
        T[str(i['id'])] = Tmax
    
    return qlearning(n) # alpha (learning rate)

def fetch_path():
    path = []
    weight_path = []
    node = src_node
    path.append(node)
    while node != des_node:
        max_node = list(qtables[node].keys())[0] 
        for i in qtables[node].keys():
            if qtables[node][max_node] < qtables[node][i]:
                max_node = i
        
        w = weights[node][max_node]
        node = max_node
        path.append(node)
        weight_path.append(w)
    
    return path, weight_path


def main():
    global qtables
    # learning rate
    # itr_lr = []
    # for k in range(1,9):
    #     alpha = k * 0.1
    nodes = []
    mean = []
    q_ini = []
    threshold = 0.3
    for num_nodes in range(9, 100, 1):
        if num_nodes % 10 == 0:
            print("Iteration " + str(num_nodes))
        nodes.append(num_nodes)
        graph_temp = generate_graph(num_nodes,threshold)
        threshold=threshold - (0.001)
        # print("threshold", threshold)

        a_temp = np.random.randint(0, num_nodes-1)
        b_temp = np.random.randint(0, num_nodes-1)
        while not (nx.has_path(graph_temp, a_temp, b_temp) and a_temp != b_temp):
            a_temp = np.random.randint(0, num_nodes-1)
            b_temp = np.random.randint(0, num_nodes-1)
        
        a = str(a_temp)
        b = str(b_temp)

        # print("Str node", a)
        # print("Des node", b)
        no_iterations = []
        # print("graph_temp")
        initialize(a,b,'graph_temp')
        q_ini_iteration = qlearning(num_nodes) # alpha (learning rate)
        q_ini.append(q_ini_iteration)
        # itr_lr.append(q_ini_iteration)
        # print("Q table",qtables)
        # print("T",T)
        # print("Cnt",Cnt)
        # print("Path:",fetch_path())

        original_qtable = dict()
        original_qtable = copy.deepcopy(qtables)

        for i in edges:
            graph_temp2 = copy.deepcopy(graph_temp)
            qtables = copy.deepcopy(original_qtable)
            from_edge = str(i['from'])
            to_edge = str(i['to'])
            # print("Path:",fetch_path())
            # print("Removing edges:",from_edge,to_edge)
            graph_temp2.remove_edge(int(from_edge), int(to_edge))
            if not nx.has_path(graph_temp2, a_temp, b_temp):
                continue
            no_iterations.append(remove_nodes(from_edge,to_edge, num_nodes, alpha))
            # print("Final Qtable")
            # print(qtables)
            # print("Path after removing:", fetch_path())

        # print("No of iterations", no_iterations)
        # print("Mean", np.mean(no_iterations))
        mean.append(np.mean(no_iterations))
    
    # plotting the trends in number of iterations wrt number of nodes in graph 
    plt.title('Trend in iterations with number of nodes \n in graph for agent wise temp implementation (before removing node)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Number of nodes in graph')
    plt.plot(nodes, q_ini, "-o")
    plt.savefig('../plots/trends_agent.jpg')
    plt.show()
    plt.close()

    # plotting the trends in number of iterations wrt learning rate
    # plt.title('Trend in iterations with \n node level temperature')
    # plt.ylabel('Number of iterations')
    # plt.xlabel('Learning rate')
    # plt.plot(nodes, itr_lr, "-o")
    # plt.savefig('../plots/trends_agent_lr.jpg')
    # plt.show()
    # plt.close()
    
    # plotting the trends in number of iterations wrt number of nodes in graph (after removing node)
    plt.title('Trend in iterations with node \n level temperature (after removing node)')
    plt.ylabel('Mean number of iterations')
    plt.xlabel('Number of nodes in graph')
    plt.plot(nodes, mean, "-o")
    plt.savefig('../plots/trends_agent_rn.jpg')
    plt.show()
    plt.close()
    # print("Mean of graphs", mean)
    # print("Mean of mean of graphs", sum(mean)/len(mean))

    pickle.dump(mean, open('../data/mean_agent.p', 'wb'))
    pickle.dump(nodes, open('../data/nodes.p', 'wb'))


if __name__ == "__main__":
    main()