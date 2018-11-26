import numpy as np
import json, math, random, copy
import matplotlib.pyplot as plt
from graph_generation import *

qtables = {}
weights = {}
q_tbl_cnt = 0
src_node = ''
des_node = ''
edges = {}
graph = {}

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
                l[str(j['to'])] = j['weights']
            elif j['to'] == i['id']:
                l[str(j['from'])] = j['weights']

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
                l[str(j['to'])] = 0.001 * random.uniform(0.1,1)
                q_tbl_cnt += 1
            elif j['to'] == i['id']:
                l[str(j['from'])] = 0.001 * random.uniform(0.1,1)
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
    # print(qtables)
    # list of probabilities
    p = []    
    # calculate probability for each action
    # den = sum_a(e^Q(s,a)/T)
    den = 0
    for i in qtables[node].items():
        den += math.exp(i[1] / T[node])
    # calculating the probability 
    for i in qtables[node].items():
        # print(i[1])
        t_p =  (math.exp(i[1] / T[node]) ) / den
        p.append((i[0],t_p))
    # print(p)
    # print(max(p)[0])
    # return best action based on probabilities
    mx = max(p, key=lambda item: item[1])
    return mx[0]


# qlearning function
def qlearning(n):
    global qtables, Cnt, T
    # count - no of iterations
    count_i = 0
    # random starting node
    # nodes = [i for i in qtables.keys()]
    # iterating till convergence
    # number of time it converges
    cnt_converge = 0
    # set converge flag to 0
    converge_flag = 0
    # for i in range(200):
    while converge_flag != 1:
        # print("t",t)
        # print(count_i)
        # store the old qtable
        old_qtable = dict()
        old_qtable = copy.deepcopy(qtables)
        # print("old_qtable",old_qtable)
        
        # start from the source node
        node = src_node
        Cnt[node] += 1
        
        # ri = random.randint(0,len(nodes)-1)
        # node = nodes[ri]
        # print("New episode. node:",node)
        while node != des_node:
            
            if Cnt[node] % n/5 == 0:
                if T[node] - 1000 > 0:
                    T[node] -= 1000
                elif T[node] - 190 > 0:
                    T[node] -= 190

            next_node = select_action(node,T)
            Cnt[next_node] += 1
            # print("next node:", next_node)
            q_old = qtables[node][next_node]
            q = []
            for i in qtables[next_node].items():
                q.append(i[1])
            
            r = -weights[node][next_node]
            if next_node == des_node:
                r = 10
            
            # print("q", q)
            # print("r, gamma, alpha, qtables[node][next_node]",r, gamma, alpha, qtables[node][next_node])
            w = alpha * (r + (gamma * max(q))-qtables[node][next_node])
            q_new = q_old + w
            qtables[node][next_node] = q_new

            # print("q_new q_old w",q_new,q_old,w)
            # print(qtables)
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
    
        # print("elements_convergence",elements_convergence)
        # print("q_tbl_cnt",q_tbl_cnt)
        if elements_convergence == q_tbl_cnt:
            cnt_converge += 1
        
        if cnt_converge > 4:
            converge_flag = 1
        # print("old_qtable_2",old_qtable)
        # print("current qtable", qtables)
        # print("cnt_converge",cnt_converge)
    return count_i


def select_action(node, T):
    global qtables
    # print("select_action-",qtables[node].items())
    # print(qtables)
    # mx_t = max(qtables[node].items(), key=lambda item: item[1])
    mx_t = softmax(node, T)
    # print("max",mx_t)
    return mx_t

def remove_nodes(a,b, n):
    global T, graph
    qtables[a][b] = -100
    
    # intializing the temperature dict
    for i in graph['nodes']:
        T[str(i['id'])] = Tmax
    
    return qlearning(n)

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
    # no_nodes = [10,15,20]
    # cnt = []
    # rem_cnt = []
    # threshold = 0.1
    # for i in range(10, 11):
    #     generate_graph(i,threshold)
    #     threshold=threshold - (0.25)/500
    #     print("threshold", threshold)
    #     a=str(i-1)
    #     b=str(i-5)
    #     initialize(a,b,'graph6_30')
    #     t_cnt = qlearning()
    #     cnt.append(t_cnt)
    #     print("t_cnt",t_cnt)

    # print(cnt)

    no_iterations = []
    print("graph3_15")
    initialize('1','3','graph3_15')
    print("No of iteration", qlearning(15))
    print("Q table",qtables)
    print("T",T)
    print("Cnt",Cnt)
    print("Path:",fetch_path())

    original_qtable = dict()
    original_qtable = copy.deepcopy(qtables)

    for i in edges:
        qtables = copy.deepcopy(original_qtable)
        from_edge = str(i['from'])
        to_edge = str(i['to'])
        print("Path:",fetch_path())
        print("Removing edges:",from_edge,to_edge)
        no_iterations.append(remove_nodes(from_edge,to_edge, 15))
        # print("Final Qtable")
        # print(qtables)
        print("Path after removing:", fetch_path())

    print("No of iterations", no_iterations)
    print("Mean", np.mean(no_iterations))
    # print("graph3_15")
    # initialize('1','9','graph3_15')
    # cnt.append(qlearning())
    # rem_cnt.append(remove_nodes('3','9'))
    # # print("Final Qtable")
    # # print(qtables)
    # print()

    # print("graph4_20")
    # initialize('2','14','graph4_20')
    # cnt.append(qlearning())
    # # print("Final Qtable")
    # # print(qtables)
    # rem_cnt.append(remove_nodes('1','7'))
    # #print("Final Qtable")
    # #print(qtables)

    # print(cnt)
    # print(rem_cnt)
    # plt.plot(no_nodes,cnt,'ro-')
    # plt.plot(no_nodes,rem_cnt,'bo-')
    # plt.show()

if __name__ == "__main__":
    main()