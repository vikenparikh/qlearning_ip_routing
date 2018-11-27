import numpy as np
import networkx as nx
import json, math, copy, random
import matplotlib.pyplot as plt
from graph_generation import *
import pickle
import time

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
epsilon = 0.5

def initialize(s,d,file_name):
    global qtables, q_tbl_cnt, src_node, des_node, weights,edges, graph
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

    # creating a dict for the index of q-tables
    q_tbl_cnt = 0
    qtables = {}
    for i in graph['nodes']:
        l = {}
        for j in graph['edges']:
            if j['from'] == i['id']:
                l[str(j['to'])] = 0.000001 * random.uniform(0.1,1)
                q_tbl_cnt += 1
            elif j['to'] == i['id']:
                l[str(j['from'])] = 0.000001 * random.uniform(0.1,1)
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

# qlearning function
def qlearning():
    global qtables
    # count - no of iterations and t = current t value
    count = 0
    epsilon = 0.5
    # random starting node
    # nodes = [i for i in qtables.keys()]
    # iterating till convergence
    # number of time it converges
    cnt_converge = 0
    # set converge flag to 0
    converge_flag = 0
    # for i in range(200):
    while converge_flag != 1 and count < 1000:
        # print("t",t)
        # print(count)
        # store the old qtable
        old_qtable = dict()
        old_qtable = copy.deepcopy(qtables)
        if count % 50 == 0:
            if epsilon - 0.1 > 0:
                epsilon = epsilon - 0.1
        # print("old_qtable",old_qtable)
        # ri = random.randint(0,len(nodes)-1)
        # node = nodes[ri]
        # start from the source node
        node = src_node
        # print("New episode. node:",node)
        while node != des_node:
            next_node = select_action(node, epsilon)
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
        count += 1
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
    return count


def select_action(node, epsilon):
    global qtables
    if random.uniform(0.1, 0.9) < epsilon: 
        nodes = list(qtables[node].keys())
        mx_t = nodes[random.randint(0,len(nodes)-1)]
    else:
        nodes = list(qtables[node].items())
        mx_t = nodes[0][1]
        for e in nodes:
            if mx_t < e[1]:
                mx_t = e[0]

    # print("max",mx_t)
    return mx_t

def remove_nodes(a,b):
    qtables[a][b] = -100
    return qlearning()


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
    mean = []
    q_ini = []
    threshold = 0.3
    for num_nodes in range(9, 20, 10):
        graph_temp = generate_graph(num_nodes,threshold)
        threshold=threshold - (0.001)
        print("threshold", threshold)

        a_temp = np.random.randint(0, num_nodes-1)
        b_temp = np.random.randint(0, num_nodes-1)
        while not (nx.has_path(graph_temp, a_temp, b_temp) and a_temp != b_temp):
            a_temp = np.random.randint(0, num_nodes-1)
            b_temp = np.random.randint(0, num_nodes-1)
        
        a = str(a_temp)
        b = str(b_temp)

        print("Str node", a)
        print("Des node", b)
        no_iterations = []
        print("graph_temp")
        initialize(a,b,'graph_temp')
        q_ini_iteration = qlearning()
        q_ini.append(q_ini_iteration)
        print("Q table",qtables)
        print("Path:",fetch_path())

        original_qtable = dict()
        original_qtable = copy.deepcopy(qtables)

        for i in edges:
            graph_temp2 = copy.deepcopy(graph_temp)
            qtables = copy.deepcopy(original_qtable)
            from_edge = str(i['from'])
            to_edge = str(i['to'])
            print("Path:",fetch_path())
            print("Removing edges:",from_edge,to_edge)
            graph_temp2.remove_edge(int(from_edge), int(to_edge))
            if not nx.has_path(graph_temp2, a_temp, b_temp):
                continue
            no_iterations.append(remove_nodes(from_edge,to_edge))
            # print("Final Qtable")
            # print(qtables)
            print("Path after removing:", fetch_path())

        print("No of iterations", no_iterations)
        print("Mean", np.mean(no_iterations))
        mean.append(np.mean(no_iterations))
        

        # if num_nodes == 9:
        #     objects = ('q learning', 'q learning after removing node')
        #     y_pos = np.arange(len(objects))
        #     performance = [q_ini_iteration,np.mean(no_iterations)]
            
        #     plt.xticks(y_pos, objects)
        #     plt.ylabel('No of iterations')
        #     plt.title('Base Algorithm')
        #     plt.bar(y_pos, performance, align='center', alpha=0.5)
        #     plt.savefig('../plots/base.jpg')
        #     plt.close()
            
    # plt.title('Trend in iterations')
    # plt.ylabel('Number of iterations')
    # plt.xlabel('Number of nodes in graph')
    # plt.plot(q_ini, "-o")
    # plt.savefig('../plots/q_n_graphs_3.jpg')
    # plt.show()
    # plt.close()

    # plt.title('Trend in iterations with change in learning rate')
    # plt.ylabel('Mean number of iterations')
    # plt.xlabel('Number of nodes in graph')
    # plt.plot(itr_lr, "-o")
    # plt.savefig('../plots/trends_sys_lr.jpg')
    # plt.show()
    # plt.close()
    
    plt.title('Trend in iterations with random (after removing node)')
    plt.ylabel('Mean number of iterations')
    plt.xlabel('Number of nodes in graph')
    plt.plot(mean, "-o")
    plt.savefig('../plots/trends_random.jpg')
    plt.show()
    plt.close()
    print("Mean of graphs", mean)
       
    # print("Mean of mean of graphs", sum(mean)/len(mean))
    # pickle.dump(mean, open('../mean_system1.p', 'wb'))

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