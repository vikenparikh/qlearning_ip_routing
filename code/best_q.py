import json, math, random, copy
import matplotlib.pyplot as plt
from graph_generation import *

qtable = {} # Q table
btable = {} # Best Q table
recov = {} # Recovery rate table
update = {} # Update time table
weights = {} # weights
q_tbl_cnt = 0
src_node = ''
des_node = ''
current_time = 1 # current time

# parameters
alpha = 1 # learning rate
beta = 0.8 # learning rate for recovery rate
gamma = 0.9 # decay of recovery rate

def initialize(s,d,file_name):
    global qtable, btable, recov, update, q_tbl_cnt, src_node, des_node, weights
    # loading graph - json file
    with open('../graphs/'+file_name+'.json','r') as f:
        graph = json.load(f)

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

    # creating a dict for the index of q-tables
    q_tbl_cnt = 0
    qtable = {}
    for i in graph['nodes']:
        l = {}
        for j in graph['edges']:
            if j['from'] == i['id']:
                l[str(j['to'])] = random.uniform(0.1,1)
                q_tbl_cnt += 1
            elif j['to'] == i['id']:
                l[str(j['from'])] = random.uniform(0.1,1)
                q_tbl_cnt += 1
            
            qtable[str(i['id'])] = l

    # intializing best q table values
    for i in qtable.keys():
        l = {}
        for j in qtable[i].keys():
            l[j] = qtable[i][j]

        btable[i] = l 

    # intializing recovery rate table values
    for i in qtable.keys():
        l = {}
        for j in qtable[i].keys():
            l[j] = 0.0

        recov[i] = l 
 
    # intializing update time table values
    for i in qtable.keys():
        l = {}
        for j in qtable[i].keys():
            l[j] = 0.0

        update[i] = l 

    # setting qvalue from destination nodes to 0
    for j in qtable[d].keys():
        qtable[d][j] = 0

    print()
    # initialized Q values table
    print("Initial Q-table")
    print(qtable)
    print()

    print()
    # initialized best q values table
    print("Initial B-table")
    print(btable)
    print()

    print()
    # initialized recovery rate table
    print("Initial R-table")
    print(recov)
    print()

    print()
    # initialized update table
    print("Initial Update table")
    print(update)
    print()
    

    # initial Parameters
    # print("Initial Parameters")
    # print("Learning rate")
    # print(alpha)
    # print("Reward discount rate")
    # print(disc)
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
    # for i in qtable.values():
    #     print("State",l[k], "->  actions", list(i))
    #     k+=1

    # print()

# qlearning function
def qlearning():
    global qtable, btable, recov, update, current_time
    # count - no of iterations 
    count = 0
    # random starting node
    # nodes = [i for i in qtable.keys()]
    # iterating till convergence
    # number of time it converges
    # number of time it converges
    cnt_converge = 0
    # set converge flag to 0
    converge_flag = 0
    # for i in range(200):
    while converge_flag != 1:

        # ri = random.randint(0,len(nodes)-1)
        # node = nodes[ri]
        # store the old qtable
        old_qtable = dict()
        old_qtable = copy.deepcopy(qtable)
        # start from the source node
        node = src_node
        # print("New episode. node:",node)
        while node != des_node:
            next_node = select_action(node, current_time)
            print("next node:", next_node)
            q_old = qtable[node][next_node]
            q = []
            for i in qtable[next_node].items():
                q.append(i[1])
            
            r = -weights[node][next_node]
            if next_node == des_node:
                r = 1000
            
            # print("q", q)
            # print("r, gamma, alpha, qtable[node][next_node]",r, gamma, alpha, qtable[node][next_node])
            dq = alpha * (r + min(q) - qtable[node][next_node])
            q_new = q_old + dq
            qtable[node][next_node] = q_new

            btable[node][next_node] = min(btable[node][next_node], qtable[node][next_node])

            if dq < 0:
                dr = dq / (current_time - update[node][next_node])
                recov[node][next_node] = recov[node][next_node] + beta * dr
            elif dq > 0:
                recov[node][next_node] = gamma * recov[node][next_node]

            update[node][next_node] = current_time  

            # print("q_new q_old w",q_new,q_old,w)
            # print(qtable)
            node = next_node
            current_time += 1

        count += 1
        # check if converges
        elements_convergence = 0
        for u in qtable.keys():
            if converge_flag == 1:
                break
            for v in qtable[u].keys():
                if math.fabs(old_qtable[u][v] - qtable[u][v]) < 0.00005:
                    elements_convergence += 1
    
        # print("elements_convergence",elements_convergence)
        # print("q_tbl_cnt",q_tbl_cnt)
        if elements_convergence == q_tbl_cnt:
            cnt_converge += 1
        
        if cnt_converge > 4:
            converge_flag = 1
        # print("old_qtable_2",old_qtable)
        # print("current qtable", qtable)
        # print("cnt_converge",cnt_converge)
    return count


def select_action(node, current_time):
    global qtable, btable, recov, update
    
    # initiaize q' table
    n_qtable = dict()
    n_qtable = copy.deepcopy(qtable)
    
    for i in qtable[node].keys():
        dt = current_time - update[node][i]
        n_qtable[node][i] = max(qtable[node][i] + dt * recov[node][i], btable[node][i])
    
    next_node = list(n_qtable[node].keys())[0]
    for i in n_qtable[node].keys():
        if n_qtable[node][next_node] > n_qtable[node][i]:
            next_node = i     

    return next_node
    
def remove_nodes(a,b):
    qtable[a][b] = -1000000000000
    return qlearning()

def log(data, filename, message):

    with open('../logs/'+filename+".log",'w') as f:
        f.write(message+"\n\n")
        for k in data.keys():
            f.write(k+" : "+str(data[k])+"\n")


def main():
    
    print("graph5_30")
    initialize('13','7','graph5_30')
    print(qlearning())
    print("Q table")
    print(qtable)
    print("B table")
    print(btable)
    print("R table")
    print(recov)
    print("U table")
    print(update)

    log(qtable, "qtable_log", "Q table")
    log(btable, "btable_log", "B table")
    # path = []
    # node = src_node
    # path.append(node)
    # while node != des_node:
    #     next_node = list(qtable[node].keys())[0]
    #     for k,e in qtable[node].items():
    #         if qtable[node][next_node] > e:
    #             next_node = k
        
    #     node = next_node
        
    #     path.append(next_node)

    # print("path:", path)
    
    # print(remove_nodes('5','20'))
    # print("Final Qtable")
    # print(qtable)
    # print()

    # print("graph3_15")
    # initialize('1','9','graph3_15')
    # cnt.append(qlearning())
    # rem_cnt.append(remove_nodes('3','9'))
    # # print("Final Qtable")
    # # print(qtable)
    # print()

    # print("graph4_20")
    # initialize('2','14','graph4_20')
    # cnt.append(qlearning())
    # # print("Final Qtable")
    # # print(qtable)
    # rem_cnt.append(remove_nodes('1','7'))
    # #print("Final Qtable")
    # #print(qtable)

    # print(cnt)
    # print(rem_cnt)
    # plt.plot(no_nodes,cnt,'ro-')
    # plt.plot(no_nodes,rem_cnt,'bo-')
    # plt.show()

if __name__ == "__main__":
    main()