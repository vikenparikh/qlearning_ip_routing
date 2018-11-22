import json, math, random, copy

# loading graph - json file
with open('./Graphs/graph1.json','r') as f:
    graph = json.load(f)

# extracting the weights of the edges
weights = {}
for i in graph['nodes']:
    l = {}
    for j in graph['edges']:
        if j[0] == i:
          l[j[1]] = j[2]
        elif j[1] == i:
            l[j[0]] = j[2]
        
        weights[i] = l

# intializing source and destination node
src_node = "a"
des_node = "g"

print()
# weights
print("Weights")
print(weights)
print()

# creating a dict for the index of q-tables
q_tbl_cnt = 0
qtables = {}
for i in graph['nodes']:
    l = {}
    for j in graph['edges']:
        if j[0] == i:
          l[j[1]] = random.uniform(1,4)
          q_tbl_cnt += 1
        elif j[1] == i:
            l[j[0]] = random.uniform(1,4)
            q_tbl_cnt += 1
        
        qtables[i] = l

# setting qvalue from destination nodes to 0
for j in qtables["g"].keys():
    qtables["g"][j] = 0

print()
# initialized Q values table
print("Initial Q-table")
print(qtables)
print()

# parameters
alpha = 0.1 # learning rate
drate = 1.0 # reward discount rate
err = 0.1 # error of approaching extremities
gamma = 0.9 # discount rate
Tmax = 2000 # Temperature max (Exploration limit)
Tmin = 10 # Temperature min (Exploitation limit)
T = Tmax # Current Temperature value

# initial Parameters
print("Initial Parameters")
print("Learning rate")
print(alpha)
print("Reward discount rate")
print(drate)
print("Error of approaching extremities")
print(err)
print("Discount rate")
print(gamma)
print("Temperature max Exploration limit")
print(Tmax)
print("Temperature min Exploitation limit")
print(Tmin)
print()

# states i.e. nodes on the graph
l=[]
print("States of the graph are:")
for i in graph['nodes']:
    l.append(i)
    print(i)

print()
# actions from all these states are:
print("Actions from all these states are:")
k=0
for i in qtables.values():
    print("State",l[k], "->  actions", list(i))
    k+=1

print()

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
        den += math.exp(i[1] / T)
    # calculating the probability 
    for i in qtables[node].items():
        # print(i[1])
        t_p =  (math.exp(i[1] / T) ) / den
        p.append((i[0],t_p))
    # print(p)
    # print(max(p)[0])
    # return best action based on probabilities
    mx = max(p, key=lambda item: item[1])
    return mx[0]

# ???
# def weight(prev_node, current_node):
#     # calculate the weight = alpha*(r + gamma * max_a' Q(s',a') - Q(s,a))
#     qdiff = []
#     for i in qtables[current_node].items():
#         qdiff.append(i[1]-qtables[prev_node][current_node])

#     w = alpha * ((-1) + gamma * max(qdiff))

#     return w

# ???
# def route(current_node, prev_node):
#     if current_node != des_node:
#         # determine next node
#         next_node = softmax(current_node)

#         # prev q value
#         q_old = qtables[current_node][next_node]

#         # check if prev_node isn't source
#         if prev_node != source_node:
#             q_new = q_old + weight(prev_node,current_node)
#         else:
#             q_new = q_old

#         # update new q value to the q table
#         qtables[current_node][next_node] = q_new
#         return 0

#     else:
#         return 0,weight(prev_node,current_node) 

# qlearning function
def qlearning():
    global qtables
    # count - no of iterations and t = current t value
    count = 0
    t = 2000
    # random starting node
    # nodes = [i for i in qtables.keys()]
    # iterating till convergence
    # number of time it converges
    cnt_converge = 0
    # set converge flag to 0
    converge_flag = 0
    # for i in range(200):
    while converge_flag != 1:
        # store the old qtable
        old_qtable = dict()
        old_qtable = copy.deepcopy(qtables)
        # print("old_qtable",old_qtable)
        if count % 15 == 0:
            t = t - 20
        # ri = random.randint(0,len(nodes)-1)
        # node = nodes[ri]
        # start from the source node
        node = src_node
        # print("New episode. node:",node)
        while node != des_node:
            next_node = select_action(node,t)
            # print("next node:", next_node)
            q_old = qtables[node][next_node]
            q = []
            for i in qtables[next_node].items():
                q.append(i[1])
            
            r = -weights[node][next_node]
            if next_node == des_node:
                r = 100
            
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
                if math.fabs(old_qtable[u][v] - qtables[u][v]) < 0.00005:
                    elements_convergence += 1
    
        # print("elements_convergence",elements_convergence)
        # print("q_tbl_cnt",q_tbl_cnt)
        if elements_convergence == q_tbl_cnt:
            cnt_converge += 1
        
        if cnt_converge > 10:
            converge_flag = 1
        # print("old_qtable_2",old_qtable)
        # print("current qtable", qtables)
        # print("cnt_converge",cnt_converge)
    print(count)


def select_action(node, t):
    global qtables
    # print("select_action-",qtables[node].items())
    # print(qtables)
    # mx_t = max(qtables[node].items(), key=lambda item: item[1])
    mx_t = softmax(node, t)
    # print("max",mx_t)
    return mx_t

# print(qtables)
qlearning()

# final q table
print()
print("Final Qtable")
print(qtables)
print()

# Change the network
edges = graph["edges"]
del_edge = edges.pop(0)
graph["edges"] = edges
print(del_edge)

# remove a -> b
qtables["a"]["b"] = -1000000000000

print("Initial qtable",qtables)
qlearning()
print("Final qtable",qtables)

# remove a -> b
qtables["e"]["d"] = -1000000000000

print("Initial qtable",qtables)
qlearning()
print("Final qtable",qtables)
