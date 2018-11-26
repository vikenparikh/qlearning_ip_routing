import matplotlib.pyplot as plt
import pickle
import numpy as np

mean_sys = pickle.load(open('../mean_system1.p','rb'))
mean_agt = pickle.load(open('../mean_agent1.p','rb'))

objects = ('q learning', 'q learning agent')
y_pos = np.arange(len(objects))
performance = [sum(mean_sys)/len(mean_sys),sum(mean_agt)/len(mean_agt)]

plt.xticks(y_pos, objects)
plt.ylabel('No of iterations')
plt.title('Q learning vs Q learning agent')
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.savefig('../plots/qlearning_vs_agent.jpg')
plt.show()
plt.close()