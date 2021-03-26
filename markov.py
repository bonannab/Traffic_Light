import os
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl


hidden_states = ['off', 'red', 'yellow', 'red-yellow', 'green']
pi = [0.2, 0.2, 0.2, 0.2, 0.2]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())


#hidden_state transition matrix (valamilyen állapotból melyikbe mehet át, kb. egyenletes eloszlás mivel nem tudunk semmit)
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.2, 0.2, 0.2, 0.2, 0.2]
a_df.loc[hidden_states[1]] = [0.3, 0.35, 0, 0.35, 0]
a_df.loc[hidden_states[2]] = [0.3, 0.35, 0.35, 0, 0]
a_df.loc[hidden_states[3]] = [0.3, 0, 0, 0.35, 0.35]
a_df.loc[hidden_states[4]] = [0.3, 0, 0.35, 0, 0.35]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))


observable_states = ['off', 'red', 'yellow', 'red-yellow', 'green']

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

#megfigyelési átmenet mátrix
#obs = np.array([0,0,1,1,1,3,3,3,4,4,2,4,4,4,4,1,2,2,2,4,1])
'''b_df.loc[hidden_states[0]] = [0.5, 0.5, 0, 0, 0]
b_df.loc[hidden_states[1]] = [0, 0.5, 0.25, 0.25, 0]
b_df.loc[hidden_states[2]] = [0, 0, 0.5, 0, 0.5]
b_df.loc[hidden_states[3]] = [0, 0, 0, 0.65, 0.35]
b_df.loc[hidden_states[4]] = [0, 0.28, 0.14, 0, 0.58]'''

#print(b_df)

#b = b_df.values
#print('\n', b, b.shape, '\n')
#print(b_df.sum(axis=1))

# observations are encoded numerically
obs_map = {'off':0, 'red':1, 'yellow':2, 'red-yellow': 3, 'green': 4}
obs = np.array([0,0,1,1,1,3,3,3,4,4,2,4,4,4,4,1,2,2,2,4,1])
obs_prob = [[0.3,0.5,0.2,0.2,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
            [0.2,0.2,0.4,0.5,0.6,0.2,0.2,0.1,0.1,0.1,0.2,0.2,0.1,0.1,0.1,0.3,0.2,0.2,0.1,0.2,0.4],
            [0.2,0.1,0.2,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.3,0.1,0.1,0.1,0.1,0.2,0.4,0.5,0.6,0.2,0.2],
            [0.1,0.1,0.1,0.1,0.1,0.3,0.4,0.5,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.2,0.2,0.1,0.1,0.2,0.2],
            [0.2,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.3,0.4,0.2,0.4,0.5,0.6,0.6,0.2,0.1,0.1,0.1,0.3,0.1]]
obs_prob = np.array(obs_prob)

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]),columns=['Obs_code', 'Obs_seq']) )

# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
#def viterbi(pi, a, b, obs, obs_prob):
def viterbi(pi, a, obs_prob):
    nStates = np.shape(a)[0]
    T = np.shape(obs_prob[1])[0]
    #T = np.shape(obs_prob)[:0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * obs_prob[:,0] #b[:, obs[0]]
    #delta[:, 0] = pi * b[:, obs_prob[:,0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * obs_prob[s,t] #* b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi} ---{delta}'.format(s=s, t=t, phi=phi[s, t], delta=delta[s,t]))

    # find optimal path
    print('-' * 50)
    print('Start Backtrace\n')
    path[T - 1] = np.argmax(delta[:, T - 1])
    # p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T - 2, -1, -1):
        path[t] = phi[int(path[t + 1]), [t + 1]]
        # p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi


#path, delta, phi = viterbi(pi, a, b, obs, obs_prob)
path, delta, phi = viterbi(pi, a, obs_prob)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

state_map = {0:'off', 1:'red', 2:'yellow', 3:'red-yellow', 4:'green'}
state_path = [state_map[v] for v in path]

print(obs_seq)
print(state_path)
(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))





