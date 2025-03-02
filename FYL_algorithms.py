import numpy as np
import networkx as nx
from RMED import RMED
from RUCB import RUCB
from Basic_functions import graph_diameter
import time
import warnings
warnings.filterwarnings("ignore")

# Find the distance of each player from the leader on the graph
# inputs: E is a (M,M) graph adjacency matrix, leader is a scalar specifying the leader
# outputs: a (M,) vector containing the distance of each player from the leader
def distances_from_leader(E,leader):
    E_nx = nx.from_numpy_array(E)
    lengths_dict = dict(nx.single_source_bellman_ford_path_length(E_nx,leader))
    distances_vec = np.array([val for val in lengths_dict.values()])
    distances_vec = distances_vec.astype(int)
    return distances_vec

# Follow Your Leader RMED with simple leader election. assuming 0 is CW
# inputs: P is a (K,K) preference matrix, E is a (M,M) graph adjacency matrix,
#         ID_array is a (M,) vector containing unique ID for each player
#         alpha is RMED parameter, T is a time horizon scalar given to algorithm
#         RMED_flag is a boolean scalar, 1 for RMED2FH, 0 for RMED1
#         f is a scalar parameter for RMED (f(k) function in the paper)
# outputs: a (M,T) matrix containing the cumulative regret for each player
def FYLRMED(P,E,ID_array,alpha,T,RMED_flag,f):
    #Definitions
    leader = np.argmin(ID_array)
    distances_vec = distances_from_leader(E, leader)
    D = graph_diameter(E)
    K, M = P.shape[0], E.shape[0]
    Delta = P[0,:] - 0.5
    T_le = D+1 # Time for leader election
    Regret_mat = np.zeros((M,T)) # Regret for the whole system
    exploit_regret = np.zeros(T) # Regret when choosing the leader CW candidate twice for each round

    # Leader
    Regret_mat[leader,:], candidate_vec = RMED(P,alpha,T,T,RMED_flag,f)
    for t in np.arange(T):
        exploit_regret[t] = Delta[candidate_vec[t]]
    # Followers
    for m in range(M):
        if m != leader:
            T_init = T_le+distances_vec[m]
            if T_init <= T: # Start with independent run and then follow leader
                Regret_mat[m,0:T_init],_ = RMED(P,alpha,T,T_init,RMED_flag,f)
                Regret_mat[m, T_init:] = exploit_regret[(T_le-1):(T-distances_vec[m]-1)]
            else: # Only independent run
                Regret_mat[m, :], _ = RMED(P, alpha, T, T, RMED_flag, f)

    Regret_mat = np.cumsum(Regret_mat,axis=1)
    return Regret_mat

# Follow Your Leader RUCB with simple leader election. assuming 0 is CW
# inputs: P is a (K,K) preference matrix, E is a (M,M) graph adjacency matrix,
#         ID_array is a (M,) vector containing unique ID for each player
#         alpha is RUCB parameter, T is a time horizon scalar given to algorithm
# outputs: a (M,T) matrix containing the cumulative regret for each player
def FYLRUCB(P,E,ID_array,alpha,T):
    # Definitions
    leader = np.argmin(ID_array)
    distances_vec = distances_from_leader(E, leader)
    D = graph_diameter(E)
    K, M = P.shape[0], E.shape[0]
    Delta = P[0,:] - 0.5
    T_le = D+1 # Time for leader election
    Regret_mat = np.zeros((M,T)) # Regret for the whole system
    exploit_regret = np.zeros(T) # Regret when choosing the leader CW candidate twice for each round

    # Leader
    Regret_mat[0,:], candidate_vec = RUCB(P,alpha,T)
    candidate_vec = candidate_vec.astype(int)
    for t in np.arange(T):
        exploit_regret[t] = Delta[candidate_vec[t]]
    # Followers
    for m in range(M):
        if m != leader:
            T_init = T_le+distances_vec[m]
            if T_init <= T: # Start with independent run and then follow leader
                Regret_mat[m,0:T_init],_ = RUCB(P,alpha,T_init)
                Regret_mat[m, T_init:] = exploit_regret[(T_le-1):(T-distances_vec[m]-1)]
            else: # Only independent run
                Regret_mat[m, :], _ = RUCB(P,alpha,T)

    Regret_mat = np.cumsum(Regret_mat,axis=1)
    return Regret_mat