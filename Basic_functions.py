import numpy as np
import networkx as nx
import pickle
import os
import json
import joblib
import contextlib
from tqdm import tqdm
import shutil
import warnings

# This function is used to display the progress during runtime of joblib scripts.
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Check that there are no parameters with different lengths.
# inputs: args is a NameSapce object.
# outputs: num_of_exp is the number of experiments to run.
def check_input(args):
    list_parameters = [args.function_to_run, args.graph, args.M, args.gamma, args.alpha, args.RMED_flag, args.rec_flag, args.average_regret_flag, args.single_player_flag, args.lr_coeff]
    # Check that list parameters are all of the same length
    lengths = {len(lst) for lst in list_parameters if len(lst) > 1}
    consistent_length = len(lengths) == 1 if lengths else True
    if consistent_length == False:
        raise IOError("For parameters with multiple values, please use the same number of values for all.")

    # Number of different experiments to run
    num_of_exp = max(len(lst) for lst in list_parameters)

    return num_of_exp

# For parameters that are the same across expriments, extend them into a list.
# inputs: args is a NameSapce object, num_of_exp is the number of experiments to run.
def extend_single_length_lists(args, num_of_exp):
    list_parameters = ['function_to_run', 'graph', 'M', 'gamma','alpha', 'RMED_flag','rec_flag',
                       'single_player_flag','average_regret_flag', 'lr_coeff']
    for attr_name in list_parameters:  # vars(args) returns a dictionary of attributes
        attr_value = getattr(args, attr_name)
        if isinstance(attr_value, list) and len(attr_value) == 1:
            setattr(args, attr_name, attr_value * num_of_exp)

# Set the right parameter values. The preference here is cmd input > config_override > config_default
# inputs: args is a NameSapce object.
#         default config is a .json file that should contain default values for all parameters.
#         config_override is a .json file that overrides values in config_default and doesn't have to give values to all parameters.
def parameters_from_file(args,config_default,config_override):
    with open(config_default, 'r') as file:
        default_config = json.load(file)
    with open(config_override, 'r') as file:
        override_config = json.load(file)
    for attr_name, attr_value in vars(args).items():
        if attr_value == None:
            if override_config.get(attr_name) != None:
                setattr(args, attr_name, override_config.get(attr_name))
            elif default_config.get(attr_name) != None:
                setattr(args, attr_name, default_config.get(attr_name))
            else:
                raise ValueError(f"No value given for {attr_name}")

# Calculate the KL divergence between each Bernoulli mean in matrix P with 0.5.
# inputs: P is a (K,K) preference matrix
# outputs: P_KL is a (K,K) matrix containing the such that P_KL[i,j] is the KL divergence between P[i,j] and 0.5.
def KL_div_half(P):
    if np.any(P>1) or np.any(P<0):
        raise ValueError("Invalid Input")
    else:
        P_KL = P*np.log(2*P)+(1-P)*np.log(2*(1-P))
        if len(P_KL[np.isnan(P_KL)])>0:
            P_KL[np.isnan(P_KL)] = np.log(2)
        return P_KL

# Find graph diameter
# inputs: E is a (M,M) graph adjacency matrix
# outputs: scalar which is the diameter of the graph
def graph_diameter(E):
    E_nx = nx.from_numpy_array(E)
    lengths_dict = dict(nx.shortest_path_length(E_nx))
    return max(len for sub_dict in lengths_dict.values() for len in sub_dict.values())

# Returns a matrix with sortest lengths between nodes, only when these are smaller than the decay parameter
# inputs: E is a (M,M) graph adjacency matrix
#         gamma (integer in [0,...,D]) is the maximal delay allowed during communication
# outputs: sp_mat is a (M,M) matrix. sp_mat[m,n] is the shortest distance between m and n minus 1 as long as the sd is not larger than gamma.
#          If the shortest distance is larger than gamma for some cell, this cell will contain -1.
# NOTE: gamma=0 will result in zeros in the diagonal and -1 otherwise. gamma=1 will result in zeros for immediate neighbors as well.
def shortest_lengths(E,gamma):
    # Definitions
    M = E.shape[0]
    E_nx = nx.from_numpy_array(E)
    sp_mat = np.zeros((M,M),int)
    sp_dict = dict(nx.all_pairs_shortest_path_length(E_nx))
    # Finding shortest paths
    for m in np.arange(M):
        for n in np.arange(M):
            sp_mat[m,n] = sp_dict[m][n]
    # Changing values which are larger than the maximal allowed delay
    sp_mat[sp_mat>gamma] = -1
    sp_mat = sp_mat - 1
    sp_mat[sp_mat == -1] = 0
    sp_mat[sp_mat == -2] = -1
    return sp_mat

# Creates a folder save_folder(n+1), given that save_foldern exists in the current directory (otherwise starts with save_folder1).
# Copies all .py files to the new folder.
# Returns new_folder, a string containing the name of the newly created folder.
# new_folder is a string.
def create_folder(save_folder):
    n = 1
    new_folder = save_folder + str(n)
    while os.path.exists(new_folder):
        n += 1
        new_folder = save_folder + str(n)
    os.makedirs(new_folder)
    return new_folder

# Save the variable save_var under the name save_name.pkl in save_folder.
def save_file(save_folder, save_var,save_name):
    with open(save_folder + '/'+save_name+'.pkl', 'wb') as f:
        pickle.dump(save_var, f)
        
# Check whether a Condorcet Winner exists and return it if it does.
# inputs: P is a (K,K) graph adjacency matrix
# outputs: The index of the CW if it exists, otherwise none
def find_condorcet_winner(P):
    K = P.shape[0]
    for i in range(K):
        P[i,i] = 0.6
        if all(P[i, :] > 0.5):
            P[i,i]=0.5
            return i
        P[i,i]=0.5
    return None

# Converts a SOC/SOI preference data file from preflib.org to a preference matrix P.
# inputs: text_file_name is the name of the file, ending with .txt 
#         Please make sure to delete all introduction lines from the file first (see the text files on github as an example). 
#         num_of_candidates - a integer, the number of candidates within the data
#         num_of_arms_to_keep - the number of arms we want in P, keeping only the best candidates
# outputs: P -  a (num_of_arms_to_keep,num_of_arms_to_keep) preference matrix. 
#          If a CW exists, it is ordered so that the first arm is the CW
def P_generator(text_file_name, num_of_candidates, num_of_arms_to_keep):
    # Initialize empty lists to store data
    num_of_repeats_list = [] # Stores the number of repeats of each line
    ordered_list = [] # Stores the order in each line

    # Read the .txt file line by line
    with open(text_file_name, 'r') as file:
        for line in file:
            # Split the line by ":" to separate the number of repeats from the rest
            parts = line.strip().split(':')

            # Convert to int and append to the list
            num_of_repeats_list.append(int(parts[0]))

            # Split the rest of the ordered preferences by "," and convert them to integers
            numbers = [int(x) for x in parts[1].split(',')]
            numbers.extend([-1] * (num_of_candidates - len(numbers)))

            # Append to the ordered_matrix
            ordered_list.append(numbers)

    # Convert the lists to NumPy arrays
    num_of_repeats_array = np.array(num_of_repeats_list)
    ordered_matrix = np.array(ordered_list)
    ordered_matrix = ordered_matrix - 1

    # Build Preference matrix
    K = num_of_candidates
    W = np.zeros((K, K))
    single_win_cnt = np.zeros((K,)) # Number of times an arm appears alone in a line
    for line in range(len(num_of_repeats_array)):
        line_len = np.where(ordered_matrix[line, :] == -2)
        if len(line_len[0]) > 0: # Partial order
            line_len = line_len[0][0]
        else: # Complete order
            line_len = K
        if line_len > 1: # At least two arms in a line, update win matrix
            for i in range(line_len):
                for j in range(i + 1, line_len):
                    W[ordered_matrix[line, i], ordered_matrix[line, j]] += num_of_repeats_array[line]
        else: # One arm in a line, update single win counter
            single_win_cnt[ordered_matrix[line, 0]] += num_of_repeats_array[line]

    num_of_wins = np.sum(W, axis=1)
    num_of_wins = num_of_wins + single_win_cnt

    # Keep only the best and build P
    sorted_indices = np.argsort(num_of_wins)
    sorted_indices_descending = sorted_indices[::-1] # Sorted indices of arms according to the number of wins
    W = W[sorted_indices_descending[:num_of_arms_to_keep]][:, sorted_indices_descending[:num_of_arms_to_keep]] # Keep only best arms
    np.fill_diagonal(W, 1)
    P = W / (W + np.transpose(W))

    # Find the Condorcet winner
    condorcet_winner = find_condorcet_winner(P)
    if condorcet_winner is not None:
        print(f"The Condorcet winner is item {condorcet_winner}.")
    else:
        print("There is no Condorcet winner.")

    # Order matrix P so that the CW is the first arm
    P_new = P.copy()
    if condorcet_winner is not None:
        P_new[0, :] = P[condorcet_winner, :]
        P_new[condorcet_winner, :] = P[0, :]
        tmp_c = P_new[:, condorcet_winner].copy()
        tmp_0 = P_new[:, 0].copy()
        P_new[:, 0] = tmp_c
        P_new[:, condorcet_winner] = tmp_0

    return P_new