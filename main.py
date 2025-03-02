import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import os
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib
from Versatile_DB import VDB
from joblib import Parallel, delayed
from FYL_algorithms import FYLRMED, FYLRUCB
from MPRUCB import MP_RUCB
from Basic_functions import KL_div_half, graph_diameter, shortest_lengths,create_folder,save_file, \
     P_generator, check_input, extend_single_length_lists,parameters_from_file, tqdm_joblib
import pickle
import json
import argparse
from tqdm import tqdm
import shutil

# Function to convert string to boolean
def str2bool(str_input):
    if isinstance(str_input, str):
        if str_input.lower() in ('True','yes', 'true', 't', 'y', '1'):
            return True
        elif str_input.lower() in ('False','no', 'false', 'f', 'n', '0'):
            return False
    return str_input
########################## Parameters #########################
# Command Line Input
parser = argparse.ArgumentParser(description="Run Multiplayer Dueling Bandit Algorithms.")
parser.add_argument('--num_of_runs', type=int, help='Number of runs for each experiment (integer).')
parser.add_argument('--num_of_workers', type=int, help='Number of parallel CPU cores to use (integer).')
parser.add_argument('--save_flag', type=str2bool, help='True to show the figure at the end (bool).')
parser.add_argument('--plot_flag', type=str2bool, help='True to save the parameters, regret and plot in a sub folder (bool).')
parser.add_argument('--new_folder_name',type=str,help='Name of the folder to save results (str).')
parser.add_argument('--function_to_run', nargs='+',type=str, choices = ['MP_RUCB','FYLRUCB','FYLRMED','VDB'], help = 'Algorithms to run from `MP_RUCB`, `FYLRUCB`, `FYLRMED`, `VDB` (str list).')
parser.add_argument('--dataset',type=str,choices=['Sushi','Irish','Six_rankers','Arithmetic'],help='Dataset to use from `Sushi`, `Irish`, `Six_rankers`, `Arithmetic` (str list).')
parser.add_argument('--T',type=int,help='Number of rounds for each experiment (integer).')
parser.add_argument('--graph', nargs='+',type=str, choices = ['complete','cycle','star','path'], help = 'Communication graph from `complete`, `cycle`, `star`, `path` (str list).')
parser.add_argument('--M', nargs='+',type=int,help='Number of players (str int).')
parser.add_argument('--gamma', nargs='+',type=int,help='Decay parameter: -1 to use the diameter. (int list)')
parser.add_argument('--lr_coeff',nargs='+',type=int,default=[2], help='Learning rate coefficient for VDB (double).')
parser.add_argument('--alpha', nargs='+',type=float,help='Exploration parameter for RUCB and RMED (float list).')
parser.add_argument('--RMED_flag',nargs='+',type=str2bool,help='True to use RMED2FH, False for RMED1 (bool list).')
parser.add_argument('--rec_flag',nargs='+',type=str2bool,help='True to use CW recommendations for MPRUCB, Fasle otherwise (bool list).')
parser.add_argument('--average_regret_flag',nargs='+',type=str2bool,help='True to display the average regret over players instead of the sum (bool list).')
parser.add_argument('--single_player_flag',nargs='+',type=str2bool,help='True to use a single-agent with `M` decisions per round (bool list).')
parser.add_argument('--config_override',type=str,default="default_parameters.json",help='Name of a .json file to override the ones in default_parameters.json (str).')
args = parser.parse_args()

# Check configuration files and update input
parameters_from_file(args,"default_parameters.json",args.config_override)
number_of_exp = check_input(args)
extend_single_length_lists(args, number_of_exp)

# Define experiment parameters
function_names =['MP_RUCB','FYLRUCB','FYLRMED','VDB']
function_dict = {name: globals()[name] for name in args.function_to_run if name in globals()}

if args.dataset == 'Sushi':
    with open('sushi.pkl', 'rb') as file:
        P = pickle.load(file) # Preference matrix
elif args.dataset == 'Irish':
    with open('irish.pkl', 'rb') as file:
        P = pickle.load(file)
elif args.dataset == 'Six_rankers':
    with open('six_arms.pkl', 'rb') as file:
        P = pickle.load(file)
elif args.dataset == 'Arithmetic':
    with open('arithmetic.pkl', 'rb') as file:
        P = pickle.load(file)

graph_names =['complete','cycle','star','path']
graph_dict = {
    'cycle': nx.cycle_graph,
    'star': nx.star_graph,
    'complete': nx.complete_graph,
    'path': nx.path_graph
}
G = [] # Communication graph
for exp in range(number_of_exp):
    if args.single_player_flag[exp]:
        graph = graph_dict[args.graph[exp]](1)
    else:
        graph = graph_dict[args.graph[exp]](args.M[exp])
    G.append(graph)

E = [nx.to_numpy_array(G[exp]) for exp in range(number_of_exp)] # Edge matrix
K = P.shape[0] # Number of arms

# Message Passing (MP) parameters
D = [graph_diameter(E[exp]) for exp in range(number_of_exp)] # Diameter
if args.gamma == [-1 for _ in range(number_of_exp)]:
    args.gamma = [D[exp] for exp in range(number_of_exp)]
sp_mat = [shortest_lengths(E[exp], args.gamma[exp]) for exp in range(number_of_exp)]  # shortest-lengths matrix (zero is the shortest length is larger than gamma)
# FYL and RMED parameters
ID_array =[np.arange(args.M[exp])*(1-args.single_player_flag[exp])+np.arange(1)*args.single_player_flag[exp] for exp in range(number_of_exp)] # unique ID for each player
f =0.3*(K**(1.01)) # RMED parameter

try:
    with open("plot_parameters.json", 'r') as file:
        config = json.load(file)
        title_str = config.get("title","Group Regret Per Round")
        xlabel_str = config.get("xlabel","round")
        ylabel_str = config.get("ylabel","Regret R(t)")
        yaxis_log = config.get("yaxis_log_flag", True)
        lower_bound_flag = config.get("lower_bound_flag",True)
        if isinstance(config.get("legend"), list) and len(config.get("legend")) == number_of_exp:
            legend_str = config.get("legend")
        else:
            legend_str = ["exp"+str(exp) for exp in range(1,number_of_exp+1)]
except:
    raise IOError(f"Please use a correct plot parameters file.")

##################### Calculating lower bounds ####################
# b_vector calculation - b_vector[i] is the challenging arm for arm i
CW_candidate = 0
Delta_CWC_row = (P - 0.5).copy()
Delta_CWC_row = Delta_CWC_row[CW_candidate,:]
inst_CWC_regret = np.transpose(np.repeat(Delta_CWC_row[:,np.newaxis],K,axis=1).copy())+np.repeat(Delta_CWC_row[:,np.newaxis],K,axis=1)
inst_CWC_regret = inst_CWC_regret/KL_div_half(P)
inst_CWC_regret[P>=0.5] = float('inf')
np.fill_diagonal(inst_CWC_regret, float('inf'))
b_vector = np.argmin(inst_CWC_regret,axis=1)
for i in range(K):
    if b_vector[i] == i:
        b_vector[i] = np.random.choice(np.setdiff1d(np.arange(0,K),i))
# lower bound
LB = 0
t = np.arange(1, args.T+1)
for i in np.arange(1, K):
    LB = LB + (P[0,i]+P[0,b_vector[i]]-1)/(2*KL_div_half(P[i,b_vector[i]]))
LB = LB*np.log(t)
##################### Running Experiment ####################
tasks = []
parameter_map = {
    'MP_RUCB': lambda exp: (P, E[exp], args.alpha[exp], args.T+(args.M[exp]-1)*args.T*args.single_player_flag[exp], args.gamma[exp], D[exp], sp_mat[exp],
                                                                   args.rec_flag[exp]),
    'FYLRUCB': lambda exp: (P, E[exp], ID_array[exp], args.alpha[exp], args.T+(args.M[exp]-1)*args.T*args.single_player_flag[exp]),
    'FYLRMED': lambda exp: (P, E[exp], ID_array[exp], args.alpha[exp], args.T+(args.M[exp]-1)*args.T*args.single_player_flag[exp],args.RMED_flag[exp],f),
    'VDB': lambda exp: (P,args.T, args.lr_coeff[exp])
}

# Append tasks
for exp in range(number_of_exp):
    func_name = args.function_to_run[exp]
    func = function_dict[func_name]
    param_func = parameter_map[func_name]
    tasks.extend(delayed(func)(*param_func(exp)) for _ in range(args.num_of_runs))



# Execute all tasks in parallel
t_start = time.time()
with tqdm_joblib(tqdm(desc="Progress", total=args.num_of_runs*number_of_exp)) as progress_bar:
    results = Parallel(n_jobs=args.num_of_workers)(tasks)
t_end = time.time()
print('total time is {:.2f} seconds'.format(t_end-t_start))

# Separate experiments and calculate statistics
Regret_vector = np.zeros((number_of_exp,args.num_of_runs,args.T))
Regret_statistics = np.zeros((number_of_exp,2,args.T))
for exp in range(number_of_exp):
    split_list = results[exp*args.num_of_runs:(exp+1)*args.num_of_runs]
    skip_rounds = 1+(args.M[exp]-1)*args.single_player_flag[exp]
    Regret_vector[exp,:,:] = np.sum(np.stack(split_list,axis=0), axis=1)[:,0:(args.T*skip_rounds):(skip_rounds)]/skip_rounds
    if (skip_rounds==1)&(args.average_regret_flag[exp]==1):
        Regret_vector[exp, :, :] =Regret_vector[exp,:,:]/args.M[exp]
    Regret_statistics[exp,0,:] = np.mean(Regret_vector[exp,:,:], axis=0)
    Regret_statistics[exp, 1, :] = np.std(Regret_vector[exp, :, :], axis=0)

##################### Plotting ####################
if args.plot_flag==0:
    matplotlib.use('Agg')
plt.figure()
for exp in range(number_of_exp):
    plt.plot(t, Regret_statistics[exp,0,:], label=legend_str[exp])
    plt.fill_between(t, Regret_statistics[exp,0,:] - Regret_statistics[exp,1,:], Regret_statistics[exp,0,:] + Regret_statistics[exp,1,:],
                     alpha=0.2)
if lower_bound_flag:
    plt.plot(t, LB, color='yellow',label='LB')
plt.xscale('log')
if yaxis_log:
    plt.yscale('log')
plt.xlabel(xlabel_str, fontsize=16)
plt.legend(fontsize=14)
plt.ylabel(ylabel_str, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(title_str, fontsize=16)
if args.save_flag ==1:
    new_folder = create_folder(args.new_folder_name)  # creates new folder and copies files
    plt.savefig(new_folder + '/figure.png', bbox_inches='tight')
if args.plot_flag:
    plt.show()

##################### Saving (please close the figure first) ####################
if args.save_flag ==1:
    # Save lower bound
    save_var = LB
    save_name = 'LB'
    save_file(new_folder, save_var, save_name) # Save LB as meta_variables.pkl

    # Save regrets
    save_var = Regret_statistics
    save_name = 'Regret Statistics'
    save_file(new_folder, save_var, save_name)

    # Save parameters used for this run
    args_dict = vars(args)  # Convert args to a dictionary
    parameters_file_path = os.path.join(new_folder, 'parameters.json')
    with open(parameters_file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)  #

    #Copy plot_parameters file
    shutil.copy("plot_parameters.json", os.path.join(new_folder, "plot_parameters.json"))