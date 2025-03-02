import numpy as np
from Basic_functions import KL_div_half
import warnings

# Decide on the second arm to draw for single player RMED.
# inputs: T is a time horizon scalar, mu is a (K,K) matrix containing current empirical probability means,
#         N is a (K,K) matrix containing number of visitations, CW_candidate (scalar) is the CW candidate minimizing the empirical divergence,
#         b_arm (scalar) is the empirical arm b (see paper) for the first drawn arm, first_arm (scalar) is the first arm drawn,
#         RMED_flag is a boolean scalar, 1 for RMED2FH, 0 for RMED1
# outputs: Returns the second arm to be drawn
def Choose_Second_Arm(T,mu, N, CW_candidate, b_arm, first_arm,RMED_flag):
    winners = np.where(mu[first_arm,:]<=0.5)[0]
    winners = np.delete(winners,np.where(winners==first_arm))
    # Choose the challenging arm
    if (RMED_flag == 1) and (b_arm in winners) and N[first_arm,CW_candidate]>=N[first_arm,b_arm]/np.log(np.log(T)):
        return b_arm
    # Choose the CW candidate
    elif (CW_candidate in winners) or (len(winners) == 0):
        return CW_candidate
    # Choose the most challenging arm
    else:
        tmp = mu[first_arm,:].copy()
        tmp = np.delete(tmp,first_arm)
        challenging_arm = np.argmin(tmp)
        if challenging_arm >= first_arm:
            challenging_arm += 1
        return challenging_arm

# Find arm b for each arm (see paper) in RMED2FH
# inputs: mu is a (K,K) containing empirical probabilities, CW_candidate (scalar) is the CW candidate minimizing the empirical divergence
# outputs: b_vector is a (K,) containing the b arm for each arm
def find_b_vector(mu,CW_candidate):
    # Defining the b_arms for RMED2FH
    K = mu.shape[0]
    b_vector = np.zeros(K)
    Delta_CWC_row = (mu - 0.5).copy()
    Delta_CWC_row = Delta_CWC_row[CW_candidate, :]
    inst_CWC_regret = (np.transpose(np.repeat(Delta_CWC_row[:, np.newaxis], K, axis=1).copy()) +
                       np.repeat(Delta_CWC_row[:, np.newaxis], K,
                                 axis=1))  # Instantaneous regret when choosing (i,j) if CW_candidate was CW
    b_matrix = inst_CWC_regret / KL_div_half(mu)
    b_matrix[mu >= 0.5] = float('inf')
    np.fill_diagonal(b_matrix, float('inf'))  # Matrix to minimize to find b_vector
    # Find b_vector for each arm
    for i in range(K):
        tmp_vec = np.where(b_matrix[i, :] == np.min(b_matrix[i, :]))[0]
        if len(tmp_vec) > 1:
            tmp_vec = tmp_vec[np.random.randint(0, len(tmp_vec))]
        b_vector[i] = tmp_vec
        if b_vector[i] == i:
            b_vector[i] = np.random.choice(np.setdiff1d(np.arange(0, K), i))
    b_vector = b_vector.astype((int))
    return b_vector

# Single-player RUCB for T rouns, return only first T_Stop rounds. assuming 0 is CW.
# inputs: P is a (K,K) preference matrix, alpha is RMED parameter
#         T is a time horizon scalar given to algorithm, T_stop is the number of rounds we actually run (<=T)
#         RMED_flag is a boolean scalar, 1 for RMED2FH, 0 for RMED1
#         f is a scalar parameter for RMED (f(k) function in the paper)
# outputs: r is a (T,) array of instantaneous regret, candidate_vec is a (T,) vector of candidate CW at each round
def RMED(P,alpha,T,T_stop,RMED_flag,f):
    # Defining variables
    K = P.shape[0]
    Delta_mat = P - 0.5
    Delta_first_row = Delta_mat[0,:]
    inst_regret_mat = 0.5*(np.transpose(np.repeat(Delta_first_row[:,np.newaxis],K,axis=1).copy())+
                           np.repeat(Delta_first_row[:,np.newaxis],K,axis=1)) # Regret incurred when drawing arms (i,j)
    Regret_vec = np.zeros(T)
    candidate_vec = np.ones(T)
    L = int(np.ceil(alpha*np.log(np.log(T))))*RMED_flag+(1-RMED_flag) # For initialization phase
    W = np.zeros((K,K)) # Winners matrix
    N = np.zeros((K, K)) # Visitation matrix
    b_vector = np.zeros(K) # Empirical arm b (see paper) for each arm
    mu = np.zeros((K,K)) # Empirical mean matrix
    L_C = np.arange(K) # Current list
    L_N = np.array([]) # Next list
    t = 0

    # Initial Phase - draw all pairs L times
    for rep in np.arange(L):
        for i in np.arange(K):
            for j in np.arange(i,K):
                # Draw and update matrices
                reward = np.random.binomial(1, P[i,j])
                W[i, j] = W[i, j] + reward
                N[i, j] += 1
                if i != j:
                    W[j, i] = W[j, i] + 1 - reward
                    N[j, i] += 1
                Regret_vec[t] = inst_regret_mat[i, j]
                mu = W / N
                # Find current CW candidate (minimizing I_vec in the paper)
                only_winners = mu <= 0.5
                np.fill_diagonal(only_winners, False)
                I_vec = np.sum(N * KL_div_half(mu) * only_winners, axis=1)
                CW_candidate = np.where(I_vec == np.min(I_vec))[0]
                if len(CW_candidate) > 1:
                    CW_candidate = CW_candidate[np.random.randint(0, len(CW_candidate))]
                CW_candidate = int(CW_candidate.item())
                candidate_vec[t] = CW_candidate
                t += 1
    # Defining the b_arms for RMED2FH
    if RMED_flag == 1:
        b_vector = find_b_vector(mu,CW_candidate)

    # Main loop
    while t<= T_stop-1:

        # Move to next list if current list is empty
        if len(L_C) == 0:
            L_N = L_N.astype(int)
            L_C = L_N
            L_N = np.array([])
        # Draw arms and obtain reward
        first_arm = L_C[0]
        second_arm = Choose_Second_Arm(T,mu,N,CW_candidate,b_vector[first_arm], first_arm,RMED_flag)
        reward = np.random.binomial(1, P[first_arm, second_arm])
        # Update matrices and regret incurred
        W[first_arm, second_arm] = W[first_arm, second_arm] + reward
        N[first_arm, second_arm] += 1
        if first_arm != second_arm:
            W[second_arm, first_arm] = W[second_arm, first_arm] + 1 - reward
            N[second_arm,first_arm] += 1
        Regret_vec[t] = inst_regret_mat[first_arm, second_arm]
        mu = W / N

        # Find current CW candidate (minimizing I_vec in the paper)
        only_winners = mu <= 0.5
        np.fill_diagonal(only_winners, False)
        I_vec = np.sum(N * KL_div_half(mu) * only_winners, axis=1)
        CW_candidate = np.where(I_vec == np.min(I_vec))[0]
        if len(CW_candidate) > 1:
            CW_candidate = CW_candidate[np.random.randint(0, len(CW_candidate))]
        CW_candidate = int(CW_candidate.item())
        candidate_vec[t] = CW_candidate
        # Update lists
        L_C = np.delete(L_C, 0)
        for i in np.setdiff1d(np.arange(0, K), L_C):
            if I_vec[i] - np.min(I_vec) <= np.log(t + 1) + f:
                if not np.isin(i, L_N):
                    L_N = np.append(L_N, i)
        t = t + 1

    r = Regret_vec[0:T_stop]
    candidate_vec = candidate_vec[0:T_stop]
    candidate_vec=candidate_vec.astype(int)
    return r, candidate_vec