import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Decide on the second arm for each player.
# inputs: U is a (K,K,M) matrix of floats (UCB matrix for all players)
#         first_arms is a (M,) vector representing each player's first chosen arm
# outputs: a (M,) vector representing each player's second chosen arm
def RUCB_decision_second_arm(U,first_arms):
    M = U.shape[2]
    U_fa = U[:,first_arms,np.arange(M)]
    return np.argmax(np.random.random(U_fa.shape) *(U_fa==np.max(U_fa,axis=0)), axis=0)

# Given past observations of all players (through W,B), each player selects two arms in this round
# inputs: W is a (K,K,M) matrix of integers
#         alpha is a scalar (float)
#         B is a (M,) vector containing integers in [-1,...,K-1]
#         B_rec_mat is (K,M) vector containing integers in [0,1]
#         t - an integer time index
# outputs: first_arms, second_arms are (M,) vectors representing each player's first and second chosen arms
#          B of the same size (updated)
def RUCB_decision(W,alpha,t,B,B_rec_mat):
    K = W.shape[0]
    M = W.shape[2]

    # Defining UCB and winner matrix
    first_arms,second_arms = np.full((M,),-1,dtype=int), np.full((M,),-1,dtype=int)
    U = W / (W + np.transpose(W, (1, 0, 2))) + np.sqrt(alpha * np.log(t + 1) / (W + np.transpose(W, (1, 0, 2))))
    U[np.isinf(U)] = 1
    U[np.isnan(U)] = 1
    U[np.arange(K), np.arange(K), :] = 0.5
    C = np.zeros((K,M),dtype = int)
    C[np.sum(U >= 0.5, axis=1) == K] = 1 # 1 for arms in C[:,m] and 0 otherwise
    B_rec_updated = C*B_rec_mat
    # Draw arms
    # players for which B[m] exists but is not in C[:,m]
    B_msk=B[B!=-1]
    B_msk[C[B_msk,B!=-1] == 0]=-1
    B[B!=-1] = B_msk

    # players for which C[:,m]=0
    mask1 = np.sum(C,axis=0)==0
    first_arms[mask1] = np.random.randint(0, K,size=len(first_arms[mask1]))

    # Players for which C[:,m] has only one arm
    mask2 = np.sum(C,axis=0)==1
    if True in mask2:
        C_msk = C[:,mask2]>=0.5 # Boolean matrix
        new_arms = np.repeat(np.arange(K)[:, None], C_msk.shape[1], axis=1)
        B[mask2] = new_arms[C_msk].copy()
        first_arms[mask2] = new_arms[C_msk].copy()

    # Change B to a recommended arm
    if np.any(B_rec_updated) != 0:
        maskBB = np.sum(B_rec_updated,axis=0)>=1 #V
        B_msk = B_rec_updated[:,maskBB].copy()
        armsB = np.argmax(np.random.random(B_msk.shape) * B_msk, axis=0)
        B[maskBB] = armsB

    # Players for which C[:,m] has several arms
    mask3 = np.sum(C,axis=0)>1
    mask4 = mask3&(B!=-1)
    mask5 = mask3&(B==-1)
    first_arms[mask4]= B[mask4].copy()
    C_msk5 = C[:,mask5].copy()
    arms = np.argmax(np.random.random(C_msk5.shape) *C_msk5, axis=0)
    first_arms[mask5] = arms.copy()

    # Draw second arm
    second_arms = RUCB_decision_second_arm(U,first_arms)
    return first_arms, second_arms, B
# In a given round, update local observations of all players using delayed communication
# inputs: W is a (K,K,M) matrix of integers representing winning counts for all players
#         sp_mat is a (M,M) graph shortest-lengths matrix resulting from the shortest_lengths function
#         draw_mat is a (M,gamma,2) matrix such that draw_mat[:,:,0] represents the first arm choices
#           of all M players in all previous gamma time steps. draw_mat[:,:,1] is for the second arm.
#         reward_mat is a (M,gamma) matrix representing the rewards of draws from all M players in all previous gamma time steps.
# outputs: a (K,K,M) matrix of integers which is the updated W matrix
def local_MP(W,sp_mat, draw_mat, reward_mat):
    # Definitions
    gamma = draw_mat.shape[1]
    sp_mat_tmp = sp_mat.copy()
    sp_mat_tmp[sp_mat_tmp >= gamma] = -1
    M = draw_mat.shape[0]
    K = W.shape[0]
    W_add = np.zeros((K,K,M)) # Addition to the current W matrix
    rewards_to_add = reward_mat[np.arange(M), sp_mat_tmp]
    rewards_to_add[sp_mat_tmp<0] = -1
    draws_to_add = draw_mat[np.arange(M), sp_mat_tmp, :]
    draws_to_add[sp_mat_tmp<0, :] = -1

    # Flatten into 1D vectors
    first_arms, second_arms = np.reshape(draws_to_add[...,0],int(M**2)), np.reshape(draws_to_add[...,1],int(M**2))
    rewards = np.reshape(rewards_to_add,int(M**2))
    arm_ind = np.reshape(np.repeat(np.arange(0,M)[:,None],M,axis=1),int(M**2))

    # Substitute into W_add only relevant (positive) data
    msk = first_arms >= 0
    first_arms, second_arms, rewards, arm_ind = first_arms[msk], second_arms[msk], rewards[msk], arm_ind[msk]
    rewards_inv = 1-rewards
    arms, indices = np.unique(np.vstack((first_arms, second_arms,arm_ind)), axis=1, return_inverse=True)
    rewards = np.bincount(indices, weights=rewards)
    rewards_inv = np.bincount(indices, weights=rewards_inv)
    W_add[arms[0, :], arms[1, :], arms[2, :]] += rewards
    W_add[arms[1, :], arms[0, :], arms[2, :]] += rewards_inv
    return W + W_add

# Message-Passing RUCB without a coin flip, assuming 0 is CW.
# inputs: P is a (K,K) preference matrix, E is a (M,M) graph adjacency matrix,
#         alpha is RUCB parameter, T is time horizon, gamma (integer in [0,...,D]) is the decay parameter,
#         D is the communication graph diameter,
#         sp_mat is a (M,M) graph shortest-lengths matrix resulting from the shortest_lengths function
#         rec_flag is True if we use CW recommendations, otherwise False.
# outputs: a (M,T) matrix of floats representing the accumulated regret for each player at each round
def MP_RUCB(P,E,alpha,T,gamma,D,sp_mat,rec_flag):
    # Definitions
    K = P.shape[0]
    M = E.shape[0]
    if gamma > D:
        raise ValueError(f'gamma should not be larger than the diameter, which is {D}')
    if gamma == 0:
        gamma += 1 # Just to handle the non-communication case
    W = np.zeros((K, K, M)) # winning matrix
    draws_mat, reward_mat = np.zeros((M,gamma,2), np.intp), np.zeros((M,gamma))
    draws_matB, results_matB = np.zeros((M, gamma, 2), np.intp), np.zeros((M, gamma))
    B = np.full((M,), -1) # candidate for each player
    B_rec_mat = np.zeros((K, M))
    r = np.zeros((M, T)) # instantaneous regret
    roll_flag = np.int32(0) # Enables to save some calculations

    # Main loop
    t = 0
    while t<= T-1:
        # Select arms to draw
        first_arm, second_arm, B = RUCB_decision(W, alpha, t, B,B_rec_mat)
        maskB = first_arm == second_arm
        # No need to update anything, proceed to next round
        if ((t > gamma) & np.all(maskB == True) & (np.all(B == 0)) & np.all((second_arm == 0) & (first_arm == 0)) & np.all(
                (draws_mat[:, 0:(gamma - 1), 0] == 0) & (draws_mat[:, 0:(gamma - 1), 1] == 0)) & np.all(
            (draws_matB[:, 0:gamma, 0] == 0) & (draws_matB[:, 0:gamma, 1] == 0))):
            roll_flag = 1
            t += 1
        # If we do not use recommendations, the non-update condition is simpler
        elif (rec_flag==0)&(np.all((second_arm==0)&(first_arm==0))&np.all((draws_mat[:, 0:(gamma-1), 0]==0)&(draws_mat[:, 0:(gamma-1), 1]==0))):
            roll_flag = 1
            t += 1
        # Draw and Update
        else:
            # Draw
            good_event_no_draw = first_arm == second_arm #  if true - no need to draw, no new information
            if False in good_event_no_draw:
                reward = np.random.binomial(1, P[first_arm, second_arm])
            else:
                reward = np.zeros(M)
            current_draws = np.zeros((M,2))
            current_draws[:,0]=first_arm
            current_draws[:,1]=second_arm
            current_rewards = reward
            r[:, t] = 0.5 * (P[0, first_arm] - 0.5) + 0.5 * (P[0, second_arm] - 0.5)

            # Communication and updating statistics - W
            if roll_flag == 1: # Update the rounds we skipped
                draws_mat = np.zeros((M,gamma,2),dtype= np.intp)
                reward_mat = np.zeros((M,gamma))
            else: # Just roll
                draws_mat = np.roll(draws_mat, 1, axis=1)
                reward_mat = np.roll(reward_mat, 1, axis=1)
            draws_mat[:,0,:] = current_draws
            reward_mat[:,0] = current_rewards
            good_event_no_lMP = ~(draws_mat[:, :, 0] == draws_mat[:, :, 1]) # If all false - no need to update W
            if np.any(good_event_no_lMP):
                W = local_MP(W, sp_mat, draws_mat[:, :np.min([t + 1, gamma])], reward_mat[:, :np.min([t + 1, gamma])])

            # Updating recommendations with local_MP
            if rec_flag == 1:
                B_rec = np.full((M,), -1)
                B_rec[maskB] = B[maskB].copy()
                current_drawsB, current_resultsB = np.zeros((M, 2), np.intp), np.zeros(M)
                current_drawsB[:, 0] = B_rec.copy()
                current_drawsB[:, 1] = B_rec.copy()
                current_drawsB[~maskB, 0] = 0
                current_drawsB[~maskB, 1] = 1
                current_resultsB[maskB] = 1

                if roll_flag == 1:
                    draws_matB = np.zeros((M, gamma, 2), dtype=np.intp)
                    results_matB = np.zeros((M, gamma))
                else:
                    draws_matB = np.roll(draws_matB, 1, axis=1)
                    results_matB = np.roll(results_matB, 1, axis=1)
                draws_matB[:, 0, :] = current_drawsB
                results_matB[:, 0] = current_resultsB

                B_rec_mat_tmp = local_MP(np.zeros((K, K, M)), sp_mat, draws_matB[:, :np.min([t + 1, gamma])],
                                           results_matB[:, :np.min([t + 1, gamma])])
                diag = np.diagonal(B_rec_mat_tmp, axis1=0, axis2=1)
                mask_final = diag > 0
                B_rec_mat = np.where(mask_final, 1, 0)
                B_rec_mat = B_rec_mat.transpose(1, 0)

            roll_flag = 0
            t += 1

    Regret = np.cumsum(r,axis=1)
    return Regret
