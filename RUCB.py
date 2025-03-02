import numpy as np
import warnings

# Single-player RUCB without a coin flip, assuming 0 is CW.
# inputs: P is a (K,K) preference matrix, alpha is RUCB parameter, T is time horizon
# outputs: r is a (T,) array of instantaneous regret, candidate_vec is a (T,) vector of candidate CW at each round
def RUCB(P,alpha,T):
    K = P.shape[0]
    W = np.zeros((K, K))
    B = []
    C = []
    candidate_vec = np.zeros(T)
    r = np.zeros((1, T))
    for t in range(T):
        # Decide arms a_c,a_d to draw
        U = W / (W + np.transpose(W))+np.sqrt(alpha*np.log(t+1)/(W+np.transpose(W)))
        U[np.isinf(U)] = 1
        U[np.isnan(U)] = 1
        U[np.eye(K) == 1] = 0.5
        C = np.where(np.sum(U >= 0.5, axis=1) == K)[0]
        if len(B) > 0 and np.sum(C == B) == 0:
            B = []
        if len(C) == 0:
            a_c = np.random.randint(0, K) # Random number in [0,...,K-1]
        elif len(C) == 1:
            B = C
            a_c = B
        elif len(C) > 1:
            if len(B) == 0:
                a_c = C[np.random.randint(0,len(C))]
            else:
                a_c = B
        a_d = np.where(U[:, a_c] == np.max(U[:, a_c]))[0]
        if len(a_d) > 1:
            if len(np.where(a_d == a_c)[0]) > 1:
                a_d[np.where(a_d == a_c)[0]] = []
            a_d = a_d[np.random.randint(0,len(a_d))]
        # Candidate vector - B if not empty, otherwise random in C
        if len(B)>0:
            candidate_vec[t] = B
        else:
            candidate_vec[t] = C[np.random.randint(0,len(C))]

        # Obtaining reward and updating winners matrix
        result = np.random.binomial(1, P[a_c, a_d])
        W[a_c, a_d] = W[a_c, a_d] + result
        W[a_d, a_c] = W[a_d, a_c] + 1 - result

        # Instantaneous regret
        r[0, t] = 0.5 * (P[0, a_c] - 0.5) + 0.5 * (P[0, a_d] - 0.5)
    return r, candidate_vec
