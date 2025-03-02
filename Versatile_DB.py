import numpy as np
import cvxpy as cp

# Solves the optimization problem introduced by the tsallis_inf algorithm at each round.
# inputs: L is a (1,K) loss matrix, lr_coefficient is the learning rate coefficient (integer), t is the current round (integer)
# outputs: w is a (K,) array of the non-normalized arm weights corresponding to the optimal solution
def solve_tsallis_inf(L, lr_coeff,t):
    K = L.shape[1]

    # Define variables
    w = cp.Variable(K, nonneg=True)

    # Objective function to maximize
    obj = cp.Maximize(cp.sum(cp.multiply(-L, w)) + ((4*cp.sqrt(t)) / lr_coeff) * cp.sum(cp.sqrt(w)))

    # Constraints: w lies in the probability simplex
    constraints = [cp.sum(w) == 1]

    # Define problem and solve
    problem = cp.Problem(obj, constraints)
    problem.solve()

    return w.value

# Single-player Versatile-DB, assuming 0 is the CW.
# inputs: P is a (K,K) preference matrix, T is time horizon,
#         lr_coeff is the learning rate coefficient, where the learning rate at round t is defined as \eta_t = lr_coeff/sqrt(t)
# outputs: a (1,T) matrix of floats representing the accumulated regret at each round
def VDB(P,T, lr_coeff):
    K = P.shape[0]
    Lp = np.zeros((1,K)) # Loss vector of player +
    Ln = np.zeros((1,K)) # Loss vector of player -
    r = np.zeros((1, T)) # Instantaneous regret
    for t in range(T):
        # Calculate OMD weights for each player
        wp = solve_tsallis_inf(Lp, lr_coeff,t+1)
        wn = solve_tsallis_inf(Ln, lr_coeff,t+1)

        # Sample from the weights
        pp = (wp / np.sum(wp)).flatten() # Normalized weights for player +
        pn =(wn / np.sum(wn)).flatten() # Normalized weights for player -
        kp = np.random.choice(np.arange(0, K), p=pp) # Sampled arm for player +
        kn = np.random.choice(np.arange(0, K), p=pn) # Sampled arm for player -
        if kn == kp:
            result = 0.5
        else:
            result = np.random.binomial(1, P[kp, kn])
        r[0, t] = 0.5 * (P[0, kp] - 0.5) + 0.5 * (P[0, kn] - 0.5)

        # printing progress (since this is a long process)
        print_times = T//100 if T > 100 else 1
        if np.mod(t, print_times) == 0:
            print("Running the VDB algorithm at round t=", t)

        # Update the losses
        lp = np.zeros((1,K)) # importance weighted instantaneous loss vector of player +
        lp[0,kp] = (1-result)/pp[kp]
        ln = np.zeros((1,K)) # importance weighted instantaneous loss vector of player -
        ln[0,kn] = result/pn[kn]

        Lp = Lp + lp
        Ln = Ln + ln

    Regret = np.cumsum(r, axis=1)
    return Regret