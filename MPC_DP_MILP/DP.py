from os import fdatasync
import numpy as np


def DP_optimal_state_action(nd, nc, s_star, Mb, Q_dot, k, N):
    """
    Generate optimal state-action pairs using dynamic programming.

    Args:
        s_star (tuple): The target state, represented as a tuple (T_a^d, T_b^d).
        Mb (float): The maximum allowed cost.
        Q_dot (list): A list of Q values, for l = k, ..., k + N - 1.
        k (int): The starting index.
        N (int): The number of steps to consider.

    Returns:
        tuple: A tuple containing the cost, next state, and optimal control input, for all states s and l = k, ..., k + N - 1.
    """

    S = generate_set_of_state(nd)
    U = generate_set_of_action(nc)
    # Initialize cost function
    V  = {s: [0] * (k+N+1) for s in S}    
    U_star = {s: [U[0]] * (k+N+1) for s in S} 
    NS = {s: [(0,0)] * (k+N+1)  for s in S} 
    for s in S:
        V[s][k+N] = psi(s - s_star)

    # Initialize feasible state set
    S = [0*(k+N+1)]
    S[k+N] = [s_star]

    # Initialize index
    l = k + N - 1

    while l >= k:
        S_l = []
        for s in S:
            V_min = Mb
            for u in U:
                s_next, Wc = F(s, u, Q_dot[l])
                if s_next in S[l+1] and constraints_satisfied(s_next, s, u, Q_dot[l]):
                    S_l.append(s_next)
                    V_cur = c(l)*Wc + V[s_next][l+1]
                    if V < V_min:
                        V[s][l] = V_cur
                        U_star[s][l] = u
                        NS[s][l] = s_next
                        V_min = V
        S[l] = S_l
        l -= 1

    return V, NS, U_star


def psi(s,s_star):
    """Quadratic penalty for the deviation between s and s_star

    Args:
        s_star (tuple): The target state, represented as a tuple (T_a^d, T_b^d) \in S.
        s (tuple): The state, represented as a tuple (T_a^d, T_b^d).

    """

    return (s[0] - s_star[0])**2 + (s[1] - s_star[1])**2

def generate_set_of_state(Tcs, Tcr, nd):
    """Generates the set of states : tuple of temperature Ta,Tb

    Args:
        Tcs (float): cold supply temperature
        Tcr (float): cold return temperature
        nd (int): number of discretized levels for temperature

    Returns:
        S: the set of states
    """
    #Cas particulier nd =1 pour le moment    
    S = set()
    if nd==1:
        S = {(Tcr,Tcs)}
    else:

        Ta_range = np.linspace(Tcs,Tcr,nd)
        Tb_range = np.linspace(Tcs,Tcr,nd)
        for Ta in Ta_range:
            for Tb in Tb_range:
                if Ta>Tb:
                    S.add((round(Ta),round(Tb)))
    return S

def generate_set_of_action(nc):
    """
    Get the action set for the given number of chillers.

    Args:
        nc (int): The number of chillers.

    Returns:
        list: The action set, as a list of binary vectors.
    """
    U = []
    for i in range(2**nc):
        u = [int(x) for x in bin(i)[2:]]
        u = [0] * (nc - len(u)) + u
        U.append(u)
    return U



def F(s, u, Q_dot):
    """
    Transition function for the chiller and TES dynamics.

    Args:
        s (tuple): The current state, represented as a tuple (T_a^d, T_b^d).
        u (list): The control input, a binary vector representing the ON/OFF set-points for the chillers.
        k (int); the step time considered
        Tcr, Tcs, Qc : the temperature supply/return and the demand at time k
    Returns:
        tuple: The next state, represented as a tuple (T_a^d, T_b^d).
    """
    nc = len(u)
    cp_w = 4.18
    mci = [120,120,139,284,237,237,284]
    a = [54.67,54.67,80.17,143.6,165.7,109.2,168.8]
    b = [318.3,318.3,225,666.1,592.8,786.7,332.1]
    T_cs = 277.6
    T_cr = 287.6
    fac = 1.1740
    fbc = 2.222
    fda = 4.138
    fdb = 1.599
    Uc = 0.0283
    Ud = 0.0197


    # Compute variables m_dot_c, Q_dot_c, and W_c
    m_dot_c = sum([mci[i]*u[i] for i in range(nc)])
    DT = (T_cr-T_cs)
    Q_dot_c = m_dot_c * cp_w * DT
    W_c = sum([u[i]*(DT*a[i] + b[i]) for i in range(nc)])

    # Determine TES operating mode
    if Q_dot_c >= Q_dot:
        sigma = 1
    else:
        sigma = 0

    # Compute variables Ts, Tr, m_dot, and m_dot_t
    if sigma == 1:
        # TES is being charged
        Ts = T_cs
        m_dot_t = (Q_dot_c-Q_dot)/(cp_w*(s[0]-Ts))
        m_dot = m_dot_c - m_dot_t
        Tr = (m_dot_c * T_cr - m_dot_t * s[0]) / m_dot
        Ta_d_next = s[0] + (fac*m_dot_t*cp_w*(s[1]-s[0]) + Uc*A*(s[1]-S[0]))*dt/(rho*cp_w)
        Tb_d_next = s[1] + (fbc*m_dot_t*cp_w*(Ts-s[1]) + Uc*A*(s[0]-S[1]))*dt/(rho*cp_w)

    else:
        # TES is being discharged
        Tr = T_cr 
        m_dot_t = (Q_dot-Q_dot_c)/(cp_w*(Tr-s[1]))
        m_dot = m_dot_c + m_dot_t
        Ts = (m_dot_c * T_cs + m_dot_t * s[1]) / m_dot
        Ta_d_next = s[0] + (fad*m_dot_t*cp_w*(Tr-s[0]) + Uc*A*(s[1]-S[0]))*dt/(rho*cp_w)
        Tb_d_next = s[1] + (fbc*m_dot_t*cp_w*(s[0]-s[1]) + Uc*A*(s[0]-S[1]))*dt/(rho*cp_w)
  
    return (Ta_d_next, Tb_d_next), W_c


def c(l):
    cost = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2, 2, 2, 2, 1, 1]
    return cost[l]*1e4

def constraints_satisfied(s_next, s, u, Q_dot[l]):
    #TODO