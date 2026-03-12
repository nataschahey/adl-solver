import numpy as np

# =============================================================================
# Leverage helpers
# =============================================================================
def compute_leverage(x, q, p_tau, p_entry, m):
    n = len(m)
    ell = np.zeros(n)
    for i in range(n):
        remaining = q[i] - x[i]
        notional  = p_tau @ np.abs(remaining)
        E_i       = q[i] @ (p_entry[i] - p_tau) + m[i]
        ell[i]    = notional / E_i if E_i > 0 else np.inf
    return ell

def compute_factor_leverage(x, q, p_tau, p_entry, m,v):
    """Uses additive model: v = sqrt(lam1)*u1, factor leverage = v^T(q_i-x_i)/E_i."""
    n = len(m)
    ell_v = np.zeros(n)
    for i in range(n):
        remaining = q[i] - x[i]
        E_i = q[i] @ (p_entry[i] - p_tau) + m[i]
        ell_v[i] = v @ remaining / E_i if E_i > 0 else np.inf
    return ell_v


def compute_equities(q, p_entry, p_tau, m):
    n = q.shape[0]
    return np.array([q[i] @ (p_entry[i] - p_tau) + m[i] for i in range(n)])