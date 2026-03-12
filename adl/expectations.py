import numpy as np
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermegauss

_GH_DEGREE = 20
_gh_nodes, _gh_weights = hermegauss(_GH_DEGREE)

_GH_DEGREE = 20
_gh_nodes, _gh_weights = hermegauss(_GH_DEGREE)

# =============================================================================
# Expectation layer
# =============================================================================

def expected_shortfall_one_factor(x_i, q_i, p_entry_i, p_tau, m_i, v):
    E_i = float(q_i @ (p_entry_i - p_tau) + m_i)
    c_i = float(v @ (q_i - x_i))

    if abs(c_i) < 1e-12:
        return max(0.0, -E_i), np.zeros_like(x_i)

    if c_i > 0:
        z   = E_i / c_i
        val = c_i * norm.pdf(z) - E_i * norm.cdf(-z)
        # d(val)/d(c_i) = phi(z)  [Leibniz rule]
        # d(c_i)/d(x_i) = -v
        grad = norm.pdf(z) * (-v)
    else:
        z   = E_i / c_i          # negative
        val = -c_i * norm.pdf(z) - E_i * norm.cdf(z)
        # By symmetry: d(val)/d(c_i) = -phi(z)
        # d(c_i)/d(x_i) = -v
        grad = -norm.pdf(z) * (-v)

    return max(val, 0.0), grad


def expected_shortfall_gbm(x_i, q_i, p_entry_i, p_tau, m_i, sigma, Delta, rho):
    E_i = float(q_i @ (p_entry_i - p_tau) + m_i)
    r_i = q_i - x_i

    L = np.array([[1.0, 0.0],
                  [rho, np.sqrt(1.0 - rho**2)]])

    drift     = -0.5 * sigma**2 * Delta
    diffusion =  sigma * np.sqrt(Delta)

    nodes, weights = _gh_nodes, _gh_weights
    K = len(nodes)

    # shape (K, K, 2): all pairs of quadrature nodes
    xi_j = nodes[:, None, None] * np.array([1.0, 0.0])  # (K,1,2)
    xi_k = nodes[None, :, None] * np.array([0.0, 1.0])  # (1,K,2)
    xi   = xi_j + xi_k                                   # (K,K,2)

    # correlated normals: Z = L @ xi, shape (K,K,2)
    Z = xi @ L.T                                         # (K,K,2)

    # GBM prices: shape (K,K,2)
    p_T = p_tau * np.exp(drift + diffusion * Z)          # (K,K,2)

    # equity: shape (K,K)
    equity    = E_i + (p_tau - p_T) @ r_i               # (K,K)
    shortfall = np.maximum(0.0, -equity)                 # (K,K)

    # weights: shape (K,K)
    W = weights[:, None] * weights[None, :]              # (K,K)

    val  = float(np.sum(W * shortfall))

    # gradient: shape (2,)
    # grad += w * (p_tau - p_T) where shortfall > 0
    mask = (shortfall > 0).astype(float)                 # (K,K)
    grad = ((W * mask)[:, :, None] * (p_tau - p_T)).sum(axis=(0,1))

    return val, grad