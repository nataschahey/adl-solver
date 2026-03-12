import numpy as np
from scipy.optimize import minimize, brentq
from .expectations import expected_shortfall_one_factor, expected_shortfall_gbm
from .utils import compute_equities

# =============================================================================
# Box bounds
# =============================================================================

def _box_bounds(Q, q_i):
    lo = np.where(Q >= 0, 0.0,               np.minimum(0.0, q_i))
    hi = np.where(Q >  0, np.maximum(0.0, q_i), 0.0)
    return lo, hi


# =============================================================================
# Per-account subproblem: min_{x_i in Y_i} E[sigma_i(x_i)] + lambda^T x_i
# =============================================================================

def _solve_account(i, lam, q, p_entry, p_tau, m, Q,
                   mode, v=None, sigma=None, Delta=None, rho=None):
    from scipy.optimize import minimize
    q_i, p_i, m_i = q[i], p_entry[i], m[i]
    lo, hi = _box_bounds(Q, q_i)
    bounds  = list(zip(lo, hi))

    def obj_and_grad(x_i):
        if mode == 'one_factor':
            es, g = expected_shortfall_one_factor(x_i, q_i, p_i, p_tau, m_i, v)
        else:
            es, g = expected_shortfall_gbm(x_i, q_i, p_i, p_tau, m_i,
                                           sigma, Delta, rho)
        return es + lam @ x_i, g + lam

    x0  = np.clip(0.5 * (lo + hi), lo, hi)
    res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                   bounds=bounds,
                   options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
    return res.x

def _solve_all(lam, Q, q, p_entry, p_tau, m,
               mode, v=None, sigma=None, Delta=None, rho=None):
    n = q.shape[0]
    X = np.zeros_like(q, dtype=float)
    for i in range(n):
        X[i] = _solve_account(i, lam, q, p_entry, p_tau, m, Q,
                               mode=mode, v=v, sigma=sigma,
                               Delta=Delta, rho=rho)
    return X
    
def _bisect_1d(resid_fn, lam_init_k, tol):
    """Find lam_k such that resid_fn(lam_k) = 0. Returns (lo, hi, lam_star)."""
    r0 = resid_fn(lam_init_k)

    if abs(r0) < tol:
        return lam_init_k, lam_init_k, lam_init_k

    if r0 > 0:
        lo = lam_init_k
        hi = max(lam_init_k + 1.0, 1.0)
        for _ in range(80):
            if resid_fn(hi) < 0:
                break
            hi = hi * 2.0 + 1.0
        else:
            return None, None, None
    else:
        hi = lam_init_k
        lo = min(lam_init_k - 1.0, -1.0)
        for _ in range(80):
            if resid_fn(lo) > 0:
                break
            lo = lo * 2.0 - 1.0
        else:
            return None, None, None

    lam_star = brentq(resid_fn, lo, hi, xtol=tol, rtol=tol, maxiter=200)
    return lo, hi, lam_star


def _compute_primal_obj(X, q, p_entry, p_tau, m,
                        mode, v=None, sigma=None, Delta=None, rho=None):
    obj = 0.0
    for i in range(len(m)):
        if mode == 'one_factor':
            es, _ = expected_shortfall_one_factor(
                X[i], q[i], p_entry[i], p_tau, m[i], v)
        else:
            es, _ = expected_shortfall_gbm(
                X[i], q[i], p_entry[i], p_tau, m[i],
                sigma, Delta, rho)
        obj += es
    return obj


def solve_adl(Q, q, p_entry, p_tau, m,
                               mode='one_factor',
                               v=None, sigma=None, Delta=None, rho=None,
                               lam_init=None,
                               tol=1e-8, max_iter=100, verbose=False):

    n, d    = q.shape
    support = np.where(np.abs(Q) > 1e-12)[0]

    if len(support) == 0:
        return np.zeros((n, d)), 0.0, np.zeros(d), True

    lam = np.zeros(d) if lam_init is None else lam_init.copy()
    if not (np.all(np.isfinite(lam)) and np.all(np.abs(lam) < 1e12)):
        lam = np.zeros(d)

    kw = dict(mode=mode, v=v, sigma=sigma, Delta=Delta, rho=rho)

    def get_gradient(lam_):
        X = _solve_all(lam_, Q, q, p_entry, p_tau, m, **kw)
        return X, X.sum(axis=0) - Q  # X and gradient

    for iteration in range(max_iter):
        X, grad = get_gradient(lam)
        grad_norm = np.linalg.norm(grad[support])

        if verbose:
            print(f"iter {iteration}: |g|={grad_norm:.3e}, lam={lam}")

        if grad_norm < tol:
            break

        # 1D line search along gradient direction
        direction = grad  # ascent direction for q*(lambda)

        def line_obj(t):
            # directional derivative of q* at lam + t*direction
            # = g(lam + t*direction)^T direction
            _, g_new = get_gradient(lam + t * direction)
            return g_new[support] @ direction[support]

        # bracket t: find t_hi where line_obj changes sign
        t_lo, t_hi = 0.0, 1.0
        r_lo = line_obj(t_lo)  # = grad^T grad > 0 at t=0
        for _ in range(60):
            if line_obj(t_hi) < 0:
                break
            t_hi *= 2.0
        else:
            # no sign change: take full gradient step
            lam = lam + t_hi * direction
            continue

        t_star = brentq(line_obj, t_lo, t_hi,
                        xtol=tol, rtol=tol, maxiter=100)
        lam = lam + t_star * direction

    X, grad = get_gradient(lam)

    # proportional rescaling
    x_star = X.copy()
    for k in support:
        total = x_star[:, k].sum()
        if abs(total) > 1e-12 and abs(Q[k]) > 1e-12:
            x_star[:, k] *= Q[k] / total
    for i in range(n):
        lo_i, hi_i = _box_bounds(Q, q[i])
        x_star[i] = np.clip(x_star[i], lo_i, hi_i)

    final_resid = np.linalg.norm((x_star.sum(axis=0) - Q)[support])
    obj = _compute_primal_obj(x_star, q, p_entry, p_tau, m, **kw)
    success = final_resid < tol * 100

    if verbose:
        print(f"final: |g|={final_resid:.3e}, obj={obj:.6f}")

    return x_star, obj, lam, success

def solve_water_filling(Q, q, p_entry, p_tau, m, v, tol=1e-12, verbose=False):
    """
    Analytical water-filling solution from Theorem 2.

    Factor leverage: ell_i^(v)(x_i) = v^T(q_i - x_i) / E_i
    Budget:          sum_i E_i * ell_i^(v) = v^T(sum_i q_i - Q) =: L_v
    Water-filling:   ell_i^* = clip(eta*, ell_lo_i, ell_hi_i)
    Recover x_i:     x_i chosen in feasible set to achieve ell_i^*

    Parameters
    ----------
    p_T_scenarios : if provided, compute primal objective at x_wf
                    (use one-factor scenarios for fair comparison)

    Returns
    -------
    x_wf   : (n, d) water-filling allocation
    ell_wf : (n,)   factor leverages after water-filling
    eta_star: scalar water level
    obj    : primal objective (nan if p_T_scenarios not provided)
    """
    n, d = q.shape
    E    = compute_equities(q, p_entry, p_tau, m)

    # Per-account feasible factor leverage range
    # x_i^k in [x_lo_ik, x_hi_ik] determined by Q
    x_lo = np.zeros((n, d))
    x_hi = np.zeros((n, d))
    for i in range(n):
        for k in range(d):
            if Q[k] > 0:
                x_lo[i, k] = 0.0
                x_hi[i, k] = max(0.0, q[i, k])
            elif Q[k] < 0:
                x_lo[i, k] = min(0.0, q[i, k])
                x_hi[i, k] = 0.0
            else:
                x_lo[i, k] = 0.0
                x_hi[i, k] = 0.0

    # ell_lo_i: minimum achievable factor leverage (max reduction)
    # ell_hi_i: maximum achievable factor leverage (no reduction)
    ell_lo = np.array([v @ (q[i] - x_hi[i]) / E[i] for i in range(n)])
    ell_hi = np.array([v @ (q[i] - x_lo[i]) / E[i] for i in range(n)])

    # Budget in factor space
    L_v = v @ (q.sum(axis=0) - Q)

    if verbose:
        print(f"L_v = {L_v:.4f}")
        print(f"ell_lo = {np.round(ell_lo,4)}")
        print(f"ell_hi = {np.round(ell_hi,4)}")
        print(f"sum E*ell_lo = {(E*ell_lo).sum():.4f}  "
              f"sum E*ell_hi = {(E*ell_hi).sum():.4f}")

    def budget_residual(eta):
        ell = np.clip(eta, ell_lo, ell_hi)
        return (E * ell).sum() - L_v

    sum_lo = (E * ell_lo).sum()
    sum_hi = (E * ell_hi).sum()

    if L_v <= sum_lo:
        eta_star = ell_lo.min() - 1.0
    elif L_v >= sum_hi:
        eta_star = ell_hi.max() + 1.0
    else:
        eta_lo = ell_lo.min() - 1.0
        eta_hi = ell_hi.max() + 1.0
        for _ in range(200):
            eta_mid = (eta_lo + eta_hi) / 2.0
            r = budget_residual(eta_mid)
            if abs(r) < tol * abs(L_v + 1):
                break
            if r > 0:
                eta_hi = eta_mid
            else:
                eta_lo = eta_mid
        eta_star = (eta_lo + eta_hi) / 2.0

    ell_wf = np.clip(eta_star, ell_lo, ell_hi)

    x_wf   = np.zeros((n, d))
    support = np.where(Q != 0)[0]

    for i in range(n):
        target_vTx = v @ q[i] - ell_wf[i] * E[i]   
        vTx_hi     = v @ x_hi[i]

        if len(support) == 1:
            # Single asset: x_i[k0] = target_vTx / v[k0]
            k0 = support[0]
            if abs(v[k0]) > 1e-14:
                x_wf[i, k0] = np.clip(target_vTx / v[k0],
                                       x_lo[i, k0], x_hi[i, k0])
        else:
            # Multi-asset: distribute proportionally to x_hi in v direction
            if abs(vTx_hi) > 1e-14:
                scale = target_vTx / vTx_hi
                x_wf[i] = np.clip(scale * x_hi[i],
                                   x_lo[i], x_hi[i])
            else:
                x_wf[i] = x_lo[i].copy()

    if verbose:
        ell_check = compute_factor_leverage(
            x_wf, q, p_entry, p_tau, m, v)
        print(f"eta* = {eta_star:.6f}")
        print(f"ell_wf = {np.round(ell_wf, 4)}")
        print(f"ell_check = {np.round(ell_check, 4)}")
        resid_clear = np.linalg.norm(x_wf.sum(axis=0)[support] - Q[support])
        print(f"clearing resid = {resid_clear:.3e}")

    obj = _compute_primal_obj(x_wf, q, p_entry, p_tau, m,
                              mode='one_factor', v=v)

    return x_wf, ell_wf, eta_star, obj

def solve_adl_fast2(Q, q, p_entry, p_tau, m,
              mode='one_factor',
              v=None, sigma=None, Delta=None, rho=None,
              lam_init=None,
              tol=1e-8, verbose=False):
    """
    Solve ADL expected-loss minimization via coordinate bisection + interpolation.

    Parameters
    ----------
    Q        : (d,)   aggregate reduction vector
    q        : (n,d)  positions
    p_entry  : (n,d)  entry prices
    p_tau    : (d,)   reference prices
    m        : (n,)   margins
    mode     : 'one_factor' or 'gbm'
    v        : (d,)   factor direction        (mode='one_factor')
    sigma    : (d,)   annualized vols         (mode='gbm')
    Delta    : float  horizon in years        (mode='gbm')
    rho      : float  correlation             (mode='gbm')
    lam_init : (d,)   warm-start dual variable
    tol      : float  bisection tolerance
    verbose  : bool

    Returns
    -------
    x_star  : (n,d)  optimal allocations
    obj     : float  primal objective
    lam     : (d,)   final dual variable (warm start for next call)
    success : bool
    """

    n, d    = q.shape
    support = np.where(np.abs(Q) > 1e-12)[0]

    if len(support) == 0:
        return np.zeros((n, d)), 0.0, np.zeros(d), True

    lam = np.zeros(d) if lam_init is None else lam_init.copy()
    if not (np.all(np.isfinite(lam)) and np.all(np.abs(lam) < 1e12)):
        lam = np.zeros(d)

    kw = dict(mode=mode, v=v, sigma=sigma, Delta=Delta, rho=rho)

    if len(support) == 1:
        k = support[0]

        def resid_1d(lam_k):
            lam_try = lam.copy()
            lam_try[k] = lam_k
            X = _solve_all(lam_try, Q, q, p_entry, p_tau, m, **kw)
            return (X.sum(axis=0) - Q)[k]

        lo, hi, lam_k_star = _bisect_1d(resid_1d, lam[k], tol)
        if lam_k_star is None:
            return np.zeros((n, d)), 0.0, lam, False

        lam[k] = lam_k_star
        x_star = _solve_all(lam, Q, q, p_entry, p_tau, m, **kw)

        # proportional rescaling to exactly hit Q
        total = x_star[:, k].sum()
        if abs(total) > 1e-12 and abs(Q[k]) > 1e-12:
            x_star[:, k] *= Q[k] / total

        # re-clip to box bounds
        for i in range(n):
            lo_i, hi_i = _box_bounds(Q, q[i])
            x_star[i]  = np.clip(x_star[i], lo_i, hi_i)

    elif len(support) == 2:
        k0, k1 = support[0], support[1]

        def inner_resid(lam_k1, lam_k0):
            lam_try = lam.copy()
            lam_try[k0] = lam_k0
            lam_try[k1] = lam_k1
            X = _solve_all(lam_try, Q, q, p_entry, p_tau, m, **kw)
            return (X.sum(axis=0) - Q)[k1]

        def solve_inner(lam_k0):
            _, _, lam_k1_star = _bisect_1d(
                lambda lk1: inner_resid(lk1, lam_k0), lam[k1], tol)
            return lam_k1_star

        def outer_resid(lam_k0):
            lam_k1_star = solve_inner(lam_k0)
            if lam_k1_star is None:
                return np.nan
            lam_try = lam.copy()
            lam_try[k0] = lam_k0
            lam_try[k1] = lam_k1_star
            X = _solve_all(lam_try, Q, q, p_entry, p_tau, m, **kw)
            return (X.sum(axis=0) - Q)[k0]

        lo0, hi0, lam_k0_star = _bisect_1d(outer_resid, lam[k0], tol)
        if lam_k0_star is None:
            return np.zeros((n, d)), 0.0, lam, False

        lam_k1_star = solve_inner(lam_k0_star)
        if lam_k1_star is None:
            return np.zeros((n, d)), 0.0, lam, False

        lam[k0] = lam_k0_star
        lam[k1] = lam_k1_star

        x_star = _solve_all(lam, Q, q, p_entry, p_tau, m, **kw)

        # proportional rescaling to exactly hit Q
        for k in support:
            total = x_star[:, k].sum()
            if abs(total) > 1e-12 and abs(Q[k]) > 1e-12:
                x_star[:, k] *= Q[k] / total

        # re-clip to box bounds
        for i in range(n):
            lo_i, hi_i = _box_bounds(Q, q[i])
            x_star[i]  = np.clip(x_star[i], lo_i, hi_i)

    else:
        raise NotImplementedError("Only d_active <= 2 supported.")

    final_resid = np.linalg.norm((x_star.sum(axis=0) - Q)[support])
    obj         = _compute_primal_obj(x_star, q, p_entry, p_tau, m, **kw)
    success     = final_resid < tol * 100

    if verbose:
        print(f"|g|={final_resid:.3e}, obj={obj:.6f}")

    return x_star, obj, lam, success