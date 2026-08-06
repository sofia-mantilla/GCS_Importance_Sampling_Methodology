"""
revision_tools.py — shared helpers for the CAGEO revision, built to REUSE the existing
repo pipeline (sklearn PCA, DGSA_Light, statsmodels MKDE) exactly as in
Final_Script_after_Naive_Simulation.ipynb / Final_Script_after_IS_Simulation.ipynb.

Both new deliverables import from here so there is ONE implementation of the workflow:
  * corrected_weights.ipynb      — fixes the IS-weight cell, reports estimator variance,
                                    Wilson/Clopper-Pearson vs Chebyshev (R2-2, R2-4)
  * seed_study.py                — multi-seed replication + batch-size ablation (R2-1, Editor)

Nothing here changes the published inputs or figures; it re-implements the same math so the
revision analyses are reproducible and consistent with the notebooks.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

def _erfinv(x):
    """scipy.special.erfinv if available (repo has scipy), else a numpy approximation."""
    try:
        from scipy.special import erfinv
        return float(erfinv(x))
    except Exception:
        a = 0.147
        ln = np.log(1 - x**2)
        t = 2 / (np.pi * a) + ln / 2
        return float(np.sign(x) * np.sqrt(np.sqrt(t**2 - ln / a) - t))

def _z(delta):
    """Two-sided normal quantile z_{1-delta/2}."""
    return np.sqrt(2) * _erfinv(1 - delta)

# ----------------------------------------------------------------------------- repo plumbing
def find_repo_root(max_up: int = 6) -> Path:
    """Same helper the notebooks use: locate the dir containing data/ and notebooks/."""
    p = Path.cwd().resolve()
    for _ in range(max_up + 1):
        if (p / "data").is_dir() and (p / "notebooks").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate repo root (looked for 'data' and 'notebooks').")

# Variance thresholds and selection count, matching the manuscript / naive notebook
VAR_THR_M1, VAR_THR_M2, VAR_THR_H2 = 0.97, 0.96, 0.99   # -> 9, 266, 47 PCs on the paper batch
N_SENSITIVE = 7
LEAK_THR = 0.001          # leaked-fraction failure criterion (0.1% of injected CO2)

# ----------------------------------------------------------------------------- PCA (sklearn, notebook convention)
def pca_scores(fields_2d, var_threshold=None, n_pc=None):
    """fields_2d: (n_features, n_samples) — same orientation the notebook transposes to.
    Returns (scores[:, :k], k). Mirrors: pca.fit_transform(np.transpose(m))."""
    from sklearn.decomposition import PCA
    data = np.transpose(fields_2d)                 # (samples, features)
    pca = PCA()
    scores = pca.fit_transform(data)
    if n_pc is None:
        ev = pca.explained_variance_ratio_
        k, c = 0, 0.0
        while c < var_threshold:
            c += ev[k]; k += 1
        n_pc = k
    return scores[:, :n_pc], n_pc

# ----------------------------------------------------------------------------- DGSA (reuse DGSA_Light)
def select_sensitive_pcs(m1_scores, m2_scores, h1, h2_scores,
                         n_select=N_SENSITIVE, n_clsters=3, n_boots=3000, root=None):
    """Faithful call to the repo's DGSA_Light, exactly as naive-notebook cells 32-37.
    Returns (sensitive_col_indices, dgsa_measures_df, PC_all_labels)."""
    import sys
    if root is None:
        root = find_repo_root()
    sys.path.append(str(root))
    from DGSA_Light.DGSA_light import DGSA_light

    x_m1, x_m2, x_h2 = m1_scores.shape[1], m2_scores.shape[1], h2_scores.shape[1]
    parameters = np.concatenate((m1_scores[:, :x_m1], m2_scores[:, :x_m2]), axis=1)
    responses  = np.concatenate((h1.reshape(-1, 1), h2_scores[:, :x_h2]), axis=1)
    labels = [f"PC{i}-surf" for i in range(1, x_m1 + 1)] + \
             [f"PC{i}-phi"  for i in range(1, x_m2 + 1)]
    dgsa = DGSA_light(parameters, responses, ParametersNames=labels,
                      n_clsters=n_clsters, n_boots=n_boots)
    order = np.argsort(dgsa.iloc[:, 0].values)[::-1]
    sens_idx = order[:n_select]
    return sens_idx, dgsa, labels

# ----------------------------------------------------------------------------- MKDE (statsmodels, notebook convention)
def build_fg(sensitive_scores_all, h1, root=None):
    """f_m on ALL standardized sensitive scores, g_m on the FAILURE subset with f's bandwidth
    — exactly naive-notebook cells 37-39. sensitive_scores_all: (n_samples, n_sensitive)."""
    import statsmodels.api as sm
    S = np.asarray(sensitive_scores_all, dtype="float64")
    S = (S - S.mean(axis=0)) / S.std(axis=0)            # stn_para_sen
    vt = "c" * S.shape[1]
    f_m = sm.nonparametric.KDEMultivariate(data=S, var_type=vt)
    bw = f_m.bw
    g_m = sm.nonparametric.KDEMultivariate(data=S[h1 == 1], var_type=vt, bw=bw)
    return f_m, g_m, bw, S

# ----------------------------------------------------------------------------- CORRECTED IS weights (the cell-4 fix)
def is_weights_correct(f_m, g_m, scores):
    """CORRECT joint-density importance weights w = f(m)/g(m), evaluated AT the given scores.
    Replaces the buggy `f_m.pdf()[scores.astype(int)]` + per-dim averaging in IS cell 4."""
    return f_m.pdf(np.atleast_2d(scores)) / g_m.pdf(np.atleast_2d(scores))

def defensive_mixture_weights(f_m, g_m, scores, alpha=0.2):
    """Weights under g' = alpha*f + (1-alpha)*g. Bounds weights at 1/alpha -> finite variance."""
    fe = f_m.pdf(np.atleast_2d(scores)); ge = g_m.pdf(np.atleast_2d(scores))
    return fe / (alpha * fe + (1 - alpha) * ge)

# ----------------------------------------------------------------------------- estimators & diagnostics
def ess(weights):
    w = np.asarray(weights, float)
    return w.sum()**2 / (w**2).sum()

def weight_diagnostics(weights):
    w = np.asarray(weights, float)
    return dict(ESS=float(ess(w)), ESS_frac=float(ess(w) / len(w)),
                CV=float(w.std() / w.mean()), max_share=float(w.max() / w.sum()))

def p_is_selfnorm(weights, h):
    w = np.asarray(weights, float)
    return float(np.sum(w * h) / np.sum(w))

def variance_report(h_naive, weights_is, h_is):
    """Returns the quantities R2 asked to be distinguished:
       Var[h]  = p(1-p)           (Bernoulli variance of the indicator — NOT the estimator var)
       Var[p_hat_naive] = p(1-p)/L
       Var[p_hat_IS]    = ESS-adjusted estimator variance."""
    p = float(np.mean(h_naive)); L = len(h_naive)
    w = np.asarray(weights_is, float); h = np.asarray(h_is, float)
    p_is = p_is_selfnorm(w, h)
    ess_is = ess(w)
    var_h = p * (1 - p)
    return dict(
        p_naive=p, p_is=p_is, L=L, ESS_is=float(ess_is),
        Var_h_naive=var_h,                         # the p(1-p) the paper mislabeled
        Var_phat_naive=var_h / L,                  # correct estimator variance, naive
        Var_phat_is=p_is * (1 - p_is) / ess_is,    # correct estimator variance, IS (ESS-adjusted)
        var_reduction=(var_h / L) / (p_is * (1 - p_is) / ess_is),
    )

# ----------------------------------------------------------------------------- confidence-bound comparison (R2-4)
def n_required(p, eps_rel=0.4, delta=0.05, method="chebyshev"):
    """Simulations needed for a relative half-width eps_rel*p at confidence 1-delta.
       'chebyshev' reproduces the paper's bound; 'clt'/'wilson' are the tighter alternatives."""
    p = float(p)
    if method == "chebyshev":
        k = 1.0 / np.sqrt(delta)                   # distribution-free
    elif method in ("clt", "wilson"):
        k = _z(delta)                              # normal quantile z_{1-delta/2}
    else:
        raise ValueError(method)
    # relative half-width h = eps_rel * p ; n = k^2 p(1-p)/h^2
    h = eps_rel * p
    return float(k**2 * p * (1 - p) / h**2)

def wilson_interval(k_succ, n, delta=0.05):
    z = _z(delta)
    phat = k_succ / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half

# ----------------------------------------------------------------------------- data loaders
def load_pool(root=None):
    """Pool outcomes + the paper-batch index map (assembled in data/real_paper_data)."""
    if root is None:
        root = find_repo_root()
    P = Path(root) / "data" / "real_paper_data" / "pool_empirical"
    return dict(
        h1=np.load(P / "pool_h1_full.npy"),
        leakfrac=np.load(P / "pool_leak_fraction_full.npy"),
        run_ids=np.loadtxt(P / "existing_indices.txt", dtype=int),
        paper_pos0=np.load(P / "paper_batch_pool_pos0.npy"),
    )
