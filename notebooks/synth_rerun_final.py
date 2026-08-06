"""Re-execution of the Section-3 synthetic example (E2).
Configuration = the manuscript's stated parameters (mu_a=2, sd_a=0.2, mu_b=2, sd_b=0.3,
c=0.2; 500 target draws; failure set a>=2.25 & b>=1.0; delta=0.15) combined with the
proposal-construction and running-statistics code of the author's own
2D_Importance_Sampling.ipynb (cells 3/47): 60th-pct tail fit, cov x2, 20% mean nudge,
5% defensive mass; weights = f / (0.95 g + 0.05 f) at the sampled points. Seeds fixed
(np.random.seed(0); default_rng(1)) exactly as in the notebook."""
import numpy as np
from scipy.stats import multivariate_normal, gaussian_kde

# 1) target data + f (cell 3, verbatim)
n_target = 500
np.random.seed(0)
a_t = np.random.normal(2.0, 0.2, n_target)
b_t = np.random.normal(2.0, 0.3, n_target) * (a_t * 0.2)
XY = np.vstack([a_t, b_t])
mu_f, cov_f = XY.mean(axis=1), np.cov(XY)
f_dist = multivariate_normal(mu_f, cov_f)

# proposal g (cell 3, verbatim knobs)
ta_, tb_ = np.percentile(a_t, 60), np.percentile(b_t, 60)
tail = XY[:, (a_t > ta_) & (b_t > tb_)]
mu_g, cov_g = tail.mean(axis=1), np.cov(tail)
eps_def, scale_g, lambda_mu = 0.05, 2.0, 0.2
cov_g = cov_g * scale_g + np.eye(2) * 1e-6
mu_g = (1 - lambda_mu) * mu_g + lambda_mu * mu_f
g_dist = multivariate_normal(mu_g, cov_g)

# draws (cell 3): 500 f-draws (naive), 500 mixture proposal draws
rng = np.random.default_rng(1)
F = rng.multivariate_normal(mu_f, cov_f, size=500).T
n_main = int((1 - eps_def) * 500)
G = np.hstack([rng.multivariate_normal(mu_g, cov_g, size=n_main).T,
               rng.multivariate_normal(mu_f, cov_f, size=500 - n_main).T])

# weights at the sampled points (cell 3's q_mix)
pts = G.T
f_at = f_dist.pdf(pts); g_at = g_dist.pdf(pts)
w = f_at / ((1 - eps_def) * g_at + eps_def * f_at)
ESS = w.sum()**2 / (w**2).sum()

# failure indicators (manuscript thresholds)
A_THR, B_THR = 2.25, 1.0
hF = ((F[0] >= A_THR) & (F[1] >= B_THR)).astype(float)
hG = ((G[0] >= A_THR) & (G[1] >= B_THR)).astype(float)
print(f"failures: naive {int(hF.sum())}/500 (p={hF.mean():.4f}); proposal draws {int(hG.sum())}/500")
p_is = np.sum(w * hG) / np.sum(w)
print(f"IS self-normalized p = {p_is:.4f}; ESS = {ESS:.1f}/500")

# running stats (cell 47, verbatim)
def run_naive(h):
    h = np.asarray(h, float); n = np.arange(1, h.size + 1)
    c1, c2 = np.cumsum(h), np.cumsum(h**2)
    mean = c1 / n
    vm = np.maximum(c2 / n - mean**2, 0.0)
    var = np.empty_like(vm); var[0] = 0.0
    var[1:] = vm[1:] * (n[1:] / (n[1:] - 1))
    return mean, np.sqrt(var) / np.sqrt(n), n
def run_is(h, w):
    h = np.asarray(h, float); w = np.asarray(w, float)
    S1, S2 = np.cumsum(w), np.cumsum(w**2)
    SwH, Sw2H = np.cumsum(w * h), np.cumsum(w**2 * h)
    mu = SwH / np.maximum(S1, 1e-12)
    num = np.maximum(Sw2H * (1 - 2 * mu) + mu**2 * S2, 0.0)
    se = np.sqrt(num) / np.maximum(S1, 1e-12)
    return mu, se, np.arange(1, h.size + 1)

mM, sM, nM = run_naive(hF)
mI, sI, nI = run_is(hG, w)
k = 1 / np.sqrt(0.15)                      # delta = 0.15 as in the manuscript
P_RISK = 0.05
def crossing(m, s, n, nmin=20):
    ub = m + k * s
    ok = np.where((ub <= P_RISK) & (n >= nmin) & (np.cumsum(m * n > 0) > 0))[0]
    # require at least one failure observed so the band is meaningful
    first_fail_M = np.argmax(m > 0)
    ok = [i for i in ok if i >= first_fail_M]
    return int(ok[0]) + 1 if len(ok) else None
nsN = crossing(mM, sM, nM); nsI = crossing(mI, sI, nI)
print(f"crossing (upper Chebyshev band <= p_risk={P_RISK}, delta=0.15, k={k:.2f}):")
print(f"  naive MC n* = {nsN};  IS n* = {nsI};  ratio = {nsN/nsI:.2f}x" if nsN and nsI else f"  naive {nsN}, IS {nsI}")

# density triple at the manuscript's tail point (2.25, 1.00), notebook's KDE conventions
pt = np.array([[A_THR], [B_THR]])
exact = float(f_dist.pdf(pt.T))
kde_f = gaussian_kde(F)                     # direct KDE on f-samples
kde_gu = gaussian_kde(G)                    # unweighted KDE on proposal draws
kde_gw = gaussian_kde(G, weights=w)         # IS-weighted KDE
vals = dict(exact=exact, direct=float(kde_f(pt)), unweighted=float(kde_gu(pt)), weighted=float(kde_gw(pt)))
print(f"\ndensity at ({A_THR},{B_THR}): exact f = {exact:.2f}")
for kk in ('direct', 'unweighted', 'weighted'):
    print(f"  {kk:10s} = {vals[kk]:.2f}   log-error = {np.log(vals[kk]/exact):+.2f}")
