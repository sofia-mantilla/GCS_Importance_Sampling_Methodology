"""
check_corrected_weights.py  —  RUN THIS FIRST.

A small, fast sanity check (a few seconds, no big files) that:
  1. loads the importance-sampling objects already in the repo,
  2. shows the BUGGY weight result vs the CORRECTED result (the ESS jump),
  3. shows that the defensive-mixture fix repairs it,
  4. prints the variance numbers in the form Reviewer 2 asked for,
  5. compares Chebyshev vs Wilson/CLT sample sizes (Reviewer 2 point 4).

It only reads files that are already in data/ — nothing to download, nothing to configure.
Run it to confirm the toolbox works on your machine before running the bigger seed_study.py.
"""
import numpy as np
import pickle
import revision_tools as RT

ROOT = RT.find_repo_root()
IS_DIR = ROOT / "data" / "Inputs_for_Final_Script_after_IS_Simulation"
NMC_DIR = ROOT / "data" / "Inputs_for_Final_Script_after_Naive_Simulation"

print("Repo root :", ROOT)
print("Loading the importance-sampling objects from:", IS_DIR.name, "\n")

# --- the proposal/target densities and the IS draws used in the paper ---
with open(IS_DIR / "f_m_Inputs_for_Final_Script_after_IS_Simulation.pkl", "rb") as fh:
    f_m = pickle.load(fh)          # target  f(m): KDE on ALL 300 sensitive scores
with open(IS_DIR / "g_m_Inputs_for_Final_Script_after_IS_Simulation.pkl", "rb") as fh:
    g_m = pickle.load(fh)          # proposal g(m): KDE on the 9 leakage scores
RS  = np.load(IS_DIR / "RS_failure_sensitive_scores.npy")   # the IS draws (300 x 7)
hIS = np.load(IS_DIR / "h_1_models_leaking_IS.npy")         # leak/no-leak of those draws
hN  = np.load(NMC_DIR / "h_1_models_leaking.npy")           # naive 300 leak outcomes

# ---------------------------------------------------------------- 1) BUGGY vs CORRECTED
# Buggy (as in the notebook cell): evaluate pdfs at their training points, index by
# integer-cast scores, average per-dimension ratios. We reproduce its effect via the
# stored arrays just to show the contrast.
f_train = f_m.pdf(); g_train = g_m.pdf()
idx = RS.astype(int)
w_buggy = np.mean(f_train[idx] / g_train[idx], axis=1)

# Correct: joint-density weight w = f(m)/g(m) evaluated AT the IS draws.
w_correct = RT.is_weights_correct(f_m, g_m, RS)

print("=" * 64)
print("1)  BUGGY vs CORRECTED importance weights")
print("=" * 64)
print(f"  BUGGY     ESS = {RT.ess(w_buggy):6.1f} / 300   (this is ~what the paper reported)")
print(f"  CORRECT   ESS = {RT.ess(w_correct):6.1f} / 300   (the honest single-run value)")
d = RT.weight_diagnostics(w_correct)
print(f"  corrected weight CV = {d['CV']:.1f},  largest single draw = {d['max_share']*100:.0f}% of all weight")

# ---------------------------------------------------------------- 2) the fix
print("\n" + "=" * 64)
print("2)  Defensive-mixture fix:  g' = alpha*f + (1-alpha)*g")
print("=" * 64)
for alpha in [0.0, 0.1, 0.2, 0.3]:
    w = RT.defensive_mixture_weights(f_m, g_m, RS, alpha=alpha)
    print(f"  alpha = {alpha:.1f}  ->  ESS = {RT.ess(w):6.1f} / 300")
print("  (alpha = 0 is the current/buggy-free single run; alpha ~ 0.2 restores a usable ESS)")

# ---------------------------------------------------------------- 3) variance (R2 point 2)
print("\n" + "=" * 64)
print("3)  Variance, in the form Reviewer 2 asked to see")
print("=" * 64)
vr = RT.variance_report(hN, w_correct, hIS)
print(f"  Var[h]      (Bernoulli var of the indicator) = {vr['Var_h_naive']:.4f}   <- what the paper")
print(f"                                                                    mislabeled as Var[p_hat]")
print(f"  Var[p_hat]  naive  = p(1-p)/L                = {vr['Var_phat_naive']:.3e}")
print(f"  Var[p_hat]  IS     = ESS-adjusted            = {vr['Var_phat_is']:.3e}")

# ---------------------------------------------------------------- 4) bounds (R2 point 4)
print("\n" + "=" * 64)
print("4)  Sample size to certify the estimate (Reviewer 2 point 4)")
print("=" * 64)
p = float(hN.mean())
nC = RT.n_required(p, 0.4, 0.05, "chebyshev")
nL = RT.n_required(p, 0.4, 0.05, "clt")
print(f"  Chebyshev (what the paper uses): n = {nC:.0f}")
print(f"  CLT / Wilson (tighter)        : n = {nL:.0f}   ->  {nC/nL:.1f}x fewer samples")
print(f"  Wilson 95% CI for the naive 9/300: {tuple(round(x,4) for x in RT.wilson_interval(9,300))}")

print("\nDone. If you see this, the toolbox works on your machine.")
