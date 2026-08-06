"""Task 8 — rare-regime re-threshold test (revision_plan Task 8 / manuscript F7).
Raise the leak-fraction threshold on the recorded pool outputs until p ~ 1e-3,
then (a) check pilot feasibility vs the binomial design rule, and
(b) replay the corrected workflow (DGSA_Light h1+h2 selection, defensive mixture)
for pilots that DO start. No new MRST. Evaluation = FULL pool."""
import sys, time, numpy as np
from pathlib import Path
import revision_tools as RT
from seed_study import kde_pdf, normal_ref_bw, dgsa_light_select

ROOT = RT.find_repo_root()
sys.path.append(str(ROOT))
P = ROOT/'data'/'real_paper_data'/'pool_empirical'
leak = np.load(P/'pool_leak_fraction_full.npy')
pool_scores = np.load(ROOT/'notebooks'/'pool_scores_275.npy')
pool_h2 = np.load(ROOT/'notebooks'/'pool_h2_scores.npy')
N = len(leak)
ALPHA, EPS_REL, DELTA, NSEL = 0.5, 0.4, 0.05, 7

def kde_pdf_chunked(x, data, bw, chunk=1500):
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        out[i:i+chunk] = kde_pdf(x[i:i+chunk], data, bw)
    return out

# ---- 1) threshold sweep to p ~ 1e-3
print(f"baseline: threshold 0.001 -> rate {np.mean(leak>0.001):.4f} ({int((leak>0.001).sum())}/{N})")
for target in (1e-2, 5e-3, 2e-3, 1e-3):
    nfail = max(2, int(round(target*N)))
    t = np.sort(leak)[-nfail]
    print(f"  target p~{target:g}: threshold {t:.5f} -> rate {np.mean(leak>=t):.2e} ({int((leak>=t).sum())}/{N})")
nfail = int(round(1e-3*N))                       # 10 failures
thr = np.sort(leak)[-nfail]
h1r = (leak >= thr).astype(float)
p_true = h1r.mean()
print(f"\nRARE REGIME: threshold {thr:.5f} (leaked fraction), {int(h1r.sum())}/{N} failures, p_true={p_true:.2e}")

# ---- 2) pilot feasibility, analytic vs empirical
print("\nfeasibility P(>=2 pilot failures):")
rng = np.random.default_rng(2026)
for k in (100, 300, 500, 1000, 2000, 4742):
    q = 1-(1-p_true)**k - k*p_true*(1-p_true)**(k-1)
    emp = np.mean([h1r[rng.permutation(N)[:k]].sum() >= 2 for _ in range(600)])
    print(f"  k={k:5d}: binomial {q:.3f} | empirical {emp:.3f}")

# ---- 3) replay for pilots that start (k=300 and k=1000)
def replay(k, want, boots=1000, max_seeds=3000):
    got, tried, res = 0, 0, []
    n_naive = (1/DELTA)*(p_true*(1-p_true))/(EPS_REL*p_true)**2
    for sd in range(1, max_seeds+1):
        pilot = np.random.default_rng(10_000+sd).permutation(N)[:k]
        hp = h1r[pilot]; tried += 1
        if hp.sum() < 2: continue
        got += 1
        sp = pool_scores[pilot]
        resp = np.concatenate([hp.reshape(-1,1), pool_h2[pilot][:, :47]], axis=1)
        sens = dgsa_light_select(sp, resp, n_select=NSEL, boots=boots)
        cols = sp[:, sens]
        mu, sd_ = cols.mean(0), cols.std(0, ddof=1); sd_[sd_==0]=1
        fz = (cols-mu)/sd_; gz = fz[hp==1]
        bw = normal_ref_bw(fz)
        Se = (pool_scores[:, sens]-mu)/sd_
        fe = kde_pdf_chunked(Se, fz, bw); ge = kde_pdf_chunked(Se, gz, bw)
        gp = ALPHA*fe+(1-ALPHA)*ge; w = fe/gp
        var_is = np.mean(h1r*w) - p_true**2
        n_is = (1/DELTA)*max(var_is,1e-15)/(EPS_REL*p_true)**2
        r = gp/fe; r=r/r.sum()
        dr = np.random.default_rng(sd).choice(N, size=k, replace=True, p=r)
        p_is = float(np.sum(w[dr]*h1r[dr])/np.sum(w[dr]))
        cov = abs(p_is-p_true) <= (1/np.sqrt(DELTA))*np.sqrt(max(var_is,1e-15)/k)
        res.append((sd, int(hp.sum()), float(k/np.mean(w)), p_is, n_naive/n_is, bool(cov)))
        print(f"    k={k} seed={sd}: fails={int(hp.sum())} ESS={k/np.mean(w):.0f} "
              f"p_IS={p_is:.2e} speedup={n_naive/n_is:.2f}x CI-covers={bool(cov)}", flush=True)
        if got >= want: break
    print(f"  k={k}: {got} started / {tried} tried (empirical start rate {got/tried:.3f})")
    return res

print("\nreplay k=300 (started pilots only):")
r300 = replay(300, want=8)
print("\nreplay k=1000:")
r1000 = replay(1000, want=8)

for k, res in ((300, r300), (1000, r1000)):
    if not res: continue
    a = np.array([(x[2], x[3], x[4], x[5]) for x in res])
    print(f"\nk={k} summary over {len(res)} started replicates:")
    print(f"  ESS median {np.median(a[:,0]):.0f}; p_IS median {np.median(a[:,1]):.2e} "
          f"(truth {p_true:.2e}); speedup median {np.median(a[:,2]):.2f}x "
          f"[{a[:,2].min():.2f},{a[:,2].max():.2f}]; CI covers truth {a[:,3].mean():.0%}")
