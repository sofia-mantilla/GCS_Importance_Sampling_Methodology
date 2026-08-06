"""
seed_study.py — Multi-seed replication & batch-size study (FAST, streaming version).

Answers Editor "replicate / vary batch size" and R2-1(i),(iii), Q1: the DISTRIBUTION of
n*, ESS and p_IS across independent random pilot batches drawn from the existing pool.
No new MRST runs (see METHOD_REFERENCE.md / revision_plan.md).

WHAT YOU SEE WHILE IT RUNS
  * a progress bar (tqdm) over replicates,
  * one printed line per replicate,
  * each row appended to seed_study_results.csv IMMEDIATELY — open it any time to see
    partial results, and if you kill the run the completed rows are already saved.

SPEED KNOBS (top of file)
  * QUICK=True  -> surfaces-only (5 of the 7 sensitive PCs), fewer seeds, fewer DGSA boots.
                  Runs in ~1-2 min and is enough to confirm everything works.
  * QUICK=False -> full study (adds the 8 GB porosity PCA, all seeds/batch sizes).
  * EVAL_N      -> evaluate the IS identities on a random EVAL_N subset of the pool
                  (a few thousand is plenty; the metrics are pool averages).
"""
from __future__ import annotations
import numpy as np, pandas as pd, time, csv
from pathlib import Path
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

import revision_tools as RT
ROOT = RT.find_repo_root()

# ------------------------------------------------------------------------- ---- config
POOL_DIR = Path("/Users/sofiamantillasalas/Library/CloudStorage/GoogleDrive-sofiams@stanford.edu/"
                "My Drive/Research_Python/Structural_trapping/Jupyter_Python_Research/"
                "Inputs_jupyter/empirical_10000_cluster_2025")
POOL_SURF = POOL_DIR / "surfaces.npy"     # (9993,184,104)  
POOL_PHI  = POOL_DIR / "porosity.npy"     # (108470,9993)  — only loaded if QUICK=False

QUICK        = False                       # QUICK=True -> surfaces-only fast sanity
BATCH_SIZES  = [300] if QUICK else [100, 300, 500]
N_SEEDS      = 5    if QUICK else 25
N_SURF_PC, N_PHI_PC = 9, 266
N_SELECT     = 7
ALPHA        = 0.5                         # balance-heuristic / combine share (not tuned)
EVAL_N       = 3000                        # pool subsample for evaluating the IS identities
# SELECT: "dgsa_light" = paper's DGSA on h1+h2 (faithful; needs pool CO2height) — RECOMMENDED.
#         "fast"       = h1-only CDF screening (fast sanity; picks spurious PCs in full space).
SELECT       = "fast" if QUICK else "dgsa_light"
DGSA_BOOTS   = 150 if QUICK else 3000     # paper's value; stabilises PC selection (slower)
EPS_REL, DELTA = 0.4, 0.05
OUT_CSV      = Path("seed_study_results.csv")

# ----------------------------------------------------------------------------- fast helpers
def pca_scores_fast(fields_2d, k):
    """fields_2d: (features, samples). Randomized PCA -> (samples, k) scores. Much faster
    than full SVD on the 8 GB porosity."""
    from sklearn.decomposition import PCA
    data = np.asarray(fields_2d).T                      # (samples, features)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    return pca.fit_transform(data)

def kde_pdf(x, data, bw):
    z = (x[:, None, :] - data[None, :, :]) / bw
    return np.prod(np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * bw), axis=2).mean(axis=1)

def normal_ref_bw(d):
    n, dim = d.shape
    return 1.06 * d.std(axis=0, ddof=1) * n ** (-1.0 / (dim + 4))

def dgsa_select(scores, h1, n_select=N_SELECT, boots=DGSA_BOOTS, seed=0):
    """Fast binary-response DGSA: per-PC L1 distance between the failure-conditioned CDF and
    the marginal CDF, normalised by a bootstrap null. Returns indices of the top PCs.
    (Swap in DGSA_Light for the paper-exact selection; this is the fast screening used here.)"""
    rng = np.random.default_rng(seed)
    n = len(h1); n1 = int(h1.sum()); idx1 = np.where(h1 == 1)[0]
    def l1(col, members):
        xs = np.sort(col)
        ref = np.searchsorted(xs, xs, side="right") / n
        sub = np.searchsorted(np.sort(col[members]), xs, side="right") / len(members)
        return np.abs(sub - ref).mean()
    obs = np.array([l1(scores[:, j], idx1) for j in range(scores.shape[1])])
    null = np.empty((scores.shape[1], boots))
    for b in range(boots):
        m = rng.choice(n, n1, replace=False)
        for j in range(scores.shape[1]):
            null[j, b] = l1(scores[:, j], m)
    sens = obs / np.maximum(np.quantile(null, 0.95, axis=1), 1e-12)
    return np.argsort(-sens)[:n_select]

def dgsa_light_select(params, responses, n_select=N_SELECT, boots=DGSA_BOOTS):
    """Paper-faithful selection via DGSA_Light (responses = h1 + h2 scores)."""
    import sys; sys.path.append(str(ROOT))
    from DGSA_Light.DGSA_light import DGSA_light
    labels = [f"PC{i}" for i in range(params.shape[1])]
    d = DGSA_light(params, responses, ParametersNames=labels, n_clsters=3, n_boots=boots)
    return np.argsort(d.iloc[:, 0].values)[::-1][:n_select]

# ----------------------------------------------------------------------------- one replicate
def run_replicate(pilot_idx, pool_scores, pool_h1, eval_idx, k, seed, label, pool_h2=None):
    s_pilot = pool_scores[pilot_idx]; h_pilot = pool_h1[pilot_idx]
    n_fail = int(h_pilot.sum())
    if n_fail < 2:
        return dict(label=label, k=k, seed=seed, n_fail=n_fail, status="too_few_failures")
    if SELECT == "dgsa_light" and pool_h2 is not None:
        resp = np.concatenate([h_pilot.reshape(-1, 1), pool_h2[pilot_idx][:, :47]], axis=1)
        sens = dgsa_light_select(s_pilot, resp)
    else:
        sens = dgsa_select(s_pilot, h_pilot, seed=seed)
    cols = s_pilot[:, sens]
    mu, sd = cols.mean(0), cols.std(0, ddof=1); sd[sd == 0] = 1
    fz = (cols - mu) / sd; gz = fz[h_pilot == 1]
    bw = normal_ref_bw(fz)
    Se = (pool_scores[eval_idx][:, sens] - mu) / sd
    fe = kde_pdf(Se, fz, bw); ge = kde_pdf(Se, gz, bw)
    gp = ALPHA * fe + (1 - ALPHA) * ge
    w = fe / gp; he = pool_h1[eval_idx]; p_ref = float(he.mean())
    ess_frac = 1.0 / np.mean(w)
    var_is = np.mean(he * w) - p_ref**2
    n_naive = (1/DELTA) * (p_ref*(1-p_ref)) / (EPS_REL*p_ref)**2
    n_is = (1/DELTA) * max(var_is, 1e-12) / (EPS_REL*p_ref)**2
    r = gp / fe; r = r / r.sum()
    dr = np.random.default_rng(seed).choice(len(eval_idx), size=k, replace=True, p=r)
    p_is = float(np.sum(w[dr]*he[dr]) / np.sum(w[dr]))
    return dict(label=label, k=k, seed=seed, n_fail=n_fail, p_is=p_is, p_ref=p_ref,
                ess=float(ess_frac*k), cv=float(w.std()/w.mean()),
                max_share=float(w.max()/w.sum()), n_star_is=float(n_is),
                n_star_naive=float(n_naive), speedup=float(n_naive/n_is),
                sensitive_pcs=list(map(int, sens)), status="ok")

# ----------------------------------------------------------------------------- driver
def main():
    t0 = time.time()
    pool = RT.load_pool(ROOT); pool_h1 = pool["h1"]; N = len(pool_h1)
    print(f"[{time.time()-t0:5.1f}s] pool outcomes: {N} runs, {int(pool_h1.sum())} failures", flush=True)

    cache = Path("pool_scores_surf.npy") if QUICK else Path("pool_scores_275.npy")
    if cache.exists():
        pool_scores = np.load(cache)
        print(f"[{time.time()-t0:5.1f}s] loaded cached pool scores {pool_scores.shape} from {cache.name}", flush=True)
    else:
        print(f"[{time.time()-t0:5.1f}s] PCA on pool surfaces (1.4 GB)...", flush=True)
        surf = np.load(POOL_SURF, mmap_mode="r")
        surf_scores = pca_scores_fast(np.asarray(surf).reshape(N, -1).T, N_SURF_PC)
        pool_scores = surf_scores
        if not QUICK:
            print(f"[{time.time()-t0:5.1f}s] PCA on pool porosity (8 GB, slow)...", flush=True)
            phi = np.load(POOL_PHI, mmap_mode="r")
            phi_scores = pca_scores_fast(np.asarray(phi), N_PHI_PC)
            pool_scores = np.hstack([surf_scores, phi_scores])
        np.save(cache, pool_scores)
        print(f"[{time.time()-t0:5.1f}s] pool scores ready {pool_scores.shape}, cached to {cache.name}", flush=True)

    pool_h2 = None
    if SELECT == "dgsa_light":
        h2cache = Path("pool_h2_scores.npy")
        if h2cache.exists():
            pool_h2 = np.load(h2cache)
            print(f"[{time.time()-t0:5.1f}s] loaded pool h2 scores {pool_h2.shape}", flush=True)
        else:
            print(f"[{time.time()-t0:5.1f}s] PCA on pool CO2height for DGSA h2...", flush=True)
            h2 = np.load(POOL_DIR / "CO2height.npy", mmap_mode="r")
            pool_h2 = pca_scores_fast(np.asarray(h2), 47); np.save(h2cache, pool_h2)

    rng = np.random.default_rng(0)
    eval_idx = rng.choice(N, size=min(EVAL_N, N), replace=False)

    # build the replicate list: paper anchor + random seeds × batch sizes
    jobs = [("paper", pool["paper_pos0"], 300, 0)]
    for kbatch in BATCH_SIZES:
        for sd in range(1, N_SEEDS + 1):
            jobs.append((f"k{kbatch}_s{sd}", np.random.default_rng(sd).permutation(N)[:kbatch], kbatch, sd))

    cols = ["label","k","seed","n_fail","p_is","p_ref","ess","cv","max_share",
            "n_star_is","n_star_naive","speedup","sensitive_pcs","status"]
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); fh.flush()
        for label, idx, kbatch, sd in tqdm(jobs, desc="replicates"):
            row = run_replicate(idx, pool_scores, pool_h1, eval_idx, kbatch, sd, label, pool_h2)
            wr.writerow({c: row.get(c, "") for c in cols}); fh.flush()       # <- partial results on disk
            if row["status"] == "ok":
                print(f"  {label:10s} k={kbatch} fails={row['n_fail']:2d}  "
                      f"ESS={row['ess']:5.1f}  p_IS={row['p_is']:.4f}  speedup={row['speedup']:.1f}x", flush=True)
            else:
                print(f"  {label:10s} k={kbatch} -> {row['status']}", flush=True)

    df = pd.read_csv(OUT_CSV)
    ok = df[df.status == "ok"]
    print(f"\n[{time.time()-t0:5.1f}s] done. {len(ok)} successful replicates -> {OUT_CSV}")
    if len(ok):
        print("\nDistribution (median [min, max]):")
        for col in ["n_fail","ess","p_is","speedup"]:
            print(f"  {col:9s}: {ok[col].median():.3f}  [{ok[col].min():.3f}, {ok[col].max():.3f}]")

if __name__ == "__main__":
    main()
