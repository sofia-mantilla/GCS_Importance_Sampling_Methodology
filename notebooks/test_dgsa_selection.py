"""
test_dgsa_selection.py — Why did the full-space seed study collapse?

Compares three ways of selecting the 7 sensitive PCs on a few pilot batches, then measures
the resulting speed-up for each:
  (A) fast screening  — the h1-only CDF-distance DGSA used in seed_study.py
  (B) DGSA_Light h1   — the paper's DGSA_Light, responses = h1 only
  (C) DGSA_Light h1+h2 — the paper's DGSA_Light, responses = h1 + h2 (what the paper actually uses)

If (C) picks mostly SURFACE PCs and recovers ~3x while (A) picks spurious porosity PCs and
collapses, then the collapse is a selection artefact, not a property of the method.

FIRST RUN builds two caches (pool_scores_275.npy ~ one-time 20-30 min PCA, pool_h2_scores.npy),
then reuses them instantly. Run:  python test_dgsa_selection.py
"""
import numpy as np, time
from pathlib import Path
import revision_tools as RT
import seed_study as S          # reuse kde_pdf, normal_ref_bw, pca_scores_fast, config

ROOT = RT.find_repo_root()
NB = 300                        # DGSA_Light bootstraps for the test (paper uses 3000)
N_PILOTS = 5                    # random k=300 pilots to test, plus the paper anchor

def build_pool_scores():
    cache = Path("pool_scores_275.npy")
    if cache.exists(): return np.load(cache)
    print("building pool_scores_275.npy (one-time PCA)...", flush=True)
    surf = np.load(S.POOL_SURF, mmap_mode="r")
    N = surf.shape[0]
    ss = S.pca_scores_fast(np.asarray(surf).reshape(N, -1).T, S.N_SURF_PC)
    phi = np.load(S.POOL_PHI, mmap_mode="r")
    ps = S.pca_scores_fast(np.asarray(phi), S.N_PHI_PC)
    out = np.hstack([ss, ps]); np.save(cache, out); return out

def build_h2_scores():
    cache = Path("pool_h2_scores.npy")
    if cache.exists(): return np.load(cache)
    print("building pool_h2_scores.npy (CO2 height PCA)...", flush=True)
    h2 = np.load(S.POOL_DIR / "CO2height.npy", mmap_mode="r")
    sc = S.pca_scores_fast(np.asarray(h2), 47); np.save(cache, sc); return sc

def dgsa_light_select(params, responses, n_select=7):
    import sys; sys.path.append(str(ROOT))
    from DGSA_Light.DGSA_light import DGSA_light
    labels = [f"PC{i}" for i in range(params.shape[1])]
    d = DGSA_light(params, responses, ParametersNames=labels, n_clsters=3, n_boots=NB)
    order = np.argsort(d.iloc[:, 0].values)[::-1]
    return order[:n_select]

def speedup_for(sens, pilot_idx, pool_scores, pool_h1, eval_idx, k, seed):
    s_pilot = pool_scores[pilot_idx]; h_pilot = pool_h1[pilot_idx]
    cols = s_pilot[:, sens]; mu, sd = cols.mean(0), cols.std(0, ddof=1); sd[sd == 0] = 1
    fz = (cols - mu) / sd; gz = fz[h_pilot == 1]; bw = S.normal_ref_bw(fz)
    Se = (pool_scores[eval_idx][:, sens] - mu) / sd
    fe = S.kde_pdf(Se, fz, bw); ge = S.kde_pdf(Se, gz, bw)
    gp = S.ALPHA * fe + (1 - S.ALPHA) * ge
    w = fe / gp; he = pool_h1[eval_idx]; p = float(he.mean())
    var_is = np.mean(he * w) - p**2
    n_naive = (1/S.DELTA) * (p*(1-p)) / (S.EPS_REL*p)**2
    n_is = (1/S.DELTA) * max(var_is, 1e-12) / (S.EPS_REL*p)**2
    return n_naive / n_is

def main():
    pool = RT.load_pool(ROOT); h1 = pool["h1"]; N = len(h1)
    ps = build_pool_scores(); h2 = build_h2_scores()
    eval_idx = np.random.default_rng(0).choice(N, size=min(S.EVAL_N, N), replace=False)
    pilots = [("paper", pool["paper_pos0"])]
    for sd in range(1, N_PILOTS + 1):
        pilots.append((f"k300_s{sd}", np.random.default_rng(sd).permutation(N)[:300]))

    print(f"\n{'pilot':9s} {'method':12s} {'#surf':>5s} {'#phi':>4s} {'speedup':>8s}   selected PCs")
    print("-" * 78)
    for label, idx in pilots:
        hp = h1[idx]
        methods = {
            "A fast":       S.dgsa_select(ps[idx], hp),
            "B light-h1":   dgsa_light_select(ps[idx], hp.reshape(-1, 1)),
            "C light-h1h2": dgsa_light_select(ps[idx], np.concatenate([hp.reshape(-1,1), h2[idx][:, :47]], axis=1)),
        }
        for m, sens in methods.items():
            nsurf = int(np.sum(np.asarray(sens) < 9)); nphi = 7 - nsurf
            su = speedup_for(np.asarray(sens), idx, ps, h1, eval_idx, 300, 0)
            print(f"{label:9s} {m:12s} {nsurf:5d} {nphi:4d} {su:8.2f}   {sorted(int(i) for i in sens)}")
        print()

if __name__ == "__main__":
    main()
