"""
stress_tests.py — Eight pool-based stress tests of the seed-study / advisor-brief claims.
Pure numpy (runs anywhere); REUSES seed_study.py's cached pool scores and the per-seed
sensitive-PC selections stored in seed_study_results.csv, so no sklearn/statsmodels/DGSA
is needed and every replicate is reconstructed bit-identically to the seed study.

Tests (see docs/STRESS_TEST_RESULTS.md for the write-up):
  t1  Coverage of the Chebyshev-certified interval at the reported n* (IS and naive; + Wilson naive)
  t2  Adversarial bound comparison: naive-with-Wilson vs IS-with-Chebyshev (+ pilot cost)
  t3  Weight sanity identity E_f[g'/f]=1 (reference-free bug detector) + buggy-cell reproduction
  t4  Split-pool: evaluate on pool \ pilot (removes pilot double-use / part of circularity)
  t5  Self-normalized vs unnormalized estimator bias
  t6  Pilot feasibility: P(>=2 failures) vs pilot size, at p~0.029 and p~1e-3
  t7  Robustness sweeps on the paper config: KDE bandwidth x{0.5,1,2}, alpha {0.25,0.5,0.75}
  t8  PC-selection frequency across seeds + does speedup track selection overlap with paper's PCs

Usage:  python3 stress_tests.py prep      # reconstruct replicates, verify vs CSV, cache
        python3 stress_tests.py t1 ... t8 # individual tests (need prep cache)
        python3 stress_tests.py all
Outputs: stress_test_results/*.csv + printed summaries.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "stress_test_results"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "replicates_cache.npz"

EPS_REL, DELTA, ALPHA, EVAL_N, N_SELECT = 0.4, 0.05, 0.5, 3000, 7  # = seed_study.py
Z95 = 1.959963984540054  # z_{1-0.05/2}

# --------------------------------------------------------------------- shared (numpy) pieces
def kde_pdf(x, data, bw):
    z = (x[:, None, :] - data[None, :, :]) / bw
    return np.prod(np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * bw), axis=2).mean(axis=1)

def normal_ref_bw(d):
    n, dim = d.shape
    return 1.06 * d.std(axis=0, ddof=1) * n ** (-1.0 / (dim + 4))

def ess_kish(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w ** 2).sum()

def load_pool():
    P = ROOT / "data" / "real_paper_data" / "pool_empirical"
    return np.load(P / "pool_h1_full.npy"), np.load(P / "paper_batch_pool_pos0.npy")

def load_rows():
    rows = list(csv.DictReader(open(HERE / "seed_study_results.csv")))
    for r in rows:
        r["k"] = int(r["k"]); r["seed"] = int(r["seed"])
        if r["status"] == "ok":
            for c in ("p_is", "p_ref", "ess", "cv", "max_share", "n_star_is",
                      "n_star_naive", "speedup"):
                r[c] = float(r[c])
            r["sens"] = np.array(json.loads(r["sensitive_pcs"]), int)
    return rows

def reconstruct(row, pool_scores, pool_h1, eval_idx, paper_pos0,
                bw_mult=1.0, alpha=ALPHA, eval_override=None):
    """Bit-identical re-run of seed_study.run_replicate given the stored PC selection."""
    k, seed = row["k"], row["seed"]
    pilot_idx = paper_pos0 if row["label"] == "paper" else \
        np.random.default_rng(seed).permutation(len(pool_h1))[:k]
    s_pilot = pool_scores[pilot_idx]; h_pilot = pool_h1[pilot_idx]
    sens = row["sens"]
    cols = s_pilot[:, sens]
    mu, sd = cols.mean(0), cols.std(0, ddof=1); sd[sd == 0] = 1
    fz = (cols - mu) / sd; gz = fz[h_pilot == 1]
    bw = normal_ref_bw(fz) * bw_mult
    ev = eval_idx if eval_override is None else eval_override
    Se = (pool_scores[ev][:, sens] - mu) / sd
    fe = kde_pdf(Se, fz, bw); ge = kde_pdf(Se, gz, bw)
    gp = alpha * fe + (1 - alpha) * ge
    w = fe / gp; he = pool_h1[ev]; p_ref = float(he.mean())
    var_is = np.mean(he * w) - p_ref ** 2
    n_naive = (1 / DELTA) * (p_ref * (1 - p_ref)) / (EPS_REL * p_ref) ** 2
    n_is = (1 / DELTA) * max(var_is, 1e-12) / (EPS_REL * p_ref) ** 2
    r = gp / fe; r = r / r.sum()
    dr = np.random.default_rng(seed).choice(len(ev), size=k, replace=True, p=r)
    p_is = float(np.sum(w[dr] * he[dr]) / np.sum(w[dr]))
    return dict(pilot_idx=pilot_idx, w=w, he=he, r=r, p_ref=p_ref, p_is=p_is,
                ess=float(k / np.mean(w)), n_star_is=float(n_is),
                n_star_naive=float(n_naive), speedup=float(n_naive / n_is),
                fz=fz, gz=gz, bw=bw, mu=mu, sd=sd)

# --------------------------------------------------------------------- prep + verification
def prep(verbose=True):
    pool_h1, paper_pos0 = load_pool()
    N = len(pool_h1)
    pool_scores = np.load(HERE / "pool_scores_275.npy")
    eval_idx = np.random.default_rng(0).choice(N, size=min(EVAL_N, N), replace=False)
    rows = [r for r in load_rows() if r["status"] == "ok"]
    store, bad = {}, 0
    for row in rows:
        rec = reconstruct(row, pool_scores, pool_h1, eval_idx, paper_pos0)
        ok = (abs(rec["ess"] - row["ess"]) < 1e-6 * max(1, row["ess"]) and
              abs(rec["p_is"] - row["p_is"]) < 1e-9 and
              abs(rec["speedup"] - row["speedup"]) < 1e-6 * row["speedup"])
        if not ok:
            bad += 1
            if verbose:
                print(f"  MISMATCH {row['label']}: ess {rec['ess']:.3f} vs {row['ess']:.3f}, "
                      f"p_is {rec['p_is']:.5f} vs {row['p_is']:.5f}")
        store[row["label"]] = rec
    np.savez_compressed(
        CACHE,
        labels=np.array(list(store)),
        eval_idx=eval_idx,
        **{f"w_{l}": store[l]["w"] for l in store},
        **{f"r_{l}": store[l]["r"] for l in store},
        **{f"pilot_{l}": store[l]["pilot_idx"] for l in store},
    )
    print(f"prep: reconstructed {len(store)} replicates, {bad} mismatches vs seed_study_results.csv"
          f" -> cache {CACHE.name}")
    return bad

def ctx():
    pool_h1, paper_pos0 = load_pool()
    pool_scores = np.load(HERE / "pool_scores_275.npy")
    z = np.load(CACHE, allow_pickle=False)
    eval_idx = z["eval_idx"]
    rows = {r["label"]: r for r in load_rows() if r["status"] == "ok"}
    return pool_h1, paper_pos0, pool_scores, eval_idx, z, rows

def write_csv(name, header, lines):
    with open(OUT / name, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(header); wr.writerows(lines)
    print(f"  -> {OUT.name}/{name}")

# --------------------------------------------------------------------- t1 coverage
def t1():
    print("T1 — empirical coverage of the certified intervals (target >= 95%)")
    pool_h1, _, _, eval_idx, z, rows = ctx()
    p_ref = float(pool_h1[eval_idx].mean()); R = 200
    lines = []
    for lab, row in rows.items():
        w, r = z[f"w_{lab}"], z[f"r_{lab}"]
        he = pool_h1[eval_idx]; n = int(np.ceil(row["n_star_is"]))
        rng = np.random.default_rng(123)
        hits = 0
        for _ in range(R):
            dr = rng.choice(len(eval_idx), size=n, replace=True, p=r)
            ph = np.sum(w[dr] * he[dr]) / np.sum(w[dr])
            hits += abs(ph - p_ref) <= EPS_REL * p_ref
        lines.append([lab, row["k"], n, hits / R])
    cov = np.array([l[3] for l in lines])
    for k in (100, 300, 500):
        c = np.array([l[3] for l in lines if l[1] == k])
        print(f"  IS, k={k}: median coverage {np.median(c):.2f}, min {c.min():.2f}, "
              f"share of seeds >=0.95: {(c >= 0.95).mean():.0%}")
    # naive baselines at p_ref
    rng = np.random.default_rng(7)
    for name, n in (("naive-Chebyshev", int(np.ceil((1/DELTA)*(1-p_ref)/(EPS_REL**2*p_ref)))),
                    ("naive-Wilson/CLT", int(np.ceil(Z95**2*(1-p_ref)/(EPS_REL**2*p_ref))))):
        ph = rng.binomial(n, p_ref, size=5000) / n
        print(f"  {name} (n={n}): coverage {(np.abs(ph-p_ref) <= EPS_REL*p_ref).mean():.3f}")
    write_csv("t1_coverage.csv", ["label", "k", "n_star_is", "coverage"], lines)
    print(f"  overall: {(cov >= 0.95).mean():.0%} of replicates meet the 95% certificate\n")

# --------------------------------------------------------------------- t2 adversarial bounds
def t2():
    print("T2 — naive-with-Wilson vs IS-with-Chebyshev (the comparison R2 can demand)")
    pool_h1, _, _, eval_idx, z, rows = ctx()
    p_ref = float(pool_h1[eval_idx].mean())
    n_wilson = Z95**2 * (1 - p_ref) / (EPS_REL**2 * p_ref)
    lines = []
    for lab, row in rows.items():
        cost_is = row["n_star_is"] + row["k"]           # pilot-inclusive
        lines.append([lab, row["k"], row["n_star_is"], cost_is, n_wilson,
                      n_wilson / row["n_star_is"], n_wilson / cost_is])
    print(f"  naive n* under Wilson/CLT at p={p_ref:.4f}: {n_wilson:.0f} "
          f"(vs Chebyshev {(1/DELTA)*(1-p_ref)/(EPS_REL**2*p_ref):.0f})")
    for k in (100, 300, 500):
        s = np.array([l[6] for l in lines if l[1] == k])   # speedup vs Wilson, pilot incl.
        print(f"  k={k}: IS beats naive-Wilson in {(s > 1).mean():.0%} of seeds "
              f"(median 'speedup' vs Wilson: {np.median(s):.2f}x)")
    write_csv("t2_bound_comparison.csv",
              ["label", "k", "n_star_is", "cost_is_incl_pilot", "n_naive_wilson",
               "speedup_vs_wilson_excl", "speedup_vs_wilson_incl"], lines)
    print()

# --------------------------------------------------------------------- t3 weight sanity
def t3():
    print("T3 — reference-free weight-sanity identity E_f[g'/f] = 1 (catches the bug class)")
    pool_h1, paper_pos0, pool_scores, eval_idx, z, rows = ctx()
    lines = []
    for lab, row in rows.items():
        w = z[f"w_{lab}"]
        m = float(np.mean(1.0 / w))       # E_f[g'/f] estimated on the prior-distributed pool
        lines.append([lab, row["k"], m, abs(m - 1)])
    dev = np.array([l[3] for l in lines])
    print(f"  corrected weights: median |E_f[g'/f]-1| = {np.median(dev):.3f}, "
          f"max = {dev.max():.3f}  (identity holds)")
    # --- reproduce the ORIGINAL buggy cell's mechanics on the paper pilot
    row = rows["paper"]
    pilot = paper_pos0; sens = row["sens"]
    cols = pool_scores[pilot][:, sens]
    mu, sd = cols.mean(0), cols.std(0, ddof=1); sd[sd == 0] = 1
    fz = (cols - mu) / sd; gz = fz[pool_h1[pilot] == 1]
    bw = normal_ref_bw(fz)
    f_tr = kde_pdf(fz, fz, bw); g_tr = kde_pdf(fz, gz, bw)       # pdf() at TRAINING points
    S_int = np.clip(fz.astype(int), -len(fz), len(fz) - 1)       # scores cast to int = indices
    buggy = np.mean(f_tr[S_int] / g_tr[S_int], axis=1)           # per-dim averaging
    print(f"  buggy-cell weights (same mechanics as IS notebook cell 4):")
    print(f"    Kish ESS = {ess_kish(buggy):.0f}/300 (healthy-looking, like the published 297)")
    print(f"    identity check mean = {np.mean(1.0/ (buggy / np.mean(buggy))):.2f} "
          f"with CV(1/w) = {np.std(1.0/buggy)/np.mean(1.0/buggy):.2f}")
    print(f"    correlation(buggy w, correct f/g at same points): "
          f"{np.corrcoef(buggy, kde_pdf(fz, fz, bw)/kde_pdf(fz, gz, bw))[0,1]:.3f}")
    write_csv("t3_weight_identity.csv", ["label", "k", "mean_inv_w", "abs_dev"], lines)
    print()

# --------------------------------------------------------------------- t4 split pool
def t4():
    print("T4 — disjoint evaluation (pool minus pilot): does the result survive?")
    pool_h1, paper_pos0, pool_scores, eval_idx, z, rows = ctx()
    N = len(pool_h1); rng = np.random.default_rng(42)
    lines = []
    for lab, row in rows.items():
        pilot = z[f"pilot_{lab}"]
        comp = np.setdiff1d(np.arange(N), pilot)
        ev = rng.choice(comp, size=EVAL_N, replace=False)
        rec = reconstruct(row, pool_scores, pool_h1, None, paper_pos0, eval_override=ev)
        lines.append([lab, row["k"], row["speedup"], rec["speedup"],
                      row["p_is"], rec["p_is"]])
    for k in (100, 300, 500):
        a = np.array([(l[2], l[3]) for l in lines if l[1] == k])
        print(f"  k={k}: median speedup {np.median(a[:,0]):.2f} -> {np.median(a[:,1]):.2f} "
              f"(disjoint eval); beat-MC {(a[:,0]>1).mean():.0%} -> {(a[:,1]>1).mean():.0%}")
    write_csv("t4_split_pool.csv",
              ["label", "k", "speedup_orig", "speedup_disjoint", "p_is_orig", "p_is_disjoint"],
              lines)
    print()

# --------------------------------------------------------------------- t5 SNIS bias
def t5():
    print("T5 — self-normalized (paper) vs unnormalized estimator: finite-k bias")
    pool_h1, _, _, eval_idx, z, rows = ctx()
    p_ref = float(pool_h1[eval_idx].mean()); R = 400
    lines = []
    for lab, row in rows.items():
        w, r = z[f"w_{lab}"], z[f"r_{lab}"]
        he = pool_h1[eval_idx]; k = row["k"]
        rng = np.random.default_rng(11)
        sn, un = np.empty(R), np.empty(R)
        for i in range(R):
            dr = rng.choice(len(eval_idx), size=k, replace=True, p=r)
            sn[i] = np.sum(w[dr] * he[dr]) / np.sum(w[dr])   # SNIS (what the paper uses)
            un[i] = np.mean(w[dr] * he[dr])                  # unnormalized w = f/g' exactly
        lines.append([lab, k, p_ref, sn.mean(), un.mean(),
                      (sn.mean() - p_ref) / p_ref, (un.mean() - p_ref) / p_ref])
    for k in (100, 300, 500):
        bs = np.array([l[5] for l in lines if l[1] == k])
        bu = np.array([l[6] for l in lines if l[1] == k])
        print(f"  k={k}: SNIS rel. bias median {np.median(bs):+.1%} "
              f"[{bs.min():+.1%}, {bs.max():+.1%}]; "
              f"unnormalized {np.median(bu):+.1%} [{bu.min():+.1%}, {bu.max():+.1%}]")
    write_csv("t5_snis_bias.csv",
              ["label", "k", "p_ref", "mean_snis", "mean_unnorm",
               "rel_bias_snis", "rel_bias_unnorm"], lines)
    print("  (SNIS is consistent but biased at finite k — avoid the word 'unbiased')\n")

# --------------------------------------------------------------------- t6 pilot feasibility
def t6():
    print("T6 — pilot feasibility: P(>= 2 pilot failures) (method cannot start below 2)")
    rows = load_rows()
    n100 = [r for r in rows if r["k"] == 100 and r["label"] != "paper"]
    emp = sum(1 for r in n100 if r["status"] == "ok") / len(n100) if n100 else float("nan")
    lines = []
    for p in (0.0287, 1e-3):
        ks = [100, 300, 500, 1000, 2000, 5000, 10000]
        need = None
        for k in range(2, 200000):
            q = 1 - (1 - p) ** k - k * p * (1 - p) ** (k - 1)
            if q >= 0.95:
                need = k; break
        for k in ks:
            q = 1 - (1 - p) ** k - k * p * (1 - p) ** (k - 1)
            lines.append([p, k, q])
        print(f"  p={p:g}: P(>=2 fails) at k=100/300/500 = "
              + "/".join(f"{1 - (1-p)**k - k*p*(1-p)**(k-1):.2f}" for k in (100, 300, 500))
              + f"; need k={need} for 95% start-probability")
    print(f"  empirical at k=100 (p~0.029): {emp:.0%} of 25 seeds could start "
          f"(binomial predicts {1-(1-0.0287)**100-100*0.0287*(1-0.0287)**99:.0%})")
    write_csv("t6_pilot_feasibility.csv", ["p", "k", "prob_ge2_failures"], lines)
    print()

# --------------------------------------------------------------------- t7 sweeps
def t7():
    print("T7 — robustness of the paper config: bandwidth x alpha sweep")
    pool_h1, paper_pos0, pool_scores, eval_idx, z, rows = ctx()
    row = rows["paper"]; lines = []
    for bwm in (0.5, 1.0, 2.0):
        for al in (0.25, 0.5, 0.75):
            rec = reconstruct(row, pool_scores, pool_h1, eval_idx, paper_pos0,
                              bw_mult=bwm, alpha=al)
            lines.append([bwm, al, rec["speedup"], rec["ess"], rec["p_is"]])
            print(f"  bw x{bwm:<4} alpha={al:<5} speedup {rec['speedup']:.2f}x  "
                  f"ESS {rec['ess']:.0f}  p_IS {rec['p_is']:.4f}")
    write_csv("t7_sweeps.csv", ["bw_mult", "alpha", "speedup", "ess", "p_is"], lines)
    print()

# --------------------------------------------------------------------- t8 PC frequency
def t8():
    print("T8 — PC-selection stability and its link to efficiency")
    _, _, _, _, z, rows = ctx()
    paper_set = set(rows["paper"]["sens"].tolist())
    from collections import Counter
    lines = []
    for k in (100, 300, 500):
        sel = [r for r in rows.values() if r["k"] == k and r["label"] != "paper"]
        cnt = Counter(int(pc) for r in sel for pc in r["sens"])
        top = ", ".join(f"PC{pc}:{c}/{len(sel)}" for pc, c in cnt.most_common(9))
        ov = np.array([len(paper_set & set(r["sens"].tolist())) for r in sel])
        sp = np.array([r["speedup"] for r in sel])
        rho = np.corrcoef(ov, sp)[0, 1] if len(sel) > 2 else float("nan")
        print(f"  k={k}: top selections {top}")
        print(f"        overlap with paper's 7 PCs: median {np.median(ov):.0f}/7; "
              f"corr(overlap, speedup) = {rho:+.2f}")
        for r in sel:
            lines.append([r["label"], k, len(paper_set & set(r["sens"].tolist())),
                          r["speedup"], json.dumps(r["sens"].tolist())])
    write_csv("t8_pc_selection.csv", ["label", "k", "overlap_with_paper", "speedup", "pcs"],
              lines)
    print()

# --------------------------------------------------------------------- t9 certification (CI-first)
P_RISK = 0.04          # the paper's prescribed risk threshold (Results / Eq. 19)
KCHEB = 1.0 / np.sqrt(DELTA)

def t9():
    print(f"T9 — CI-first certification analysis (paper rule: upper band < p_risk={P_RISK})")
    pool_h1, _, _, eval_idx, z, rows = ctx()
    he = pool_h1[eval_idx]; p_true = float(he.mean())
    n_naive_cheb = KCHEB**2 * p_true * (1 - p_true) / (P_RISK - p_true)**2
    n_naive_clt = Z95**2 * p_true * (1 - p_true) / (P_RISK - p_true)**2
    print(f"  true p={p_true:.4f} (margin {(P_RISK-p_true)/p_true:.0%}); naive certification "
          f"cost: Chebyshev {n_naive_cheb:.0f}, Wilson/CLT {n_naive_clt:.0f}")
    rng = np.random.default_rng(99); R = 200
    lines = []
    for lab, row in rows.items():
        k, p_is = row["k"], row["p_is"]
        w, r = z[f"w_{lab}"], z[f"r_{lab}"]
        var_is = float(np.mean(he * w) - p_true**2)
        n_cert = KCHEB**2 * var_is / (P_RISK - p_is)**2 if p_is < P_RISK else np.inf
        covers = abs(p_is - p_true) <= KCHEB * np.sqrt(var_is / k)   # CI at own batch
        danger = np.nan
        if np.isfinite(n_cert):
            n = int(np.ceil(n_cert)); bad = 0
            for _ in range(R):
                dr = rng.choice(len(eval_idx), size=n, replace=True, p=r)
                ph = np.sum(w[dr] * he[dr]) / np.sum(w[dr])
                if ph + KCHEB * np.sqrt(var_is / n) < p_true:
                    bad += 1                                          # false-safety certificate
            danger = bad / R
        lines.append([lab, k, p_is, var_is, n_cert, n_cert + k, covers, danger])
    for kk in (100, 300, 500):
        v = [l for l in lines if l[1] == kk]
        fin = [l[5] for l in v if np.isfinite(l[4])]
        nocert = sum(1 for l in v if not np.isfinite(l[4]))
        cov = sum(l[6] for l in v) / len(v)
        dang = [l[7] for l in v if np.isfinite(l[4])]
        beat = sum(1 for l in v if np.isfinite(l[4]) and l[5] < n_naive_cheb) / max(1, len(v) - nocert)
        print(f"  k={kk}: cert cost incl pilot median {np.median(fin):.0f} "
              f"[{min(fin):.0f},{max(fin):.0f}] | cannot certify: {nocert}/{len(v)} | "
              f"CI covers truth at own batch: {cov:.0%} | danger rate median "
              f"{np.median(dang):.3f} | beats naive-Chebyshev: {beat:.0%}")
    p = next(l for l in lines if l[0] == "paper")
    print(f"  paper config: {p[5]:.0f} incl pilot -> {n_naive_cheb/p[5]:.1f}x vs naive-Chebyshev "
          f"(its p_hat={p[2]:.4f} sits near the threshold — do not quote as an anchor)")
    write_csv("t9_certification.csv",
              ["label", "k", "p_is", "var_is", "n_cert", "n_cert_incl_pilot",
               "ci_covers_truth_at_own_batch", "danger_rate"], lines)
    print("  NOTE: per-seed costs are projections from the seed's own p_hat; a sequential run "
          "self-corrects,\n        so extremes (incl. 'cannot certify') are bounds, not outcomes.\n")

# --------------------------------------------------------------------- main
TESTS = dict(t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6, t7=t7, t8=t8, t9=t9)

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if "prep" in args or (not CACHE.exists() and args != ["t6"]):
        prep()
    for a in args:
        if a == "prep":
            continue
        if a == "all":
            for f in TESTS.values():
                f()
        else:
            TESTS[a]()
