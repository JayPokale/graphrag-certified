"""
run_all.py — orchestrate E2/E3/E4, emit figures/ + results.json.

  python3 run_all.py --quick     # small, ~seconds
  python3 run_all.py             # fuller, a few minutes

Honest design notes:
  E2  measures motif spread sigma_M (directed triangles, exact).
  E3  measures sample-complexity to detect a planted k-cycle in TWO regimes
      (radial = unfavorable for single-source PPR; hub-concentrated = favorable),
      across 4 samplers (uniform/degree/ppr/bippr). Reports samples + p_min.
  E4  validates the replication certificate: (a) clean detection vs rho (the tradeoff),
      (b) verdict-flip vs poison budget for optimal vs random adversary (tightness).
"""
from __future__ import annotations
import argparse, json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import masc

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def mean_ci(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) == 0:
        return float("nan"), 0.0
    m = xs.mean()
    se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return float(m), float(1.96 * se)


# ----------------------------- E2: motif spread ---------------------------- #
def exp_spread(seeds, sizes):
    models = {
        "scale_free": lambda n, s: masc.gen_scale_free(n, s),
        "chung_lu_2.5": lambda n, s: masc.gen_chung_lu(n, 2.5, s),
        "er_control": lambda n, s: masc.gen_er(n, 6.0, s),
    }
    out = {}
    for name, gen in models.items():
        out[name] = {}
        for n in sizes:
            sig = []
            for s in seeds:
                res = masc.directed_triangle_spread(gen(n, s))
                if res:
                    sig.append(res["sigma_M"])
            m, ci = mean_ci(sig)
            out[name][n] = dict(sigma_mean=m, sigma_ci=ci, n_samples=len(sig))
    return out


def plot_spread(data, sizes):
    plt.figure(figsize=(6, 4))
    for name, series in data.items():
        xs = [n for n in sizes if not np.isnan(series[n]["sigma_mean"])]
        ys = [series[n]["sigma_mean"] for n in xs]
        es = [series[n]["sigma_ci"] for n in xs]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
    # polylog reference
    ref = np.array(sizes, float)
    plt.plot(sizes, 3 * np.log(ref) ** 2, "k:", alpha=0.5, label=r"$\sim\log^2 n$ ref")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("n (nodes)"); plt.ylabel(r"motif spread $\sigma_M$ (directed triangles)")
    plt.title(r"E2: $\sigma_M$ vs size — grows with $n$, not polylog")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "E2_sigma_spread.pdf"))
    plt.savefig(os.path.join(FIG, "E2_sigma_spread.png"), dpi=130)
    plt.close()


# --------------------------- E3: sample complexity ------------------------- #
def exp_sample_complexity(seeds, ks, w, n_decoy):
    modes = ["uniform", "degree", "ppr", "bippr"]
    regimes = {
        "radial": lambda k, s: masc.make_branching_with_cycle(w, k, s),
        "hub": lambda k, s: masc.make_hub_concentrated(max(w, k), n_decoy, k, s),
    }
    out = {}
    for rname, gen in regimes.items():
        out[rname] = {"samples": {m: {} for m in modes},
                      "pmin": {m: {} for m in modes}, "ball": {}}
        for k in ks:
            per_s = {m: [] for m in modes}
            per_p = {m: [] for m in modes}
            bsz = []
            for s in seeds:
                G, v, planted = gen(k, s)
                eb, _ = masc.ball_edges(G, v, k)
                bsz.append(len(eb))
                rng = np.random.default_rng(7000 + s)
                for m in modes:
                    edges, p = masc.edge_distribution(G, v, k, m)
                    pmin, ok = masc.min_motif_mass(edges, p, planted)
                    if ok:
                        per_p[m].append(pmin)
                    draws, capped = masc.samples_to_collect(edges, p, planted, rng)
                    if not capped:
                        per_s[m].append(draws)
            for m in modes:
                mm, ci = mean_ci(per_s[m])
                out[rname]["samples"][m][k] = dict(mean=mm, ci=ci, n=len(per_s[m]))
                pm, pci = mean_ci(per_p[m])
                out[rname]["pmin"][m][k] = dict(mean=pm, ci=pci)
            out[rname]["ball"][k] = float(np.mean(bsz))
    return out


def plot_sample_complexity(data, ks, w):
    for rname in data:
        plt.figure(figsize=(6, 4))
        for m, series in data[rname]["samples"].items():
            ys = [series[k]["mean"] for k in ks]
            es = [series[k]["ci"] for k in ks]
            plt.errorbar(ks, ys, yerr=es, marker="o", capsize=3, label=m)
        plt.plot(ks, [data[rname]["ball"][k] for k in ks], "k--", alpha=0.5,
                 label=r"$|E_k(v)|$")
        plt.yscale("log")
        plt.xlabel("k (cycle length)")
        plt.ylabel("edge samples to detect planted k-cycle")
        plt.title(f"E3 [{rname}]: sample complexity vs k (w={w})")
        plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"E3_samples_{rname}.pdf"))
        plt.savefig(os.path.join(FIG, f"E3_samples_{rname}.png"), dpi=130)
        plt.close()


# ----------------------- E4: replication certificate ----------------------- #
def exp_detection(seeds, Ks, rhos, d):
    """E4a: clean detection prob vs rho for several K (the isolation/detectability tradeoff)."""
    out = {}
    for K in Ks:
        out[K] = {}
        for rho in rhos:
            if rho > K:
                continue
            ps = [masc.clean_detection_prob(K, rho, d, s)["detect_prob"] for s in seeds]
            out[K][rho] = dict(detect=float(np.mean(ps)),
                               threshold=K ** (1 - 1.0 / d))
    return out


def plot_detection(data, d):
    plt.figure(figsize=(6, 4))
    for K, series in data.items():
        rhos = sorted(series)
        plt.plot(rhos, [series[r]["detect"] for r in rhos], marker="o", label=f"K={K}")
        thr = K ** (1 - 1.0 / d)
        plt.axvline(thr, ls="--", alpha=0.4)
    plt.xlabel(r"replication factor $\rho$")
    plt.ylabel("clean detection probability")
    plt.title(f"E4a: detectability vs replication (d={d})\n"
              r"dashed = $\rho=K^{1-1/d}$ predicted onset")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "E4a_detection_tradeoff.pdf"))
    plt.savefig(os.path.join(FIG, "E4a_detection_tradeoff.png"), dpi=130)
    plt.close()


def exp_flip(seeds, K, mu, rho):
    """E4b: flip rate vs total budget for optimal vs random adversary at fixed margin mu."""
    out = {"optimal": [], "random": []}
    for adv in out:
        for m_s in [0, 1]:
            for m_d in range(0, K):
                fr = [masc.flip_vs_budget(K, mu, rho, m_d, m_s, s, adversary=adv)["flip_rate"]
                      for s in seeds]
                out[adv].append(dict(m_d=m_d, m_s=m_s, budget=m_d + rho * m_s,
                                     flip=float(np.mean(fr))))
    return out


def plot_flip(data, K, mu, rho):
    plt.figure(figsize=(6, 4))
    for adv, recs in data.items():
        recs = sorted(recs, key=lambda r: r["budget"])
        xs = [r["budget"] for r in recs]
        ys = [r["flip"] for r in recs]
        plt.plot(xs, ys, marker="o", ls="none" if adv == "random" else "-", label=adv, alpha=0.7)
    plt.axvline(mu, color="red", ls="--", label=fr"certificate $\mu={mu}$")
    plt.xlabel(r"poison budget $m_d + \rho\, m_s$")
    plt.ylabel("verdict-flip rate")
    plt.title(f"E4b: tightness of certificate (K={K}, $\\rho$={rho}, $\\mu$={mu})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "E4b_certificate_tightness.pdf"))
    plt.savefig(os.path.join(FIG, "E4b_certificate_tightness.png"), dpi=130)
    plt.close()


# --------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    if args.quick:
        seeds = list(range(5)); sizes = [200, 500, 1000]; ks = [3, 4, 5]; w = 3
        n_decoy = 300; Ks = [9, 15]; rhos = [1, 2, 3, 4, 5, 7, 9]; d = 3
        cert_seeds = list(range(6)); fK, fmu, frho = 11, 3, 2
    else:
        seeds = list(range(15)); sizes = [500, 1000, 2000, 4000, 8000]
        ks = [3, 4, 5, 6, 7]; w = 4
        n_decoy = 1500; Ks = [9, 15, 25]; rhos = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25]; d = 3
        cert_seeds = list(range(30)); fK, fmu, frho = 15, 4, 3

    R = {"config": dict(seeds=len(seeds), sizes=sizes, ks=ks, w=w, n_decoy=n_decoy,
                        Ks=Ks, rhos=rhos, d=d, fK=fK, fmu=fmu, frho=frho, quick=args.quick)}

    print("E2 spread..."); s = exp_spread(seeds, sizes)
    R["E2_spread"] = s; plot_spread(s, sizes)

    print("E3 sample complexity..."); sc = exp_sample_complexity(seeds, ks, w, n_decoy)
    R["E3"] = sc; plot_sample_complexity(sc, ks, w)

    print("E4a detection..."); det = exp_detection(cert_seeds, Ks, rhos, d)
    R["E4a_detection"] = {str(k): v for k, v in det.items()}; plot_detection(det, d)

    print("E4b flip/tightness..."); fl = exp_flip(cert_seeds, fK, fmu, frho)
    R["E4b_flip"] = fl; plot_flip(fl, fK, fmu, frho)

    R["runtime_sec"] = time.time() - t0
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(f"done in {R['runtime_sec']:.1f}s -> results.json + figures/")


if __name__ == "__main__":
    main()
