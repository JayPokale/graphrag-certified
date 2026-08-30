"""
theta_lowerbound.py — empirical evidence for the MASC lower bound (T2 Theorem/Conjecture).

Builds the BALANCED THETA gadget: a forward w-ary tree of depth a=(k-1)//2 from v and a
backward w-ary tree of depth b=(k-1)-a INTO v, joined by ONE planted edge (forward-leaf ->
backward-leaf) that closes a single directed k-cycle through v. The planted edge is the
"needle"; every sampler must traverse it to detect the cycle.

Claim under test: on balanced theta, NO local sampler (uniform / single-source ppr /
bidirectional bippr) collects the needle in fewer than ~w^{(k-1)/2} samples — the polynomial
advantage bippr enjoys in the RADIAL regime collapses to a constant factor here. This is the
empirical face of the Omega(w^{k/2}) lower bound (T2): bippr is optimal only up to constants,
and cannot be beaten polynomially.

Output: results_theta.json + figures/theta_lowerbound.png
"""
from __future__ import annotations
import json, os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import masc

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")


def make_theta(w, k, seed):
    a = (k - 1) // 2
    b = (k - 1) - a                      # a + 1 + b = k edges in the cycle
    rng = np.random.default_rng(seed)
    G = nx.DiGraph(); v = 0; nid = 1
    fl = [v]
    for _ in range(a):                   # forward tree: edges directed AWAY from v
        nf = []
        for node in fl:
            for _ in range(w):
                G.add_edge(node, nid); nf.append(nid); nid += 1
        fl = nf
    bl = [v]
    for _ in range(b):                   # backward tree: edges directed TOWARD v
        nb = []
        for node in bl:
            for _ in range(w):
                G.add_edge(nid, node); nb.append(nid); nid += 1
        bl = nb
    lf = int(rng.choice(fl)); lb = int(rng.choice(bl))
    G.add_edge(lf, lb)                    # planted closing edge (the needle)
    return G, v, (lf, lb), a, b


def run(ws=(3, 4), ks=(3, 5, 7, 9), seeds=range(20)):
    out = {}
    for w in ws:
        out[str(w)] = {}
        for k in ks:
            a = (k - 1) // 2
            rows = {m: [] for m in ["uniform", "ppr", "bippr"]}
            balls = []
            for s in seeds:
                G, v, needle, a, b = make_theta(w, k, s)
                eb, _ = masc.ball_edges(G, v, k); balls.append(len(eb))
                rng = np.random.default_rng(500 + s)
                for m in rows:
                    edges, p = masc.edge_distribution(G, v, k, m)
                    d, capped = masc.samples_to_collect(edges, p, [needle], rng, cap_mult=300)
                    if not capped:
                        rows[m].append(d)
            ref = w ** a                  # w^{(k-1)/2}
            out[str(w)][str(k)] = dict(
                a=a, ref_w_pow_a=ref, ball=float(np.mean(balls)),
                **{m: (float(np.mean(rows[m])) if rows[m] else None) for m in rows},
                **{m + "_over_ref": (float(np.mean(rows[m])) / ref if rows[m] else None)
                   for m in rows})
    return out


def plot(out, ws, ks):
    fig, axes = plt.subplots(1, len(ws), figsize=(5 * len(ws), 4), squeeze=False)
    for j, w in enumerate(ws):
        ax = axes[0][j]
        for m in ["uniform", "ppr", "bippr"]:
            ys = [out[str(w)][str(k)][m] for k in ks]
            ax.plot(ks, ys, marker="o", label=m)
        ax.plot(ks, [out[str(w)][str(k)]["ref_w_pow_a"] for k in ks], "k--",
                alpha=0.6, label=r"$w^{(k-1)/2}$")
        ax.set_yscale("log"); ax.set_xlabel("k"); ax.set_title(f"theta gadget, w={w}")
        ax.set_ylabel("samples to collect the needle"); ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("MASC lower bound: on balanced theta, ALL samplers ~ $w^{(k-1)/2}$ "
                 "(bippr's radial advantage collapses to a constant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "theta_lowerbound.pdf"))
    fig.savefig(os.path.join(FIG, "theta_lowerbound.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    ws = (3, 4); ks = (3, 5, 7, 9)
    out = run(ws, ks)
    plot(out, ws, ks)
    with open(os.path.join(HERE, "results_theta.json"), "w") as f:
        json.dump(out, f, indent=2)
    # console summary: ratio to w^{(k-1)/2} should stay ~O(1), not shrink
    print(f"{'w':>2} {'k':>2} {'w^a':>6} {'uniform':>8} {'ppr':>8} {'bippr':>8} "
          f"{'bippr/ref':>10}")
    for w in ws:
        for k in ks:
            d = out[str(w)][str(k)]
            print(f"{w:>2} {k:>2} {d['ref_w_pow_a']:>6} "
                  f"{(d['uniform'] or -1):>8.0f} {(d['ppr'] or -1):>8.0f} "
                  f"{(d['bippr'] or -1):>8.0f} {(d['bippr_over_ref'] or -1):>10.2f}")
    print("\nbippr/ref staying ~constant (not ->0) as k grows = bippr cannot beat w^{k/2} "
          "polynomially on the hard instance = supports Omega(w^{k/2}).")
