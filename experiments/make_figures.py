#!/usr/bin/env python3
"""make_figures.py -- the paper's figures, drawn from the committed results.

Everything data-bearing here is read from results_*.json, so a figure cannot drift away
from the reproduction gate.  Output is vector PDF at IEEEtran column width, greyscale-safe
apart from one accent reserved throughout for the adversary.

    python3 make_figures.py        (writes ../figures/fig_*.pdf)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

COL = 3.4          # IEEEtran conference \columnwidth, inches
INK = "#000000"
MID = "#666666"
PALE = "#BBBBBB"
ADV = "#8B1A1A"    # reserved for the adversary, everywhere

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("  " + os.path.relpath(p, HERE))


def node(ax, x, y, r=0.055, fc="#FFFFFF", ec=INK, lw=0.9, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


# --------------------------------------------------------------- Fig 1: independence
def fig_independence():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.30))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # left: six chains, all through one relation
    ax = axes[0]
    v, bott, a = (0.08, 0.5), (0.55, 0.5), (0.92, 0.5)
    for y in (0.86, 0.72, 0.58, 0.44, 0.30, 0.16):
        ax.plot([v[0], 0.30], [v[1], y], color=PALE, lw=0.55, zorder=1)
        ax.plot([0.30, bott[0]], [y, bott[1]], color=PALE, lw=0.55, zorder=1)
        node(ax, 0.30, y, r=0.028, lw=0.6)
    ax.plot([bott[0], a[0]], [0.5, 0.5], color=ADV, lw=2.0, zorder=2)
    node(ax, *bott, r=0.045, fc="#FFFFFF", ec=ADV, lw=1.2)
    node(ax, *v, fc=INK, ec=INK)
    node(ax, *a, fc=MID, ec=MID)
    ax.text(0.5, 0.985, r"six chains,  $\omega = 1$", ha="center", va="top", fontsize=7.5)
    ax.text(0.73, 0.40, "shared", ha="center", va="top", fontsize=6.5, color=ADV)

    # right: three chains that share nothing
    ax = axes[1]
    v, a = (0.08, 0.5), (0.92, 0.5)
    for y in (0.80, 0.50, 0.20):
        ax.plot([v[0], 0.36], [v[1], y], color=INK, lw=0.8, zorder=1)
        ax.plot([0.36, 0.64], [y, y], color=INK, lw=0.8, zorder=1)
        ax.plot([0.64, a[0]], [y, a[1]], color=INK, lw=0.8, zorder=1)
        node(ax, 0.36, y, r=0.032); node(ax, 0.64, y, r=0.032)
    node(ax, *v, fc=INK, ec=INK)
    node(ax, *a, fc=MID, ec=MID)
    ax.text(0.5, 0.985, r"three chains,  $\omega = 3$", ha="center", va="top", fontsize=7.5)

    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_independence")


# --------------------------------------------------------------- Fig 2: the trap
def fig_trap():
    # equal aspect, coordinates matched to the canvas, so circles stay circles
    W, H = COL, 1.30
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_aspect("equal")
    ax.set_xlim(0, W / H); ax.set_ylim(-0.18, 1.0); ax.axis("off")

    src = (0.30, 0.80)
    node(ax, *src, r=0.055, fc="#FFFFFF", ec=ADV, lw=1.3)
    ax.text(0.30, 0.90, "one relation", ha="center", va="bottom",
            fontsize=6.8, color=ADV)

    xs = [0.95, 1.42, 1.89, 2.36]
    for x in xs:
        ax.add_patch(FancyBboxPatch((x - 0.17, 0.10), 0.34, 0.34,
                                    boxstyle="round,pad=0.012",
                                    facecolor="#FFFFFF", edgecolor=MID, linewidth=0.7))
        ax.annotate("", xy=(x, 0.46), xytext=(src[0] + 0.05, src[1] - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color=ADV, lw=0.6,
                                    mutation_scale=5, shrinkA=2, shrinkB=1))
        node(ax, x, 0.27, r=0.035, fc=ADV, ec=ADV)
    ax.text((xs[0] + xs[-1]) / 2, -0.16,
            r"every cell it reaches falls together: $\rho$ of $K$",
            ha="center", va="bottom", fontsize=6.8, color=ADV)
    save(fig, "fig_trap")


# --------------------------------------------------------------- Fig 3: three designs
def fig_designs():
    d = json.load(open(os.path.join(HERE, "results_witness.json")))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True)
    for ax, key, title in ((axes[0], "E32_E33_2path", r"$2$-path"),
                           (axes[1], "E32_E33_C4", r"anchored $C_4$")):
        recs = d[key]["records"]
        ks = [r["K"] for r in recs]
        pub = [r["med_b_paper"] for r in recs]
        tig = [r["med_b_tight"] for r in recs]
        wit = [r["med_b_witness"] for r in recs]
        idx = range(len(ks))
        w = 0.26
        ax.bar([i - w for i in idx], pub, w, color=PALE, edgecolor=INK, linewidth=0.5,
               label="published")
        ax.bar(list(idx), tig, w, color=MID, edgecolor=INK, linewidth=0.5,
               label="reachable-damage")
        ax.bar([i + w for i in idx], wit, w, color=INK, edgecolor=INK, linewidth=0.5,
               label="witness packing")
        for i, v in zip(idx, wit):
            ax.text(i + w, v + 0.08, ("%g" % v), ha="center", va="bottom", fontsize=6.2)
        for i, v in zip(idx, pub):   # a zero bar draws nothing; say so
            ax.text(i - w, 0.08, "0", ha="center", va="bottom", fontsize=6.2, color=ADV)
        ax.set_xticks(list(idx)); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel(r"cells $K$", labelpad=1.5)
        ax.set_title(title, fontsize=7.5, pad=2.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    axes[0].set_ylabel("certified budget", labelpad=2)
    axes[0].set_ylim(0, 4.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.4, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), handlelength=1.0, handletextpad=0.4,
               columnspacing=1.2)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_designs")


# --------------------------------------------------------------- Fig 4: omega spectrum
def fig_omega():
    m = json.load(open(os.path.join(HERE, "results_menger.json")))["E35_menger"]
    hist = {int(k): v for k, v in m["omega_hist"].items()}
    ks = sorted(hist)
    vals = [hist[k] for k in ks]
    n = sum(vals)
    dead = sum(v for k, v in hist.items() if k < 3)

    fig, ax = plt.subplots(figsize=(COL, 1.25))
    cols = [PALE if k < 3 else INK for k in ks]
    ax.bar(ks, vals, 0.78, color=cols, edgecolor=INK, linewidth=0.5)
    ax.axvline(2.5, color=ADV, lw=0.8, ls=(0, (3, 2)))
    ax.text(2.75, max(vals) * 0.93, r"certifies $\geq 1$", fontsize=6.5, color=ADV,
            va="top", ha="left")
    ax.text(2.3, max(vals) * 0.93, "abstain", fontsize=6.5, color=ADV,
            va="top", ha="right")
    ax.set_xlabel(r"independent chains $\omega_a$ supporting an answer", labelpad=1.5)
    ax.set_ylabel("answers", labelpad=2)
    ax.set_xticks(ks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.text(0.99, 0.98, r"$%d/%d$ answers certify nothing" % (dead, n),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    save(fig, "fig_omega")


if __name__ == "__main__":
    print("figures:")
    fig_independence()
    fig_trap()
    fig_designs()
    fig_omega()
