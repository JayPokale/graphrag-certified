"""kg_omega.py -- how much independent corroboration real knowledge graphs carry.

The certificate of Cor. menger is per answer: omega_a is the maximum number of
edge-disjoint typed support paths from the question entity to the answer, and the
budget is ceil(omega_a/2) - 1.  flow-check measured the distribution of omega_a on a
power grid.  This measures it where GraphRAG is actually deployed:

  metaqa    the 2-hop and 3-hop test questions of MetaQA over the WikiMovies KB
            (134,741 triples).  Each question template's relation path is inferred from
            the KB itself, as the typed path whose endpoint set best matches the gold
            answers (mean Jaccard over a sample), and every gold answer of every test
            question is scored.  Matching on the ANSWERS and not merely on connectivity
            is what separates the intended path from a hub path such as
            release_year / release_year^-1, which connects most pairs too.
  hetionet  Hetionet v1.0 (47,031 nodes, 2.25M typed edges), the Medical domain of
            the GragPoison benchmark, asked four metapath questions, among them the
            introduction's own "which compound binds a gene this gene regulates".

omega_a is exact.  A typed path makes the witness graph layered by hop, every path has
the same length, and no physical triple can sit at two hop positions once the question
entity and the answer are barred from intermediate layers, so a unit-capacity max-flow
counts edge-disjoint witnesses at any length (Cor. menger, typed case); two hops need
only a common-neighbour count.  The disjointness of the extracted paths is re-checked
on every answer and the number of violations is recorded (it is zero).

Writes results_kgomega.json.  CPU only; a few minutes.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 20260917
HUB_CAP = 3_000_000          # skip a (question, answer) whose layered graph would exceed this


# ==========================================================================
# max edge-disjoint paths in a unit-capacity layered DAG (Ford-Fulkerson, DFS)
# ==========================================================================
def max_disjoint_paths(succ, src, dst):
    """succ: node -> list of successors.  Returns (value, paths) with the paths as
    node lists, so the caller can re-check physical edge-disjointness."""
    flow = {}
    res_adj = defaultdict(list)
    for u, vs in succ.items():
        for v in vs:
            res_adj[u].append(v)
            res_adj[v].append(u)
    fwd = {u: set(vs) for u, vs in succ.items()}

    def cap(u, v):
        if v in fwd.get(u, ()):
            return 1 - flow.get((u, v), 0)
        return flow.get((v, u), 0)

    value = 0
    while True:
        parent = {src: None}
        stack = [src]
        found = False
        while stack and not found:
            u = stack.pop()
            for v in res_adj[u]:
                if v not in parent and cap(u, v) > 0:
                    parent[v] = u
                    if v == dst:
                        found = True
                        break
                    stack.append(v)
        if not found:
            break
        v = dst
        while parent[v] is not None:
            u = parent[v]
            if v in fwd.get(u, ()):
                flow[(u, v)] = flow.get((u, v), 0) + 1
            else:
                flow[(v, u)] -= 1
            v = u
        value += 1
    # decompose the flow into paths
    out = defaultdict(list)
    for (u, v), f in flow.items():
        if f > 0:
            out[u].append(v)
    paths = []
    for _ in range(value):
        p, u = [src], src
        while u != dst:
            v = out[u].pop()
            p.append(v)
            u = v
        paths.append(p)
    return value, paths


# ==========================================================================
# typed graph: typed[(x, rel, d)] = set of y with an edge x -(rel,d)- y
# ==========================================================================
class TypedGraph:
    def __init__(self):
        self.typed = defaultdict(set)
        self.nbrs = defaultdict(list)      # x -> [(rel, d, y)]
        self.n_edges = 0

    def add(self, s, r, o, directed=True):
        self.typed[(s, r, +1)].add(o)
        self.typed[(o, r, -1)].add(s)
        self.nbrs[s].append((r, +1, o))
        self.nbrs[o].append((r, -1, s))
        if not directed:                    # an undirected relation reads the same both ways
            self.typed[(s, r, -1)].add(o)
            self.typed[(o, r, +1)].add(s)
        self.n_edges += 1

    def step(self, x, hop):
        return self.typed.get((x,) + tuple(hop), ())

    def omega(self, v, a, path):
        """Max edge-disjoint typed paths v -> a along `path`, a list of (rel, d) hops,
        with v and a barred from intermediate layers.  Returns (omega, violations,
        skipped) where violations counts physical-triple reuse among extracted paths
        (must be 0) and skipped flags a hub instance beyond HUB_CAP."""
        h = len(path)
        if v == a:
            return 0, 0, False
        if h == 1:
            return (1 if a in self.step(v, path[0]) else 0), 0, False
        L1 = [x for x in self.step(v, path[0]) if x != v and x != a]
        if h == 2:
            last = path[1]
            return sum(1 for x in L1 if a in self.step(x, last)), 0, False
        # h >= 3: layered flow.  Layer h-1 is restricted to the in-neighbours of a.
        last = path[-1]
        Y = self.step(a, (last[0], -last[1]))
        Y = {y for y in Y if y != v and y != a}
        succ = defaultdict(list)
        frontier = {("s",): None}
        cur = {x: ("1", x) for x in L1}
        for x in L1:
            succ[("s",)].append(("1", x))
        size = len(L1)
        for i in range(1, h - 1):                      # layers 1 .. h-2 -> next layer
            hop = path[i]
            nxt = {}
            for x, node in cur.items():
                targets = self.step(x, hop)
                if i == h - 2:
                    targets = [y for y in targets if y in Y]
                else:
                    targets = [y for y in targets if y != v and y != a]
                size += len(targets)
                if size > HUB_CAP:
                    return None, 0, True
                for y in targets:
                    ynode = (str(i + 1), y)
                    succ[node].append(ynode)
                    nxt[y] = ynode
            cur = nxt
        for y, node in cur.items():
            succ[node].append(("t",))
        val, paths = max_disjoint_paths(succ, ("s",), ("t",))
        # physical-disjointness check: every hop of every path is a distinct triple
        seen, viol = set(), 0
        for p in paths:
            ents = [v] + [n[1] for n in p[1:-1]] + [a]
            for i in range(h):
                e = (ents[i], path[i][0], ents[i + 1]) if path[i][1] > 0 else (ents[i + 1], path[i][0], ents[i])
                if e in seen:
                    viol += 1
                seen.add(e)
        return val, viol, False


def summarize(omegas, label):
    om = np.array(omegas, dtype=int)
    hist = Counter(int(o) for o in om)
    n = len(om)
    return dict(
        n_answers=int(n),
        omega_hist={str(k): hist[k] for k in sorted(hist)},
        median=float(np.median(om)) if n else None,
        mean=float(np.mean(om)) if n else None,
        max=int(om.max()) if n else None,
        frac_omega_1=float((om == 1).mean()) if n else None,
        frac_certify_nothing=float((om <= 2).mean()) if n else None,   # ceil(omega/2)-1 = 0
        frac_certify_ge1=float((om >= 3).mean()) if n else None,
        frac_certify_ge3=float((om >= 7).mean()) if n else None,
        label=label)


# ==========================================================================
# MetaQA
# ==========================================================================
def load_metaqa_kb():
    G = TypedGraph()
    with open(os.path.join(DATA, "metaqa_kb.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            G.add(parts[0], parts[1], parts[2], directed=True)
    return G


def load_metaqa_questions(h):
    qs = []
    with open(os.path.join(DATA, f"metaqa_{h}-hop_test.txt"), encoding="utf-8") as f:
        for line in f:
            q, _, ans = line.rstrip("\n").partition("\t")
            m = re.search(r"\[(.*?)\]", q)
            if not m or not ans:
                continue
            qs.append(dict(text=q, topic=m.group(1), template=re.sub(r"\[.*?\]", "[E]", q),
                           answers=[a for a in ans.split("|") if a]))
    return qs


def typed_sequences(G, v, a, h):
    """All typed relation paths of length h from v to a (intermediates != v, a)."""
    seqs = set()
    nv, na = G.nbrs.get(v, []), G.nbrs.get(a, [])
    if h == 2:
        for r1, d1, x in nv:
            if x in (v, a):
                continue
            for r2, d2, y in G.nbrs.get(x, []):
                if y == a:
                    seqs.add(((r1, d1), (r2, d2)))
    elif h == 3:
        if len(nv) * len(na) > 400_000:
            return seqs
        ends = defaultdict(set)                     # y -> {(r3, d3)} hop into a
        for r3, d3, y in na:
            if y not in (v, a):
                ends[y].add((r3, -d3))
        for r1, d1, x in nv:
            if x in (v, a):
                continue
            for r2, d2, y in G.nbrs.get(x, []):
                if y in ends:
                    for hop3 in ends[y]:
                        seqs.add(((r1, d1), (r2, d2), hop3))
    return seqs


def reach(G, v, path, cap=200_000):
    """Every entity at the end of a typed path from v (intermediates != v)."""
    layer = {v}
    for hop in path:
        nxt = set()
        for x in layer:
            nxt.update(y for y in G.step(x, hop) if y != v)
            if len(nxt) > cap:
                return None
        layer = nxt
    return layer


def infer_template_paths(G, qs, h, sample=40, rng=None):
    """The relation path of a question template, inferred from the KB.  Candidates are
    the typed paths that connect sampled (topic, answer) pairs; the winner is the
    candidate whose endpoint set best matches the gold answers (mean Jaccard over the
    sample), which separates the intended path from a hub path such as
    release_year / release_year^-1 that also happens to connect most pairs."""
    by_t = defaultdict(list)
    for q in qs:
        by_t[q["template"]].append(q)
    out = {}
    for t, group in by_t.items():
        pick = group if len(group) <= sample else rng.sample(group, sample)
        cands = set()
        for q in pick:
            for a in q["answers"][:5]:
                cands |= typed_sequences(G, q["topic"], a, h)
        if not cands:
            out[t] = dict(path=None, support=0.0, n_questions=len(group))
            continue
        best, best_j = None, -1.0
        for c in cands:
            js = []
            for q in pick:
                R = reach(G, q["topic"], c)
                A = set(q["answers"])
                js.append(0.0 if R is None else len(R & A) / max(len(R | A), 1))
            j = float(np.mean(js))
            if j > best_j:
                best, best_j = c, j
        out[t] = dict(path=[list(x) for x in best], support=best_j, n_questions=len(group))
    return out


def run_metaqa(G, h, rng, max_questions=None):
    qs = load_metaqa_questions(h)
    paths = infer_template_paths(G, qs, h, rng=rng)
    scored = qs if max_questions is None or len(qs) <= max_questions else rng.sample(qs, max_questions)
    omegas, viol, skipped, no_path = [], 0, 0, 0
    q_any_cert, q_scored = 0, 0
    t0 = time.time()
    for q in scored:
        p = paths[q["template"]]["path"]
        if p is None:
            no_path += 1
            continue
        path = [tuple(x) for x in p]
        best = 0
        got = False
        for a in q["answers"]:
            if a == q["topic"]:
                continue
            w, vl, sk = G.omega(q["topic"], a, path)
            if sk:
                skipped += 1
                continue
            got = True
            omegas.append(w)
            viol += vl
            best = max(best, w)
        if got:
            q_scored += 1
            q_any_cert += (best >= 3)
    summ = summarize(omegas, f"MetaQA {h}-hop")
    summ.update(n_questions_total=len(qs), n_questions_scored=q_scored,
                n_templates=len(paths), templates=paths,
                frac_questions_with_a_certifiable_answer=(q_any_cert / q_scored if q_scored else None),
                disjointness_violations=viol, answers_skipped_hub=skipped,
                questions_without_inferred_path=no_path, seconds=round(time.time() - t0, 1))
    return summ


# ==========================================================================
# Hetionet
# ==========================================================================
UNDIRECTED = {"CrC", "DrD", "GiG", "GcG"}     # symmetric metaedges in Hetionet v1.0

METAPATHS = {
    # name: (question, anchor kind, hops)
    "CbGaD": ("which diseases are associated with a gene this compound binds",
              "Compound", [("CbG", +1), ("DaG", -1)]),
    "GrGbC": ("which compounds bind a gene this gene regulates",
              "Gene", [("Gr>G", +1), ("CbG", -1)]),
    "CrCtD": ("which diseases are treated by a compound resembling this one",
              "Compound", [("CrC", +1), ("CtD", +1)]),
    "CtDaGbC": ("which compounds bind a gene associated with a disease this compound treats",
                "Compound", [("CtD", +1), ("DaG", +1), ("CbG", -1)]),
}


def load_hetionet():
    G = TypedGraph()
    kinds = {}
    with open(os.path.join(DATA, "hetionet-nodes.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            nid, _, kind = line.rstrip("\n").split("\t")
            kinds[nid] = kind
    with gzip.open(os.path.join(DATA, "hetionet-edges.sif.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, me, o = line.rstrip("\n").split("\t")
            G.add(s, me, o, directed=me not in UNDIRECTED)
    return G, kinds


def run_hetionet(G, kinds, rng, anchors_per_path=300):
    out = {}
    for name, (question, akind, hops) in METAPATHS.items():
        t0 = time.time()
        anchors = sorted(x for x, k in kinds.items() if k == akind and G.step(x, hops[0]))
        if len(anchors) > anchors_per_path:
            anchors = sorted(rng.sample(anchors, anchors_per_path))
        omegas, viol, skipped, n_pairs = [], 0, 0, 0
        for v in anchors:
            # every answer reachable along the path
            layer = {v}
            for hop in hops:
                layer = {y for x in layer for y in G.step(x, hop) if y != v}
            for a in sorted(layer):
                w, vl, sk = G.omega(v, a, hops)
                if sk:
                    skipped += 1
                    continue
                omegas.append(w)
                viol += vl
                n_pairs += 1
        summ = summarize(omegas, f"Hetionet {name}")
        summ.update(question=question, anchor_kind=akind,
                    path=[list(h) for h in hops], n_anchors=len(anchors),
                    disjointness_violations=viol, answers_skipped_hub=skipped,
                    seconds=round(time.time() - t0, 1))
        out[name] = summ
        print(f"  {name:8s} anchors={len(anchors):4d} answers={summ['n_answers']:6d} "
              f"median={summ['median']:.0f} max={summ['max']} "
              f"omega=1: {100*summ['frac_omega_1']:.1f}%  certify>=1: "
              f"{100*summ['frac_certify_ge1']:.1f}%  ({summ['seconds']}s)", flush=True)
    return out


if __name__ == "__main__":
    rng = random.Random(SEED)
    res = dict(seed=SEED, hub_cap=HUB_CAP)

    print("MetaQA")
    G = load_metaqa_kb()
    res["metaqa"] = dict(kb_triples=G.n_edges, entities=len(G.nbrs))
    for h in (2, 3):
        s = run_metaqa(G, h, rng)
        res["metaqa"][f"{h}-hop"] = s
        print(f"  {h}-hop: {s['n_questions_scored']} questions, {s['n_answers']} answers, "
              f"median omega {s['median']:.0f}, max {s['max']}, omega=1: "
              f"{100*s['frac_omega_1']:.1f}%, certify>=1: {100*s['frac_certify_ge1']:.1f}%, "
              f"violations {s['disjointness_violations']}, skipped {s['answers_skipped_hub']} "
              f"({s['seconds']}s)", flush=True)
        for t, p in sorted(s["templates"].items(), key=lambda kv: -kv[1]["n_questions"]):
            print(f"      {p['n_questions']:5d}  support {p['support']:.2f}  {t}  ->  {p['path']}")

    print("Hetionet")
    H, kinds = load_hetionet()
    res["hetionet"] = dict(nodes=len(kinds), edges=H.n_edges,
                           metapaths=run_hetionet(H, kinds, rng))

    with open(os.path.join(HERE, "results_kgomega.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote results_kgomega.json")
