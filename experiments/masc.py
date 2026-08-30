"""
masc.py — core library for the MASC / replication-tradeoff experiments.

Everything here is synthetic. Goal: validate or falsify the
three theory claims with measured numbers.

  E2  motif spread sigma_M : exact for directed triangles via sparse A^3.
  E3  sample complexity    : #edge-samples to collect a planted k-cycle,
                             uniform vs degree-weighted vs PPR-weighted.
  E4  replication certificate: empirical verdict-flip threshold vs  mu > m_d + rho*m_s.

No global k-cycle counting for k>3 (it is #P-hard and would dominate runtime); the
k-dependence is instead exercised by E3's planted-motif retrieval, which needs no count.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# Graph generators (all simple digraphs, no self-loops)                       #
# --------------------------------------------------------------------------- #
def _clean_digraph(G: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph(G)                       # collapse multiedges
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def gen_scale_free(n: int, seed: int, alpha=0.41, beta=0.54, gamma=0.05) -> nx.DiGraph:
    """Directed scale-free (Bollobas et al.) — power-law in/out degrees, tau in (2,3)."""
    G = nx.scale_free_graph(n, alpha=alpha, beta=beta, gamma=gamma, seed=seed)
    return _clean_digraph(G)


def gen_ba_oriented(n: int, m: int, seed: int) -> nx.DiGraph:
    """Barabasi-Albert backbone, each edge oriented low-id -> high-id then a random
    fraction reversed, so directed cycles can form. Heavy-tailed degree."""
    rng = np.random.default_rng(seed)
    U = nx.barabasi_albert_graph(n, m, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for a, b in U.edges():
        if rng.random() < 0.5:
            G.add_edge(a, b)
        else:
            G.add_edge(b, a)
    return _clean_digraph(G)


def gen_chung_lu(n: int, tau: float, seed: int, avg_deg: float = 6.0) -> nx.DiGraph:
    """Chung-Lu expected-degree model with a power-law weight sequence, exponent tau."""
    rng = np.random.default_rng(seed)
    # power-law weights w_i ~ i^{-1/(tau-1)}, rescaled to mean avg_deg
    i = np.arange(1, n + 1)
    w = i ** (-1.0 / (tau - 1.0))
    w = w * (avg_deg * n / w.sum())
    w = np.clip(w, 0, np.sqrt(w.sum()))          # remove self-loop bias (standard)
    S = w.sum()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # sample edges proportional to w_i * w_j / S  (sparse: only sample ~ avg_deg*n)
    n_edges = int(avg_deg * n)
    p = w / w.sum()
    src = rng.choice(n, size=n_edges, p=p)
    dst = rng.choice(n, size=n_edges, p=p)
    for a, b in zip(src, dst):
        if a != b:
            G.add_edge(int(a), int(b))
    return _clean_digraph(G)


def gen_er(n: int, avg_deg: float, seed: int) -> nx.DiGraph:
    """Directed Erdos-Renyi control with matched average degree."""
    p = avg_deg / (n - 1)
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    return _clean_digraph(G)


# --------------------------------------------------------------------------- #
# E2 — motif spread for directed triangles (exact, sparse A^3)                 #
# --------------------------------------------------------------------------- #
def directed_triangle_spread(G: nx.DiGraph):
    """Return (sigma_M, n_triangles, max_edge_count, n_edges) for the directed
    3-cycle motif class. Exact:  count(u->v) = A[u,v] * (A^2)[v,u]."""
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    N = len(nodes)
    rows = [idx[u] for u, _ in G.edges()]
    cols = [idx[v] for _, v in G.edges()]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A2 = (A @ A).tocsr()
    # per-edge triangle count
    counts = []
    for u, v in G.edges():
        c = A2[idx[v], idx[u]]                      # # w : v->w->u
        counts.append(float(c))
    counts = np.asarray(counts)
    n_edges = len(counts)
    total = counts.sum() / 3.0                       # each triangle counted 3x
    if total <= 0:
        return None                                  # no triangles; spread undefined
    avg_per_edge = total / n_edges
    sigma = counts.max() / avg_per_edge
    return dict(sigma_M=float(sigma), n_triangles=float(total),
                max_edge_count=float(counts.max()), n_edges=int(n_edges))


# --------------------------------------------------------------------------- #
# E3 — planted k-cycle + edge samplers + samples-to-collect                    #
# --------------------------------------------------------------------------- #
def make_branching_with_cycle(w: int, k: int, seed: int):
    """Controlled graph: a depth-k out-branching tree from anchor v (so |E_k(v)| ~ w^k),
    PLUS one planted directed k-cycle through v using near edges.
    Returns (G, anchor, planted_edge_list)."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    v = 0
    G.add_node(v)
    # build a w-ary out-tree of depth k (capped so it stays runnable)
    frontier = [v]
    nxt = 1
    max_nodes = 20000
    for _ in range(k):
        new_frontier = []
        for node in frontier:
            for _ in range(w):
                if nxt >= max_nodes:
                    break
                G.add_edge(node, nxt)
                new_frontier.append(nxt)
                nxt += 1
        frontier = new_frontier
        if nxt >= max_nodes:
            break
    # plant a directed k-cycle through v along a random root-to-depth chain + back edge
    chain = [v]
    cur = v
    for _ in range(k - 1):
        succ = list(G.successors(cur))
        if not succ:
            break
        cur = rng.choice(succ)
        chain.append(int(cur))
    # close cycle: last -> v
    G.add_edge(chain[-1], v)
    planted = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)] + [(chain[-1], v)]
    return G, v, planted


def make_hub_concentrated(H: int, n_decoy: int, k: int, seed: int):
    """FAVORABLE regime for targeted sampling: anchor v points to H hubs that carry the
    planted k-cycle (all cycle tails are 1 hop from v -> high PPR, no depth decay), while
    n_decoy low-degree nodes hang off the hubs to inflate |E_k(v)| (so uniform must search
    a large ball). Returns (G, v, planted_edges)."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    v = 0
    hubs = list(range(1, H + 1))
    G.add_node(v)
    for h in hubs:
        G.add_edge(v, h)                       # v -> every hub (hubs at depth 1)
    # planted directed k-cycle through v using hubs:  v -> h1 -> h2 -> ... -> h_{k-1} -> v
    chain = [v] + [hubs[i % H] for i in range(k - 1)]
    planted = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)] + [(chain[-1], v)]
    G.add_edges_from(planted)
    # decoys inflate the ball with low-PPR edges
    nid = H + 1
    for _ in range(n_decoy):
        h = int(rng.choice(hubs))
        G.add_edge(h, nid)
        if rng.random() < 0.5 and nid > H + 2:
            G.add_edge(nid, int(rng.integers(H + 1, nid)))   # decoy-decoy edges, depth>=2
        nid += 1
    return G, v, planted


def plant_cycle_in(G: nx.DiGraph, v, k: int, seed: int):
    """Plant a directed k-cycle through existing anchor v using nearby nodes; returns edges."""
    rng = np.random.default_rng(seed)
    # walk out k-1 steps following existing edges where possible, else pick random nodes
    nodes = list(G.nodes())
    chain = [v]
    cur = v
    for _ in range(k - 1):
        succ = [s for s in G.successors(cur) if s not in chain]
        cur = int(rng.choice(succ)) if succ else int(rng.choice(nodes))
        chain.append(cur)
    edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)] + [(chain[-1], v)]
    G.add_edges_from(edges)
    return edges


def ball_edges(G: nx.DiGraph, v, r: int):
    """Edges with both endpoints in the directed r-ball B_r(v) (out-reachable <= r)."""
    dist = nx.single_source_shortest_path_length(G, v, cutoff=r)
    ball = set(dist)
    return [(a, b) for a, b in G.edges() if a in ball and b in ball], ball


def edge_distribution(G: nx.DiGraph, v, r: int, mode: str, alpha: float = 0.15):
    """Return (edges, prob) over B_r(v) for mode in {uniform, degree, ppr}."""
    edges, ball = ball_edges(G, v, r)
    if not edges:
        return edges, np.array([])
    if mode == "uniform":
        p = np.ones(len(edges))
    elif mode == "degree":
        p = np.array([G.out_degree(a) + G.in_degree(a) +
                      G.out_degree(b) + G.in_degree(b) for a, b in edges], float)
    elif mode == "ppr":
        sub = G.subgraph(ball)
        ppr = nx.pagerank(sub, alpha=1 - alpha, personalization={v: 1.0})
        # probability the personalized walk traverses (a,b): ppr[a]/outdeg(a)
        p = np.array([ppr.get(a, 0.0) / max(1, G.out_degree(a)) for a, b in edges], float)
        p = p + 1e-12
    elif mode == "bippr":
        # bidirectional PPR: forward PPR from v + backward PPR into v.
        # An edge on a short cycle through v has a high-mass tail (reachable from v) AND a
        # high-mass head (reaching v), so it scores high even when antipodal to v.
        # IMPORTANT: forward PPR is computed on the out-ball; backward PPR MUST be computed on
        # the IN-ball (nodes that can REACH v within r) -- computing it on the out-ball hides
        # competing backward branches and artifactually concentrates mass on the cycle.
        sub = G.subgraph(ball)
        ppr_f = nx.pagerank(sub, alpha=1 - alpha, personalization={v: 1.0})
        Grev = G.reverse(copy=True)
        din = nx.single_source_shortest_path_length(Grev, v, cutoff=r)
        inball = set(din)
        subR = Grev.subgraph(inball)
        ppr_b = (nx.pagerank(subR, alpha=1 - alpha, personalization={v: 1.0})
                 if subR.number_of_nodes() > 1 else {v: 1.0})
        p = np.array([ppr_f.get(a, 0.0) / max(1, G.out_degree(a)) +
                      ppr_b.get(b, 0.0) / max(1, G.in_degree(b)) for a, b in edges], float)
        p = p + 1e-12
    else:
        raise ValueError(mode)
    p = p / p.sum()
    return edges, p


def min_motif_mass(edges, probs, targets):
    """p_min: the smallest sampling probability assigned to any motif (target) edge.
    samples-to-collect is governed by ~ (1/p_min)*ln(k). Returns (p_min, all_present)."""
    eidx = {e: i for i, e in enumerate(edges)}
    masses = [probs[eidx[t]] for t in targets if t in eidx]
    if len(masses) < len(targets):
        return 0.0, False
    return float(min(masses)), True


def samples_to_collect(edges, probs, targets, rng, cap_mult=50):
    """Coupon-collector: draw iid edges ~ probs until all `targets` seen. Return draw count
    (capped at cap_mult * |edges|; capped runs flagged)."""
    if len(edges) == 0:
        return None, True
    eidx = {e: i for i, e in enumerate(edges)}
    tset = set(eidx[t] for t in targets if t in eidx)
    if not tset:
        return None, True                       # target edge not even in ball
    cap = cap_mult * len(edges)
    seen = set()
    draws = 0
    n = len(edges)
    idxs = np.arange(n)
    while seen != tset and draws < cap:
        batch = rng.choice(idxs, size=min(256, cap - draws), p=probs)
        for j in batch:
            draws += 1
            if j in tset:
                seen.add(j)
            if seen == tset:
                break
    return draws, (seen != tset)


# --------------------------------------------------------------------------- #
# E4 — rho-replicated K-partition certificate simulation                       #
# --------------------------------------------------------------------------- #
def clean_detection_prob(K, rho, d, seed, trials=2000):
    """E4a (the tradeoff): probability the rho-replicated K-cell defense detects a clean
    motif. Each of the d motif edges is hashed to rho of K cells; a cell fires iff it holds
    ALL d edges; the motif is 'detected' iff >=1 cell fires. Theory: P[a cell fires] =
    prod over edges of (rho/K) ~ (rho/K)^d  ->  detection needs rho ~ K^{1-1/d}."""
    rng = np.random.default_rng(seed)
    det = 0
    fired_counts = []
    for _ in range(trials):
        present = np.ones((K,), dtype=bool)
        for _e in range(d):
            cells = rng.choice(K, size=min(rho, K), replace=False)
            mask = np.zeros(K, dtype=bool); mask[cells] = True
            present &= mask
        nf = int(present.sum())
        fired_counts.append(nf)
        if nf >= 1:
            det += 1
    return dict(K=K, rho=rho, d=d, detect_prob=det / trials,
                mean_firing_cells=float(np.mean(fired_counts)),
                predicted_firing=K * (rho / K) ** d)


def flip_vs_budget(K, mu, rho, m_d, m_s, seed, adversary="optimal", trials=400):
    """E4b (tightness): start from a clean positive verdict with margin mu (so c1 = floor(K/2)+mu
    cells fire). Apply a poison budget: m_d single-cell flips + m_s structural poisons, each
    structural poison landing in rho cells. Measure empirical verdict-flip rate.

      optimal  adversary (white-box, cell-deterministic): aims every flip at a firing cell ->
               flips min(m_d + rho*m_s, c1) firing cells.  Certificate: stable iff mu > m_d+rho*m_s.
      random   adversary: structural poisons hit rho RANDOM cells (some wasted on non-firing
               or overlapping cells) -> needs more budget; shows the certificate is conservative.
    """
    rng = np.random.default_rng(seed)
    c1 = K // 2 + mu                               # firing cells (the positive-voting majority)
    flips = 0
    for _ in range(trials):
        firing = np.zeros(K, dtype=bool); firing[:c1] = True
        if adversary == "optimal":
            removable = min(m_d + rho * m_s, c1)   # all budget aimed at firing cells
            c1_adv = c1 - removable
        else:  # random
            silenced = np.zeros(K, dtype=bool)
            # document poisons: pick random cells
            for _ in range(m_d):
                silenced[int(rng.integers(K))] = True
            # structural poisons: each hits rho random cells
            for _ in range(m_s):
                cells = rng.choice(K, size=min(rho, K), replace=False)
                silenced[cells] = True
            c1_adv = int((firing & ~silenced).sum())
        adv_pos = c1_adv > K / 2
        if not adv_pos:                            # clean was positive (c1>K/2 since mu>=1); flipped
            flips += 1
    return dict(K=K, mu=mu, rho=rho, m_d=m_d, m_s=m_s, adversary=adversary,
                flip_rate=flips / trials, budget=m_d + rho * m_s,
                cert_stable=(mu > m_d + rho * m_s))
