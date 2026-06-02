//! Network generation models.

use crate::graph::{Graph, DirectedGraph};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

/// Erdős-Rényi G(n, p) model. Each edge exists independently with probability p.
pub fn erdos_renyi(n: usize, p: f64) -> Graph {
    let mut g = Graph::with_n_nodes(n);
    let mut rng = rand::thread_rng();
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                g.add_edge(u, v);
            }
        }
    }
    g
}

/// Erdős-Rényi G(n, M) model — exactly M random edges.
pub fn erdos_renyi_gnm(n: usize, m: usize) -> Graph {
    let mut g = Graph::with_n_nodes(n);
    let mut rng = rand::thread_rng();
    let max_edges = n * (n - 1) / 2;
    let m = m.min(max_edges);
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(m);
    let mut used = HashSet::new();
    while edges.len() < m {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        if u == v { continue; }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if used.insert((a, b)) {
            edges.push((a, b));
        }
    }
    for (u, v) in edges {
        g.add_edge(u, v);
    }
    g
}

/// Barabási-Albert preferential attachment model.
/// Start with m0 nodes, add nodes one at a time, each connecting to m existing nodes
/// with probability proportional to their degree.
pub fn barabasi_albert(n: usize, m: usize) -> Graph {
    if n == 0 || m == 0 { return Graph::new(); }
    let m = m.min(n - 1);
    let m0 = m + 1; // initial clique size

    let mut g = Graph::with_n_nodes(m0);
    // Start as a clique
    for u in 0..m0 {
        for v in (u + 1)..m0 {
            g.add_edge(u, v);
        }
    }

    // Repeated indices list for weighted sampling
    let mut repeated: Vec<usize> = Vec::new();
    for u in 0..m0 {
        for _ in 0..m0 - 1 { // each node has degree m0-1 in the clique
            repeated.push(u);
        }
    }

    let mut rng = rand::thread_rng();
    for new_node in m0..n {
        g.add_node(new_node);
        let mut targets = HashSet::new();
        while targets.len() < m {
            let pick = repeated[rng.gen_range(0..repeated.len())];
            targets.insert(pick);
        }
        for &t in &targets {
            g.add_edge(new_node, t);
            repeated.push(new_node);
            repeated.push(t);
        }
    }
    g
}

/// Watts-Strogatz small-world model.
/// Start with a ring lattice where each node connects to k nearest neighbors,
/// then rewire each edge with probability beta.
pub fn watts_strogatz(n: usize, k: usize, beta: f64) -> Graph {
    assert!(k % 2 == 0, "k must be even");
    let half_k = k / 2;
    let mut g = Graph::with_n_nodes(n);
    let mut rng = rand::thread_rng();

    // Track edges for potential rewiring: (u, v) with u < v
    let mut edge_list: Vec<(usize, usize)> = Vec::new();

    // Ring lattice
    for u in 0..n {
        for j in 1..=half_k {
            let v = (u + j) % n;
            g.add_edge(u, v);
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            edge_list.push((a, b));
        }
    }

    // Rewire
    for &(u, orig_v) in &edge_list {
        if rng.gen::<f64>() < beta {
            // Pick a random node w != u that is not already a neighbor
            let neighbors = g.neighbors(u);
            let neighbor_set: HashSet<usize> = neighbors.into_iter().collect();
            let mut candidates: Vec<usize> = (0..n)
                .filter(|&w| w != u && !neighbor_set.contains(&w))
                .collect();
            if candidates.is_empty() { continue; }
            candidates.shuffle(&mut rng);
            let w = candidates[0];
            g.remove_edge(u, orig_v);
            g.add_edge(u, w);
        }
    }
    g
}

/// Generate a random directed graph using Erdős-Rényi G(n, p).
pub fn directed_erdos_renyi(n: usize, p: f64) -> DirectedGraph {
    let mut g = DirectedGraph::with_n_nodes(n);
    let mut rng = rand::thread_rng();
    for u in 0..n {
        for v in 0..n {
            if u != v && rng.gen::<f64>() < p {
                g.add_edge(u, v);
            }
        }
    }
    g
}
