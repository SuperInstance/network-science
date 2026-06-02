//! Small-world properties.

use crate::graph::Graph;
use std::collections::HashMap;

/// Average shortest path length in the graph (over the largest component).
pub fn average_path_length(graph: &Graph) -> f64 {
    let lc = graph.largest_component();
    let nodes = lc.nodes();
    let n = nodes.len();
    if n <= 1 { return 0.0; }

    let mut total = 0usize;
    let mut count = 0usize;
    for &node in &nodes {
        let dists = lc.bfs_distances(node);
        for (&other, &d) in &dists {
            if other != node {
                total += d;
                count += 1;
            }
        }
    }

    if count == 0 { return 0.0; }
    // Each pair counted twice
    total as f64 / count as f64
}

/// Local clustering coefficient for a single node.
pub fn local_clustering_coefficient(graph: &Graph, node: usize) -> f64 {
    let neighbors = graph.neighbors(node);
    let k = neighbors.len();
    if k < 2 { return 0.0; }

    let neighbor_set: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
    let mut triangles = 0usize;
    for &u in &neighbors {
        for v in graph.neighbors(u) {
            if neighbor_set.contains(&v) {
                triangles += 1;
            }
        }
    }

    // Each triangle counted 2 times (u->v and v->u), divide by 2
    triangles as f64 / (k * (k - 1)) as f64
}

/// Global (average) clustering coefficient.
pub fn clustering_coefficient(graph: &Graph) -> f64 {
    let nodes = graph.nodes();
    if nodes.is_empty() { return 0.0; }

    let sum: f64 = nodes.iter().map(|&n| local_clustering_coefficient(graph, n)).sum();
    sum / nodes.len() as f64
}

/// Transitivity (global clustering coefficient) — ratio of triangles to triplets.
pub fn transitivity(graph: &Graph) -> f64 {
    let nodes = graph.nodes();
    let mut triangles = 0usize;
    let mut triplets = 0usize;

    for &v in &nodes {
        let neighbors = graph.neighbors(v);
        let k = neighbors.len();
        if k < 2 { continue; }
        triplets += k * (k - 1) / 2;

        let neighbor_set: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.has_edge(neighbors[i], neighbors[j]) {
                    triangles += 1;
                }
            }
        }
    }

    if triplets == 0 { return 0.0; }
    // triangles is counted once per vertex (3 times per actual triangle)
    // transitivity = 3 * actual_triangles / connected_triples = triangles / triplets
    triangles as f64 / triplets as f64
}

/// Small-world coefficient: sigma = (C/C_random) / (L/L_random).
/// Compares the graph's clustering and path length to a random graph with same n and m.
pub fn small_world_coefficient(graph: &Graph) -> f64 {
    let n = graph.node_count();
    let m = graph.edge_count();
    if n < 2 || m == 0 { return 0.0; }

    let c = clustering_coefficient(graph);
    let l = average_path_length(graph);

    // Random graph expectations
    let p = 2.0 * m as f64 / (n * (n - 1)) as f64;
    let c_random = p; // E[C] ≈ p for random graph
    let l_random = n as f64; // approximation

    if c_random == 0.0 || l_random == 0.0 { return 0.0; }
    if l == 0.0 { return 0.0; }

    (c / c_random) / (l / l_random)
}

/// Compute both C/C_random and L/L_random.
pub fn small_world_metrics(graph: &Graph) -> (f64, f64, f64) {
    let n = graph.node_count();
    let m = graph.edge_count();
    if n < 2 || m == 0 { return (0.0, 0.0, 0.0); }

    let c = clustering_coefficient(graph);
    let l = average_path_length(graph);
    let p = 2.0 * m as f64 / (n * (n - 1)) as f64;
    let c_random = p;
    let l_random = n as f64;

    let gamma = if c_random > 0.0 { c / c_random } else { 0.0 };
    let lambda = if l_random > 0.0 && l > 0.0 { l / l_random } else { 0.0 };
    let sigma = if lambda > 0.0 { gamma / lambda } else { 0.0 };

    (sigma, gamma, lambda)
}
