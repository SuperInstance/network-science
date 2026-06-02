//! Centrality measures.

use crate::graph::{Graph, DirectedGraph};
use std::collections::{HashMap, HashSet, VecDeque};

/// Degree centrality for all nodes. Returns normalized values (degree / (n-1)).
pub fn degree_centrality(graph: &Graph) -> HashMap<usize, f64> {
    let n = graph.node_count();
    if n <= 1 {
        return graph.nodes().into_iter().map(|n| (n, 0.0)).collect();
    }
    let norm = (n - 1) as f64;
    graph.nodes().into_iter().map(|node| {
        (node, graph.degree(node) as f64 / norm)
    }).collect()
}

/// Betweenness centrality using Brandes' algorithm.
pub fn betweenness_centrality(graph: &Graph) -> HashMap<usize, f64> {
    let nodes = graph.nodes();
    let n = nodes.len();
    let mut bc: HashMap<usize, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();

    for &s in &nodes {
        let mut stack: Vec<usize> = Vec::new();
        let mut sigma: HashMap<usize, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
        sigma.insert(s, 1.0);
        let mut dist: HashMap<usize, f64> = nodes.iter().map(|&v| (v, -1.0)).collect();
        dist.insert(s, 0.0);
        let mut pred: HashMap<usize, Vec<usize>> = nodes.iter().map(|&v| (v, Vec::new())).collect();
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in graph.adjacency().get(&v).unwrap_or(&HashSet::new()) {
                if dist[&w] < 0.0 {
                    dist.insert(w, dist[&v] + 1.0);
                    queue.push_back(w);
                }
                if (dist[&w] - dist[&v] - 1.0).abs() < 1e-10 {
                    *sigma.get_mut(&w).unwrap() += sigma[&v];
                    pred.get_mut(&w).unwrap().push(v);
                }
            }
        }

        let mut delta: HashMap<usize, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
        while let Some(w) = stack.pop() {
            for &v in &pred[&w] {
                let ratio = sigma[&v] / sigma[&w];
                *delta.get_mut(&v).unwrap() += ratio * (1.0 + delta[&w]);
            }
            if w != s {
                *bc.get_mut(&w).unwrap() += delta[&w];
            }
        }
    }

    // Normalize for undirected graph
    if n > 2 {
        let norm = ((n - 1) * (n - 2)) as f64 / 2.0;
        if norm > 0.0 {
            for v in bc.values_mut() {
                *v /= norm;
            }
        }
    }

    bc
}

/// Closeness centrality for all nodes.
pub fn closeness_centrality(graph: &Graph) -> HashMap<usize, f64> {
    let n = graph.node_count();
    if n <= 1 {
        return graph.nodes().into_iter().map(|n| (n, 0.0)).collect();
    }

    let nodes = graph.nodes();
    let mut result = HashMap::new();
    for &node in &nodes {
        let dists = graph.bfs_distances(node);
        let reachable = dists.len() - 1; // exclude self
        if reachable == 0 {
            result.insert(node, 0.0);
            continue;
        }
        let sum_d: usize = dists.values().sum::<usize>();
        if sum_d == 0 {
            result.insert(node, 0.0);
        } else {
            result.insert(node, reachable as f64 / sum_d as f64);
        }
    }
    result
}

/// Eigenvector centrality using power iteration.
pub fn eigenvector_centrality(graph: &Graph, max_iter: usize, tolerance: f64) -> HashMap<usize, f64> {
    let nodes = graph.nodes();
    let n = nodes.len();
    if n == 0 { return HashMap::new(); }

    let mut x: HashMap<usize, f64> = nodes.iter().map(|&v| (v, 1.0)).collect();

    for _ in 0..max_iter {
        let mut x_new: HashMap<usize, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
        for &v in &nodes {
            for &w in graph.adjacency().get(&v).unwrap_or(&HashSet::new()) {
                *x_new.get_mut(&w).unwrap() += x[&v];
            }
        }

        // Normalize
        let norm: f64 = x_new.values().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 { break; }
        for val in x_new.values_mut() {
            *val /= norm;
        }

        // Check convergence
        let diff: f64 = nodes.iter().map(|&v| (x[&v] - x_new[&v]).abs()).sum::<f64>() / n as f64;
        x = x_new;
        if diff < tolerance { break; }
    }

    x
}

/// PageRank for an undirected graph (treats as directed with bidirectional edges).
pub fn pagerank(graph: &Graph, damping: f64, max_iter: usize, tolerance: f64) -> HashMap<usize, f64> {
    let nodes = graph.nodes();
    let n = nodes.len();
    if n == 0 { return HashMap::new(); }

    let uniform = 1.0 / n as f64;
    let mut pr: HashMap<usize, f64> = nodes.iter().map(|&v| (v, uniform)).collect();

    for _ in 0..max_iter {
        let mut new_pr: HashMap<usize, f64> = nodes.iter().map(|&v| (v, (1.0 - damping) / n as f64)).collect();

        for &v in &nodes {
            let degree = graph.degree(v);
            if degree == 0 { continue; }
            let share = damping * pr[&v] / degree as f64;
            for &w in graph.adjacency().get(&v).unwrap_or(&HashSet::new()) {
                *new_pr.get_mut(&w).unwrap() += share;
            }
        }

        let diff: f64 = nodes.iter().map(|&v| (pr[&v] - new_pr[&v]).abs()).sum::<f64>();
        pr = new_pr;
        if diff < tolerance { break; }
    }

    pr
}

/// PageRank for a directed graph.
pub fn pagerank_directed(dg: &DirectedGraph, damping: f64, max_iter: usize, tolerance: f64) -> HashMap<usize, f64> {
    let nodes = dg.nodes();
    let n = nodes.len();
    if n == 0 { return HashMap::new(); }

    let uniform = 1.0 / n as f64;
    let mut pr: HashMap<usize, f64> = nodes.iter().map(|&v| (v, uniform)).collect();

    for _ in 0..max_iter {
        let mut new_pr: HashMap<usize, f64> = nodes.iter().map(|&v| (v, (1.0 - damping) / n as f64)).collect();

        for &v in &nodes {
            let out_deg = dg.out_degree(v);
            if out_deg == 0 {
                // Dangling node: distribute to all
                let share = damping * pr[&v] / n as f64;
                for &w in &nodes {
                    *new_pr.get_mut(&w).unwrap() += share;
                }
            } else {
                let share = damping * pr[&v] / out_deg as f64;
                for &w in dg.out_edges_map().get(&v).unwrap() {
                    *new_pr.get_mut(&w).unwrap() += share;
                }
            }
        }

        let diff: f64 = nodes.iter().map(|&v| (pr[&v] - new_pr[&v]).abs()).sum::<f64>();
        pr = new_pr;
        if diff < tolerance { break; }
    }

    pr
}
