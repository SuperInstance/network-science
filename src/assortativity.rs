//! Assortativity and mixing patterns.

use crate::graph::Graph;
use std::collections::HashMap;

/// Degree assortativity coefficient (Pearson correlation of degrees at edge endpoints).
pub fn degree_assortativity(graph: &Graph) -> f64 {
    let edges = graph.edges();
    if edges.is_empty() { return 0.0; }

    let m = edges.len() as f64;
    let mut sum_jk = 0.0; // sum of j*k
    let mut sum_j_plus_k = 0.0; // sum of (j+k)
    let mut sum_sq = 0.0; // sum of j^2 + k^2

    for &(u, v) in &edges {
        let j = graph.degree(u) as f64;
        let k = graph.degree(v) as f64;
        sum_jk += j * k;
        sum_j_plus_k += j + k;
        sum_sq += j * j + k * k;
    }

    // Pearson correlation of degrees at edge endpoints
    let num = (1.0 / m) * sum_jk - ((1.0 / (2.0 * m)) * sum_j_plus_k).powi(2);
    let den = (1.0 / (2.0 * m)) * sum_sq - ((1.0 / (2.0 * m)) * sum_j_plus_k).powi(2);

    if den == 0.0 { return 0.0; }
    num / den
}

/// Mixing matrix: e[i][j] = fraction of edges connecting degree i to degree j.
/// Returns (matrix, degree_values) where degree_values indexes the matrix.
pub fn mixing_matrix(graph: &Graph) -> (Vec<Vec<f64>>, Vec<usize>) {
    let edges = graph.edges();
    if edges.is_empty() { return (vec![], vec![]); }

    // Get unique degree values
    let mut degree_set: Vec<usize> = graph.nodes().iter().map(|&n| graph.degree(n)).collect();
    degree_set.sort();
    degree_set.dedup();

    let n = degree_set.len();
    let mut matrix = vec![vec![0usize; n]; n];
    let deg_idx: HashMap<usize, usize> = degree_set.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    for &(u, v) in &edges {
        let du = graph.degree(u);
        let dv = graph.degree(v);
        let iu = deg_idx[&du];
        let iv = deg_idx[&dv];
        matrix[iu][iv] += 1;
        if iu != iv {
            matrix[iv][iu] += 1;
        }
    }

    let total = edges.len() as f64;
    let float_matrix: Vec<Vec<f64>> = matrix.iter()
        .map(|row| row.iter().map(|&v| v as f64 / total).collect())
        .collect();

    (float_matrix, degree_set)
}

/// Average neighbor degree for each node.
pub fn average_neighbor_degree(graph: &Graph) -> HashMap<usize, f64> {
    let nodes = graph.nodes();
    nodes.into_iter().map(|node| {
        let neighbors = graph.neighbors(node);
        if neighbors.is_empty() {
            (node, 0.0)
        } else {
            let sum: usize = neighbors.iter().map(|&nb| graph.degree(nb)).sum();
            (node, sum as f64 / neighbors.len() as f64)
        }
    }).collect()
}

/// knn(k): average nearest neighbor degree for nodes of degree k.
pub fn knn(graph: &Graph) -> HashMap<usize, f64> {
    let ann = average_neighbor_degree(graph);
    let mut groups: HashMap<usize, Vec<f64>> = HashMap::new();
    for (&node, &avg_nb_deg) in &ann {
        let k = graph.degree(node);
        groups.entry(k).or_default().push(avg_nb_deg);
    }
    groups.into_iter().map(|(k, vals)| {
        let avg = vals.iter().sum::<f64>() / vals.len() as f64;
        (k, avg)
    }).collect()
}
