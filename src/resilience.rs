//! Network resilience: percolation, attacks.

use crate::graph::Graph;
use std::collections::HashMap;

/// Result of a resilience analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResilienceResult {
    /// Fraction of nodes/edges removed at each step.
    pub fraction_removed: Vec<f64>,
    /// Size of largest component at each step (as fraction of original).
    pub largest_component_fraction: Vec<f64>,
    /// Whether the giant component exists at each step.
    pub giant_component_exists: Vec<bool>,
}

/// Random node percolation: remove nodes one by one in random order.
pub fn node_percolation(graph: &Graph, steps: usize) -> ResilienceResult {
    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;

    let original_n = graph.node_count();
    if original_n == 0 {
        return ResilienceResult {
            fraction_removed: vec![],
            largest_component_fraction: vec![],
            giant_component_exists: vec![],
        };
    }

    let mut g = graph.clone();
    let mut nodes: Vec<usize> = g.nodes();
    nodes.shuffle(&mut rng);

    let giant_threshold = original_n as f64 * 0.05; // 5% of original as threshold
    let mut result = ResilienceResult {
        fraction_removed: vec![0.0],
        largest_component_fraction: vec![1.0],
        giant_component_exists: vec![true],
    };

    let step_size = (original_n / steps).max(1);
    for i in (0..nodes.len()).step_by(step_size) {
        let end = (i + step_size).min(nodes.len());
        for j in i..end {
            g.remove_node(nodes[j]);
        }
        let lc_size = g.largest_component().node_count() as f64;
        let frac_removed = end as f64 / original_n as f64;
        result.fraction_removed.push(frac_removed);
        result.largest_component_fraction.push(lc_size / original_n as f64);
        result.giant_component_exists.push(lc_size > giant_threshold);
    }

    result
}

/// Random edge percolation.
pub fn edge_percolation(graph: &Graph, steps: usize) -> ResilienceResult {
    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;

    let original_n = graph.node_count();
    let original_m = graph.edge_count();
    if original_m == 0 {
        return ResilienceResult {
            fraction_removed: vec![],
            largest_component_fraction: vec![],
            giant_component_exists: vec![],
        };
    }

    let mut g = graph.clone();
    let mut edges: Vec<(usize, usize)> = g.edges();
    edges.shuffle(&mut rng);

    let giant_threshold = original_n as f64 * 0.05;
    let mut result = ResilienceResult {
        fraction_removed: vec![0.0],
        largest_component_fraction: vec![1.0],
        giant_component_exists: vec![true],
    };

    let step_size = (original_m / steps).max(1);
    for i in (0..edges.len()).step_by(step_size) {
        let end = (i + step_size).min(edges.len());
        for j in i..end {
            g.remove_edge(edges[j].0, edges[j].1);
        }
        let lc_size = g.largest_component().node_count() as f64;
        let frac_removed = end as f64 / original_m as f64;
        result.fraction_removed.push(frac_removed);
        result.largest_component_fraction.push(lc_size / original_n as f64);
        result.giant_component_exists.push(lc_size > giant_threshold);
    }

    result
}

/// Targeted attack: remove nodes in order of decreasing degree.
pub fn targeted_attack(graph: &Graph, steps: usize) -> ResilienceResult {
    let original_n = graph.node_count();
    if original_n == 0 {
        return ResilienceResult {
            fraction_removed: vec![],
            largest_component_fraction: vec![],
            giant_component_exists: vec![],
        };
    }

    let mut g = graph.clone();
    let mut nodes: Vec<usize> = g.nodes();
    // Sort by degree descending
    nodes.sort_by(|&a, &b| g.degree(b).cmp(&g.degree(a)));

    let giant_threshold = original_n as f64 * 0.05;
    let mut result = ResilienceResult {
        fraction_removed: vec![0.0],
        largest_component_fraction: vec![1.0],
        giant_component_exists: vec![true],
    };

    let step_size = (original_n / steps).max(1);
    let mut removed = 0;
    for chunk in nodes.chunks(step_size) {
        for &node in chunk {
            g.remove_node(node);
            removed += 1;
        }
        let lc_size = g.largest_component().node_count() as f64;
        result.fraction_removed.push(removed as f64 / original_n as f64);
        result.largest_component_fraction.push(lc_size / original_n as f64);
        result.giant_component_exists.push(lc_size > giant_threshold);
    }

    result
}

/// Critical threshold for random failure (Molloy-Reed criterion).
/// Returns the fraction of nodes that must be removed to destroy the giant component.
pub fn critical_threshold(graph: &Graph) -> f64 {
    let degrees: Vec<f64> = graph.nodes().iter().map(|&n| graph.degree(n) as f64).collect();
    let n = degrees.len() as f64;
    if n == 0.0 { return 0.0; }

    let mean_k = degrees.iter().sum::<f64>() / n;
    let mean_k_sq = degrees.iter().map(|k| k * k).sum::<f64>() / n;

    if mean_k == 0.0 { return 0.0; }
    let kappa = mean_k_sq / mean_k; // <k²>/<k>

    // Threshold: 1 - 1/(kappa - 1)
    if kappa <= 1.0 { return 0.0; }
    1.0 - 1.0 / (kappa - 1.0)
}
