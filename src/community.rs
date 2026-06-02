//! Community detection algorithms.

use crate::graph::Graph;
use std::collections::HashMap;

/// Compute modularity of a partition.
/// `membership[node] = community_id`
pub fn modularity(graph: &Graph, membership: &HashMap<usize, usize>) -> f64 {
    let m = graph.edge_count() as f64;
    if m == 0.0 { return 0.0; }

    let nodes = graph.nodes();
    let mut k: HashMap<usize, f64> = HashMap::new();
    for &n in &nodes {
        k.insert(n, graph.degree(n) as f64);
    }

    let mut q = 0.0;
    for (&u, nbs) in graph.adjacency() {
        for &v in nbs {
            if membership.get(&u) == membership.get(&v) {
                q += 1.0 - k[&u] * k[&v] / (2.0 * m);
            }
        }
    }
    q / (2.0 * m)
}

/// Louvain community detection (first phase: local moves).
/// Returns a mapping from node to community ID.
pub fn louvain(graph: &Graph) -> HashMap<usize, usize> {
    let nodes = graph.nodes();
    if nodes.is_empty() { return HashMap::new(); }

    // Each node starts in its own community
    let mut membership: HashMap<usize, usize> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        membership.insert(node, i);
    }

    let m = graph.edge_count() as f64;
    if m == 0.0 { return membership; }

    let max_passes = 20;
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < max_passes {
        improved = false;
        passes += 1;
        for &node in &nodes {
            let current_comm = membership[&node];
            let neighbors = graph.neighbors(node);

            // Calculate sigma_tot and k_i_in for each neighboring community
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            let mut comm_sigma_tot: HashMap<usize, f64> = HashMap::new();

            // Get all neighboring communities
            let mut neighbor_comms: Vec<usize> = Vec::new();
            for &nb in &neighbors {
                let c = membership[&nb];
                *comm_weights.entry(c).or_insert(0.0) += 1.0;
                if !neighbor_comms.contains(&c) {
                    neighbor_comms.push(c);
                }
            }

            // Calculate sigma_tot for each community
            for &c in &neighbor_comms {
                let sigma: f64 = nodes.iter()
                    .filter(|&&n| membership.get(&n) == Some(&c))
                    .map(|&n| graph.degree(n) as f64)
                    .sum();
                comm_sigma_tot.insert(c, sigma);
            }

            let ki = graph.degree(node) as f64;

            // Try moving to each neighboring community
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;

            // Remove node from current community for gain calculation
            for &c in &neighbor_comms {
                let ki_in = comm_weights.get(&c).copied().unwrap_or(0.0);
                let sigma_tot = comm_sigma_tot.get(&c).copied().unwrap_or(0.0);
                let gain = ki_in / m - sigma_tot * ki / (2.0 * m * m);
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            if best_comm != current_comm {
                membership.insert(node, best_comm);
                improved = true;
            }
        }
    }

    // Renumber communities to 0..num_comms
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for node in &nodes {
        let c = membership[node];
        let mapped = *remap.entry(c).or_insert_with(|| { let id = next_id; next_id += 1; id });
        membership.insert(*node, mapped);
    }

    membership
}

/// Label propagation community detection.
pub fn label_propagation(graph: &Graph, max_iter: usize) -> HashMap<usize, usize> {
    let nodes = graph.nodes();
    if nodes.is_empty() { return HashMap::new(); }

    let mut labels: HashMap<usize, usize> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        labels.insert(node, i);
    }

    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;

    for _ in 0..max_iter {
        let mut order = nodes.clone();
        order.shuffle(&mut rng);
        let mut changed = false;

        for &node in &order {
            let neighbors = graph.neighbors(node);
            if neighbors.is_empty() { continue; }

            // Count labels among neighbors
            let mut label_counts: HashMap<usize, usize> = HashMap::new();
            for &nb in &neighbors {
                *label_counts.entry(labels[&nb]).or_insert(0) += 1;
            }

            // Find most common label
            let best_label = label_counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| label)
                .unwrap();

            if labels[&node] != best_label {
                labels.insert(node, best_label);
                changed = true;
            }
        }

        if !changed { break; }
    }

    // Renumber
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for node in &nodes {
        let c = labels[node];
        let mapped = *remap.entry(c).or_insert_with(|| { let id = next_id; next_id += 1; id });
        labels.insert(*node, mapped);
    }

    labels
}

/// Get communities from membership map.
pub fn get_communities(membership: &HashMap<usize, usize>) -> Vec<Vec<usize>> {
    let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&node, &comm) in membership {
        communities.entry(comm).or_default().push(node);
    }
    communities.into_values().collect()
}

/// Normalized mutual information between two partitions (simplified).
pub fn nmi(membership1: &HashMap<usize, usize>, membership2: &HashMap<usize, usize>) -> f64 {
    let nodes: Vec<usize> = membership1.keys().copied().collect();
    let n = nodes.len() as f64;
    if n == 0.0 { return 0.0; }

    // Get community sets
    let mut comms1: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut comms2: HashMap<usize, Vec<usize>> = HashMap::new();
    for &node in &nodes {
        if let Some(&c1) = membership1.get(&node) {
            comms1.entry(c1).or_default().push(node);
        }
        if let Some(&c2) = membership2.get(&node) {
            comms2.entry(c2).or_default().push(node);
        }
    }

    let mut h1 = 0.0;
    for comm in comms1.values() {
        let p = comm.len() as f64 / n;
        if p > 0.0 { h1 -= p * p.ln(); }
    }

    let mut h2 = 0.0;
    for comm in comms2.values() {
        let p = comm.len() as f64 / n;
        if p > 0.0 { h2 -= p * p.ln(); }
    }

    if h1 == 0.0 || h2 == 0.0 { return 0.0; }

    // Compute MI
    let mut mi = 0.0;
    let set2: std::collections::HashSet<usize> = comms2.keys().copied().collect();
    for c1_list in comms1.values() {
        let set1: std::collections::HashSet<usize> = c1_list.iter().copied().collect();
        for c2_list in comms2.values() {
            let set2_nodes: std::collections::HashSet<usize> = c2_list.iter().copied().collect();
            let inter = set1.intersection(&set2_nodes).count() as f64;
            if inter > 0.0 {
                let p_joint = inter / n;
                let p1 = set1.len() as f64 / n;
                let p2 = set2_nodes.len() as f64 / n;
                mi += p_joint * (p_joint / (p1 * p2)).ln();
            }
        }
    }

    2.0 * mi / (h1 + h2)
}
