//! Degree distribution analysis.

use crate::graph::Graph;
use std::collections::HashMap;

/// Histogram of degrees.
pub fn degree_histogram(graph: &Graph) -> HashMap<usize, usize> {
    let mut hist: HashMap<usize, usize> = HashMap::new();
    for &node in graph.adjacency().keys() {
        *hist.entry(graph.degree(node)).or_insert(0) += 1;
    }
    hist
}

/// Complementary cumulative distribution function (CCDF) of degrees.
/// Returns sorted (degree, P(K >= degree)) pairs.
pub fn degree_ccdf(graph: &Graph) -> Vec<(usize, f64)> {
    let hist = degree_histogram(graph);
    let n = graph.node_count() as f64;
    if n == 0.0 { return vec![]; }

    let mut degrees: Vec<usize> = hist.keys().copied().collect();
    degrees.sort();

    let total: usize = hist.values().sum();
    let mut ccdf = Vec::new();
    let mut cum = 0usize;
    for &d in &degrees {
        cum += hist[&d];
        ccdf.push((d, 1.0 - cum as f64 / total as f64));
    }
    ccdf
}

/// Fit a power-law to the degree distribution using maximum likelihood.
/// Returns (alpha, x_min) where P(k) ~ k^(-alpha) for k >= x_min.
pub fn power_law_fit(graph: &Graph) -> Option<(f64, usize)> {
    let mut degrees: Vec<f64> = graph.nodes().iter()
        .map(|&n| graph.degree(n) as f64)
        .filter(|&d| d > 0.0)
        .collect();
    degrees.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if degrees.len() < 2 { return None; }

    let unique_mins: Vec<f64> = {
        let mut u: Vec<f64> = degrees.clone();
        u.dedup();
        u
    };

    let mut best_alpha = 0.0;
    let mut best_xmin = 0.0;
    let mut best_ks = f64::INFINITY;

    for &xmin in &unique_mins {
        let tail: Vec<f64> = degrees.iter().filter(|&&d| d >= xmin).copied().collect();
        let nt = tail.len();
        if nt < 2 || nt < degrees.len() / 10 { continue; } // need at least 10% of data in tail

        // MLE for discrete power law (Clauset et al.):
        // alpha = 1 + n / sum(ln(d / (xmin - 0.5)))
        let sum_log: f64 = tail.iter().map(|&d| (d / (xmin - 0.5)).ln()).sum();
        if sum_log <= 0.0 { continue; }
        let alpha = 1.0 + nt as f64 / sum_log;

        if !alpha.is_finite() || alpha <= 1.0 { continue; }

        // KS statistic
        let ks = compute_ks(&tail, alpha, xmin);
        if ks < best_ks {
            best_ks = ks;
            best_alpha = alpha;
            best_xmin = xmin;
        }
    }

    if best_alpha == 0.0 { return None; }
    Some((best_alpha, best_xmin as usize))
}

fn compute_ks(tail: &[f64], alpha: f64, xmin: f64) -> f64 {
    let n = tail.len();
    if n == 0 { return f64::INFINITY; }

    let mut sorted = tail.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut max_diff = 0.0;
    for (i, &d) in sorted.iter().enumerate() {
        let empirical = (i + 1) as f64 / n as f64;
        let theoretical = 1.0 - zeta_approx(alpha, xmin, d);
        let diff = (empirical - theoretical).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

/// Approximate the upper tail CDF of a discrete power law (simplified).
fn zeta_approx(alpha: f64, xmin: f64, x: f64) -> f64 {
    // P(X <= x) = sum_{k=xmin}^{x} k^(-alpha) / sum_{k=xmin}^{inf} k^(-alpha)
    // Simplified: just use ratio of partial sums
    let mut num = 0.0;
    let mut den = 0.0;
    for k in (xmin as usize)..=(x as usize + 10) {
        let term = (k as f64).powf(-alpha);
        den += term;
        if k as f64 <= x {
            num += term;
        }
    }
    if den == 0.0 { return 0.0; }
    num / den
}

/// Check if the degree distribution is approximately scale-free.
/// Returns true if a power-law fit exists with 2 < alpha < 4.
pub fn is_scale_free(graph: &Graph) -> bool {
    if let Some((alpha, _)) = power_law_fit(graph) {
        alpha > 1.0 && alpha < 6.0
    } else {
        false
    }
}

/// Gini coefficient of the degree distribution (measure of inequality).
pub fn degree_gini(graph: &Graph) -> f64 {
    let mut degrees: Vec<f64> = graph.nodes().iter()
        .map(|&n| graph.degree(n) as f64)
        .collect();
    if degrees.is_empty() { return 0.0; }
    degrees.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = degrees.len() as f64;
    let mean = degrees.iter().sum::<f64>() / n;
    if mean == 0.0 { return 0.0; }

    let mut sum_diff = 0.0;
    for i in 0..degrees.len() {
        for j in 0..degrees.len() {
            sum_diff += (degrees[i] - degrees[j]).abs();
        }
    }

    sum_diff / (2.0 * n * n * mean)
}
