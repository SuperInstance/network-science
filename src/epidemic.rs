//! Epidemic spreading models on networks.

use crate::graph::Graph;
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SirResult {
    /// Time series of S, I, R counts.
    pub s_counts: Vec<usize>,
    pub i_counts: Vec<usize>,
    pub r_counts: Vec<usize>,
    /// Final epidemic size (total ever infected).
    pub final_size: usize,
    /// Duration (time steps until I=0).
    pub duration: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SisResult {
    /// Time series of S and I counts.
    pub s_counts: Vec<usize>,
    pub i_counts: Vec<usize>,
    /// Prevalence at end of simulation.
    pub final_prevalence: usize,
    /// Duration simulated.
    pub duration: usize,
}

/// SIR model on a network.
/// - beta: transmission probability per edge per time step
/// - gamma: recovery probability per infected node per time step
/// - initially_infected: starting infected nodes
/// - max_steps: maximum simulation length
pub fn sir_model(
    graph: &Graph,
    beta: f64,
    gamma: f64,
    initially_infected: &[usize],
    max_steps: usize,
) -> SirResult {
    let nodes = graph.nodes();
    let n = nodes.len();
    let mut rng = rand::thread_rng();

    // State: 0=S, 1=I, 2=R
    let mut state: HashMap<usize, u8> = HashMap::new();
    for &node in &nodes {
        state.insert(node, 0);
    }
    for &node in initially_infected {
        if state.contains_key(&node) {
            state.insert(node, 1);
        }
    }

    let mut s_counts = Vec::new();
    let mut i_counts = Vec::new();
    let mut r_counts = Vec::new();
    let mut ever_infected: std::collections::HashSet<usize> = initially_infected.iter().copied().collect();

    for _ in 0..max_steps {
        let s = state.values().filter(|&&s| s == 0).count();
        let i = state.values().filter(|&&s| s == 1).count();
        let r = state.values().filter(|&&s| s == 2).count();
        s_counts.push(s);
        i_counts.push(i);
        r_counts.push(r);

        if i == 0 { break; }

        // Build next state
        let mut next_state = state.clone();
        for &node in &nodes {
            if state[&node] == 1 {
                // Try to infect neighbors
                for &nb in &graph.neighbors(node) {
                    if state[&nb] == 0 && rng.gen::<f64>() < beta {
                        next_state.insert(nb, 1);
                        ever_infected.insert(nb);
                    }
                }
                // Try to recover
                if rng.gen::<f64>() < gamma {
                    next_state.insert(node, 2);
                }
            }
        }
        state = next_state;
    }

    let final_size = ever_infected.len();
    let duration = i_counts.len();

    SirResult { s_counts, i_counts, r_counts, final_size, duration }
}

/// SIS model on a network.
/// Infected nodes can recover to susceptible (not immune).
pub fn sis_model(
    graph: &Graph,
    beta: f64,
    gamma: f64,
    initially_infected: &[usize],
    max_steps: usize,
) -> SisResult {
    let nodes = graph.nodes();
    let mut rng = rand::thread_rng();

    // true = infected, false = susceptible
    let mut infected: HashMap<usize, bool> = HashMap::new();
    for &node in &nodes {
        infected.insert(node, false);
    }
    for &node in initially_infected {
        if infected.contains_key(&node) {
            infected.insert(node, true);
        }
    }

    let mut s_counts = Vec::new();
    let mut i_counts = Vec::new();

    for _ in 0..max_steps {
        let s = infected.values().filter(|&&inf| !inf).count();
        let i = infected.values().filter(|&&inf| inf).count();
        s_counts.push(s);
        i_counts.push(i);

        let mut next = infected.clone();
        for &node in &nodes {
            if infected[&node] {
                // Try to infect neighbors
                for &nb in &graph.neighbors(node) {
                    if !infected[&nb] && rng.gen::<f64>() < beta {
                        next.insert(nb, true);
                    }
                }
                // Try to recover
                if rng.gen::<f64>() < gamma {
                    next.insert(node, false);
                }
            }
        }
        infected = next;
    }

    let final_prevalence = infected.values().filter(|&&inf| inf).count();
    SisResult { s_counts, i_counts, final_prevalence, duration: max_steps }
}

/// Theoretical epidemic threshold: λ_c = <k> / <k²>
/// An epidemic can spread if β/γ > λ_c.
pub fn epidemic_threshold(graph: &Graph) -> f64 {
    let degrees: Vec<f64> = graph.nodes().iter().map(|&n| graph.degree(n) as f64).collect();
    let n = degrees.len() as f64;
    if n == 0.0 { return f64::INFINITY; }

    let mean_k = degrees.iter().sum::<f64>() / n;
    let mean_k_sq = degrees.iter().map(|k| k * k).sum::<f64>() / n;

    if mean_k_sq == 0.0 { return f64::INFINITY; }
    mean_k / mean_k_sq
}

/// Run multiple SIR simulations and return average final size.
pub fn sir_average_final_size(
    graph: &Graph,
    beta: f64,
    gamma: f64,
    initial_fraction: f64,
    num_simulations: usize,
    max_steps: usize,
) -> f64 {
    let nodes = graph.nodes();
    let n = nodes.len();
    if n == 0 { return 0.0; }
    let num_initial = (n as f64 * initial_fraction).ceil() as usize;

    let mut rng = rand::thread_rng();
    let mut total = 0usize;
    for _ in 0..num_simulations {
        use rand::seq::SliceRandom;
        let mut shuffled = nodes.clone();
        shuffled.shuffle(&mut rng);
        let initial: Vec<usize> = shuffled.into_iter().take(num_initial).collect();
        let result = sir_model(graph, beta, gamma, &initial, max_steps);
        total += result.final_size;
    }
    total as f64 / num_simulations as f64
}
