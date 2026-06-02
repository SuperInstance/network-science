//! Agent social network modeling and analysis.

use crate::graph::{Graph, DirectedGraph};
use crate::centrality::{degree_centrality, betweenness_centrality, pagerank};
use crate::community::{louvain, label_propagation, get_communities, modularity};
use crate::small_world::{clustering_coefficient, average_path_length};
use crate::assortativity::degree_assortativity;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An agent in the social network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    pub id: usize,
    pub name: String,
    pub attributes: HashMap<String, String>,
}

/// An agent social network with metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentNetwork {
    pub graph: Graph,
    pub agents: HashMap<usize, Agent>,
    pub interaction_weights: HashMap<(usize, usize), f64>,
}

impl AgentNetwork {
    pub fn new() -> Self {
        AgentNetwork {
            graph: Graph::new(),
            agents: HashMap::new(),
            interaction_weights: HashMap::new(),
        }
    }

    /// Add an agent to the network.
    pub fn add_agent(&mut self, agent: Agent) {
        self.graph.add_node(agent.id);
        self.agents.insert(agent.id, agent);
    }

    /// Add a communication link between two agents.
    pub fn add_communication_link(&mut self, from: usize, to: usize, weight: f64) {
        self.graph.add_edge(from, to);
        self.interaction_weights.insert((from.min(to), from.max(to)), weight);
    }

    /// Record an interaction between agents (adds or strengthens link).
    pub fn record_interaction(&mut self, from: usize, to: usize) {
        let key = (from.min(to), from.max(to));
        *self.interaction_weights.entry(key).or_insert(0.0) += 1.0;
        self.graph.add_edge(from, to);
    }

    /// Get the number of agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get agent by ID.
    pub fn get_agent(&self, id: usize) -> Option<&Agent> {
        self.agents.get(&id)
    }

    /// Find the most influential agents by PageRank.
    pub fn most_influential(&self, top_k: usize) -> Vec<(usize, f64)> {
        let pr = pagerank(&self.graph, 0.85, 100, 1e-6);
        let mut ranked: Vec<(usize, f64)> = pr.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.into_iter().take(top_k).collect()
    }

    /// Detect agent communities.
    pub fn detect_communities(&self) -> HashMap<usize, Vec<usize>> {
        let membership = louvain(&self.graph);
        let communities = get_communities(&membership);
        let mut result = HashMap::new();
        for (i, comm) in communities.into_iter().enumerate() {
            result.insert(i, comm);
        }
        result
    }

    /// Identify bridge agents (high betweenness).
    pub fn bridge_agents(&self, top_k: usize) -> Vec<(usize, f64)> {
        let bc = betweenness_centrality(&self.graph);
        let mut ranked: Vec<(usize, f64)> = bc.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.into_iter().take(top_k).collect()
    }

    /// Compute network density.
    pub fn density(&self) -> f64 {
        let n = self.graph.node_count();
        if n <= 1 { return 0.0; }
        let max_edges = n * (n - 1) / 2;
        self.graph.edge_count() as f64 / max_edges as f64
    }

    /// Summary statistics.
    pub fn summary(&self) -> AgentNetworkSummary {
        let pr = pagerank(&self.graph, 0.85, 100, 1e-6);
        let membership = louvain(&self.graph);
        let n_communities = {
            let mut comms: Vec<usize> = membership.values().copied().collect();
            comms.sort();
            comms.dedup();
            comms.len()
        };

        AgentNetworkSummary {
            num_agents: self.agents.len(),
            num_links: self.graph.edge_count(),
            density: self.density(),
            clustering_coefficient: clustering_coefficient(&self.graph),
            average_path_length: average_path_length(&self.graph),
            assortativity: degree_assortativity(&self.graph),
            num_communities: n_communities,
            modularity: modularity(&self.graph, &membership),
        }
    }
}

/// Summary statistics for an agent network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNetworkSummary {
    pub num_agents: usize,
    pub num_links: usize,
    pub density: f64,
    pub clustering_coefficient: f64,
    pub average_path_length: f64,
    pub assortativity: f64,
    pub num_communities: usize,
    pub modularity: f64,
}

/// Analyzer for comparing agent networks.
pub struct AgentNetworkAnalyzer;

impl AgentNetworkAnalyzer {
    /// Compare two agent networks by their structural properties.
    pub fn compare(net1: &AgentNetwork, net2: &AgentNetwork) -> NetworkComparison {
        let s1 = net1.summary();
        let s2 = net2.summary();
        NetworkComparison {
            density_diff: s1.density - s2.density,
            clustering_diff: s1.clustering_coefficient - s2.clustering_coefficient,
            path_length_diff: s1.average_path_length - s2.average_path_length,
            community_diff: s1.num_communities as f64 - s2.num_communities as f64,
        }
    }

    /// Build an agent network from a communication log.
    /// Each entry is (sender_id, receiver_id, message_count).
    pub fn from_communication_log(log: &[(usize, usize, f64)], agents: Vec<Agent>) -> AgentNetwork {
        let mut net = AgentNetwork::new();
        for agent in agents {
            net.add_agent(agent);
        }
        for &(from, to, weight) in log {
            net.add_communication_link(from, to, weight);
        }
        net
    }

    /// Generate a random agent network using the Erdős-Rényi model.
    pub fn random_network(num_agents: usize, connection_prob: f64, agent_names: &[String]) -> AgentNetwork {
        let mut net = AgentNetwork::new();
        for i in 0..num_agents {
            let name = agent_names.get(i).cloned().unwrap_or_else(|| format!("Agent-{}", i));
            net.add_agent(Agent {
                id: i,
                name,
                attributes: HashMap::new(),
            });
        }
        let g = crate::models::erdos_renyi(num_agents, connection_prob);
        net.graph = g;
        net
    }
}

/// Comparison between two networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkComparison {
    pub density_diff: f64,
    pub clustering_diff: f64,
    pub path_length_diff: f64,
    pub community_diff: f64,
}
