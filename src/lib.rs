//! # network-science
//!
//! A comprehensive network science library implementing models, centrality measures,
//! community detection, epidemic spreading, and social network analysis.

pub mod graph;
pub mod models;
pub mod centrality;
pub mod community;
pub mod small_world;
pub mod assortativity;
pub mod resilience;
pub mod epidemic;
pub mod degree_distribution;
pub mod agent_network;

pub use graph::{Graph, DirectedGraph};
pub use models::{erdos_renyi, barabasi_albert, watts_strogatz};
pub use centrality::{degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank};
pub use community::{louvain, label_propagation};
pub use small_world::{average_path_length, clustering_coefficient, small_world_coefficient};
pub use assortativity::{degree_assortativity, mixing_matrix};
pub use resilience::{node_percolation, edge_percolation, targeted_attack};
pub use epidemic::{sir_model, sis_model, epidemic_threshold};
pub use degree_distribution::{power_law_fit, is_scale_free, degree_histogram};
pub use agent_network::{AgentNetwork, AgentNetworkAnalyzer};
// Re-export Agent struct for convenience
pub use agent_network::Agent;
