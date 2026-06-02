#[cfg(test)]
mod tests {
    use network_science::*;
    use network_science::graph::{Graph, DirectedGraph};
    use network_science::models::*;
    use network_science::centrality::*;
    use network_science::community::*;
    use network_science::small_world::*;
    use network_science::assortativity::*;
    use network_science::resilience::*;
    use network_science::epidemic::*;
    use network_science::degree_distribution::*;
    use network_science::agent_network::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    // ==================== Graph Basics ====================

    #[test]
    fn test_graph_creation() {
        let g = Graph::with_n_nodes(5);
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut g = Graph::with_n_nodes(3);
        g.add_edge(0, 1);
        assert_eq!(g.edge_count(), 1);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0)); // undirected
    }

    #[test]
    fn test_remove_edge() {
        let mut g = Graph::with_n_nodes(3);
        g.add_edge(0, 1);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count(), 0);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_remove_node() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 2);
        g.remove_node(0);
        assert_eq!(g.edge_count(), 1);
        assert!(!g.has_node(0));
    }

    #[test]
    fn test_connected_components() {
        let mut g = Graph::with_n_nodes(6);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(3, 4);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 3);
    }

    #[test]
    fn test_is_connected() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        assert!(g.is_connected());
    }

    #[test]
    fn test_largest_component() {
        let mut g = Graph::with_n_nodes(6);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(3, 4);
        let lc = g.largest_component();
        assert_eq!(lc.node_count(), 3);
    }

    // ==================== Erdős-Rényi ====================

    #[test]
    fn test_erdos_renyi_basic() {
        let g = erdos_renyi(100, 0.1);
        assert_eq!(g.node_count(), 100);
        // Expected ~495 edges, should be within reasonable range
        assert!(g.edge_count() > 200);
        assert!(g.edge_count() < 800);
    }

    #[test]
    fn test_erdos_renyi_zero_prob() {
        let g = erdos_renyi(20, 0.0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_erdos_renyi_full() {
        let g = erdos_renyi(10, 1.0);
        assert_eq!(g.edge_count(), 45); // n*(n-1)/2
    }

    #[test]
    fn test_erdos_renyi_gnm() {
        let g = erdos_renyi_gnm(50, 100);
        assert_eq!(g.node_count(), 50);
        assert_eq!(g.edge_count(), 100);
    }

    // ==================== Barabási-Albert ====================

    #[test]
    fn test_barabasi_albert_basic() {
        let g = barabasi_albert(100, 3);
        assert_eq!(g.node_count(), 100);
        // Each new node adds 3 edges: 3 * (100 - 4) + 6 = 294
        assert!(g.edge_count() > 50);
    }

    #[test]
    fn test_barabasi_albert_connected() {
        let g = barabasi_albert(50, 2);
        assert!(g.is_connected());
    }

    #[test]
    fn test_barabasi_albert_hub() {
        // Early nodes should have higher degree on average
        let g = barabasi_albert(200, 3);
        let early_avg: f64 = (0..4).map(|n| g.degree(n) as f64).sum::<f64>() / 4.0;
        let late_avg: f64 = (196..200).map(|n| g.degree(n) as f64).sum::<f64>() / 4.0;
        assert!(early_avg > late_avg);
    }

    // ==================== Watts-Strogatz ====================

    #[test]
    fn test_watts_strogatz_basic() {
        let g = watts_strogatz(100, 6, 0.3);
        assert_eq!(g.node_count(), 100);
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_watts_strogatz_zero_rewire() {
        let g = watts_strogatz(20, 4, 0.0);
        assert_eq!(g.edge_count(), 20 * 2); // k/2 edges per node = 20 * 2
        assert!(g.is_connected());
    }

    #[test]
    fn test_watts_strogatz_full_rewire() {
        let g = watts_strogatz(50, 4, 1.0);
        assert_eq!(g.node_count(), 50);
        // Still should have roughly the same number of edges
        assert!(g.edge_count() > 30);
    }

    // ==================== Centrality ====================

    #[test]
    fn test_degree_centrality() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        let dc = degree_centrality(&g);
        assert_relative_eq!(dc[&0], 1.0, epsilon = 1e-10);
        for i in 1..=3 {
            assert_relative_eq!(dc[&i], 1.0 / 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_betweenness_centrality_star() {
        let mut g = Graph::with_n_nodes(5);
        // Star: node 0 at center
        for i in 1..5 {
            g.add_edge(0, i);
        }
        let bc = betweenness_centrality(&g);
        // Center node has highest betweenness
        assert!(bc[&0] > bc[&1]);
    }

    #[test]
    fn test_betweenness_centrality_line() {
        let mut g = Graph::with_n_nodes(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let bc = betweenness_centrality(&g);
        // Middle node (2) should have highest betweenness
        assert!(bc[&2] >= bc[&0]);
        assert!(bc[&2] >= bc[&4]);
    }

    #[test]
    fn test_closeness_centrality() {
        let mut g = Graph::with_n_nodes(5);
        // Star
        for i in 1..5 {
            g.add_edge(0, i);
        }
        let cc = closeness_centrality(&g);
        // Center should have highest closeness
        assert!(cc[&0] > cc[&1]);
    }

    #[test]
    fn test_eigenvector_centrality() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        let ec = eigenvector_centrality(&g, 1000, 1e-8);
        // Center node should have highest eigenvector centrality
        assert!(ec[&0] > ec[&1]);
    }

    #[test]
    fn test_pagerank_convergence() {
        let mut g = Graph::with_n_nodes(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 0);
        let pr = pagerank(&g, 0.85, 200, 1e-8);
        // All nodes should have similar PageRank in a cycle
        let vals: Vec<f64> = pr.values().copied().collect();
        let max_diff = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - vals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max_diff < 0.05);
    }

    #[test]
    fn test_pagerank_sum_to_one() {
        let g = erdos_renyi(30, 0.2);
        let pr = pagerank(&g, 0.85, 100, 1e-6);
        let sum: f64 = pr.values().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_pagerank_directed() {
        let mut dg = DirectedGraph::with_n_nodes(4);
        dg.add_edge(0, 1);
        dg.add_edge(1, 2);
        dg.add_edge(2, 0);
        dg.add_edge(2, 3);
        let pr = pagerank_directed(&dg, 0.85, 100, 1e-8);
        let sum: f64 = pr.values().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    // ==================== Community Detection ====================

    #[test]
    fn test_louvain_two_communities() {
        // Create two cliques connected by a single edge
        let mut g = Graph::with_n_nodes(20);
        // Clique 1: 0-9
        for i in 0..10 {
            for j in (i+1)..10 {
                g.add_edge(i, j);
            }
        }
        // Clique 2: 10-19
        for i in 10..20 {
            for j in (i+1)..20 {
                g.add_edge(i, j);
            }
        }
        // Single bridge
        g.add_edge(5, 15);

        let membership = louvain(&g);
        let comms = get_communities(&membership);
        assert!(comms.len() >= 2);
    }

    #[test]
    fn test_louvain_modularity_positive() {
        let mut g = Graph::with_n_nodes(20);
        for i in 0..10 {
            for j in (i+1)..10 {
                g.add_edge(i, j);
            }
        }
        for i in 10..20 {
            for j in (i+1)..20 {
                g.add_edge(i, j);
            }
        }
        g.add_edge(5, 15);

        let membership = louvain(&g);
        let q = modularity(&g, &membership);
        assert!(q > 0.0);
    }

    #[test]
    fn test_label_propagation() {
        let mut g = Graph::with_n_nodes(20);
        for i in 0..10 {
            for j in (i+1)..10 {
                g.add_edge(i, j);
            }
        }
        for i in 10..20 {
            for j in (i+1)..20 {
                g.add_edge(i, j);
            }
        }
        g.add_edge(5, 15);

        let membership = label_propagation(&g, 100);
        assert!(!membership.is_empty());
        // Should find at least 2 communities
        let comms = get_communities(&membership);
        assert!(comms.len() >= 2);
    }

    #[test]
    fn test_nmi_same_partition() {
        let mut m1 = HashMap::new();
        let mut m2 = HashMap::new();
        for i in 0..10 {
            m1.insert(i, i % 3);
            m2.insert(i, i % 3);
        }
        let score = nmi(&m1, &m2);
        assert_relative_eq!(score, 1.0, epsilon = 0.01);
    }

    // ==================== Small World ====================

    #[test]
    fn test_clustering_coefficient_complete() {
        let mut g = Graph::with_n_nodes(5);
        for i in 0..5 {
            for j in (i+1)..5 {
                g.add_edge(i, j);
            }
        }
        let cc = clustering_coefficient(&g);
        assert_relative_eq!(cc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clustering_coefficient_ring() {
        let mut g = Graph::with_n_nodes(6);
        for i in 0..6 {
            g.add_edge(i, (i + 1) % 6);
        }
        let cc = clustering_coefficient(&g);
        assert_eq!(cc, 0.0); // No triangles in a ring
    }

    #[test]
    fn test_average_path_length() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let apl = average_path_length(&g);
        // Distances: 0-1:1, 0-2:2, 0-3:3, 1-2:1, 1-3:2, 2-3:1
        // Average = (1+2+3+1+2+1)/6 = 10/6
        assert_relative_eq!(apl, 10.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_small_world_coefficient_ws() {
        // Watts-Strogatz with small beta should have high clustering
        let g = watts_strogatz(100, 6, 0.1);
        let (sigma, gamma, _lambda) = small_world_metrics(&g);
        // Gamma should be > 1 (higher clustering than random)
        assert!(gamma > 1.0);
    }

    #[test]
    fn test_transitivity() {
        let mut g = Graph::with_n_nodes(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        let t = transitivity(&g);
        assert_relative_eq!(t, 1.0, epsilon = 1e-10);
    }

    // ==================== Assortativity ====================

    #[test]
    fn test_assortativity_complete() {
        let mut g = Graph::with_n_nodes(5);
        for i in 0..5 {
            for j in (i+1)..5 {
                g.add_edge(i, j);
            }
        }
        let r = degree_assortativity(&g);
        // Complete graph: all degrees equal, assortativity undefined/zero
        // All nodes have degree 4, correlation is undefined
        assert!(r.is_nan() || r.abs() < 0.01 || r == 0.0);
    }

    #[test]
    fn test_mixing_matrix() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let (matrix, degs) = mixing_matrix(&g);
        assert!(!matrix.is_empty());
    }

    #[test]
    fn test_knn() {
        let mut g = Graph::with_n_nodes(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let knn_vals = knn(&g);
        assert!(!knn_vals.is_empty());
    }

    // ==================== Resilience ====================

    #[test]
    fn test_node_percolation() {
        let g = erdos_renyi(50, 0.1);
        let result = node_percolation(&g, 10);
        assert!(result.fraction_removed.len() > 1);
        // First should be fully connected
        assert_relative_eq!(result.largest_component_fraction[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_edge_percolation() {
        let g = erdos_renyi(50, 0.15);
        let result = edge_percolation(&g, 10);
        assert!(result.fraction_removed.len() > 1);
    }

    #[test]
    fn test_targeted_attack() {
        let g = barabasi_albert(50, 2);
        let result = targeted_attack(&g, 10);
        assert!(result.fraction_removed.len() > 1);
    }

    #[test]
    fn test_critical_threshold() {
        let g = erdos_renyi(100, 0.05);
        let tc = critical_threshold(&g);
        assert!(tc >= 0.0 && tc <= 1.0);
    }

    // ==================== Epidemic ====================

    #[test]
    fn test_sir_model_basic() {
        let g = erdos_renyi(50, 0.1);
        let result = sir_model(&g, 0.5, 0.1, &[0], 100);
        assert!(result.s_counts.len() > 0);
        assert_eq!(result.s_counts.iter().sum::<usize>()
            + result.i_counts.iter().sum::<usize>()
            + result.r_counts.iter().sum::<usize>(),
            50 * result.s_counts.len() // total preserved
        );
    }

    #[test]
    fn test_sir_zero_beta() {
        let g = erdos_renyi(20, 0.2);
        let result = sir_model(&g, 0.0, 0.5, &[0], 50);
        // With beta=0, only initial node gets infected
        assert!(result.final_size <= 1);
    }

    #[test]
    fn test_sir_full_recovery() {
        let g = erdos_renyi(30, 0.15);
        let result = sir_model(&g, 0.3, 1.0, &[0], 100);
        // With gamma=1, infected recover immediately
        // Very limited spread
    }

    #[test]
    fn test_sis_model() {
        let g = erdos_renyi(30, 0.15);
        let result = sis_model(&g, 0.3, 0.1, &[0, 1, 2], 50);
        assert!(result.s_counts.len() > 0);
        for i in 0..result.s_counts.len() {
            assert_eq!(result.s_counts[i] + result.i_counts[i], 30);
        }
    }

    #[test]
    fn test_epidemic_threshold() {
        let g = erdos_renyi(100, 0.05);
        let lambda_c = epidemic_threshold(&g);
        assert!(lambda_c > 0.0);
        assert!(lambda_c < 1.0);
    }

    // ==================== Degree Distribution ====================

    #[test]
    fn test_degree_histogram() {
        let g = erdos_renyi(100, 0.1);
        let hist = degree_histogram(&g);
        let total: usize = hist.values().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_power_law_fit_ba() {
        let g = barabasi_albert(500, 3);
        let fit = power_law_fit(&g);
        assert!(fit.is_some());
        let (alpha, _) = fit.unwrap();
        // BA model should have alpha > 1
        assert!(alpha > 1.0 && alpha < 8.0);
    }

    #[test]
    fn test_is_scale_free_ba() {
        let g = barabasi_albert(500, 3);
        assert!(is_scale_free(&g));
    }

    #[test]
    fn test_degree_gini() {
        let mut g = Graph::with_n_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        let gini = degree_gini(&g);
        assert!(gini > 0.0 && gini <= 1.0);
    }

    #[test]
    fn test_degree_ccdf() {
        let g = erdos_renyi(50, 0.1);
        let ccdf = degree_ccdf(&g);
        assert!(!ccdf.is_empty());
        // CCDF should be monotonically non-increasing
        for i in 1..ccdf.len() {
            assert!(ccdf[i].1 <= ccdf[i-1].1 + 1e-10);
        }
    }

    // ==================== Directed Graph ====================

    #[test]
    fn test_directed_graph() {
        let mut dg = DirectedGraph::with_n_nodes(3);
        dg.add_edge(0, 1);
        dg.add_edge(1, 2);
        assert_eq!(dg.edge_count(), 2);
        assert_eq!(dg.out_degree(0), 1);
        assert_eq!(dg.in_degree(1), 1);
    }

    #[test]
    fn test_directed_to_undirected() {
        let mut dg = DirectedGraph::with_n_nodes(3);
        dg.add_edge(0, 1);
        dg.add_edge(1, 0);
        dg.add_edge(1, 2);
        let ug = dg.to_undirected();
        assert_eq!(ug.edge_count(), 2); // 0-1 and 1-2
    }

    #[test]
    fn test_directed_erdos_renyi() {
        let dg = directed_erdos_renyi(30, 0.2);
        assert_eq!(dg.node_count(), 30);
        assert!(dg.edge_count() > 50);
    }

    // ==================== Agent Network ====================

    #[test]
    fn test_agent_network_basic() {
        let mut net = AgentNetwork::new();
        for i in 0..10 {
            net.add_agent(Agent {
                id: i,
                name: format!("Agent-{}", i),
                attributes: HashMap::new(),
            });
        }
        net.add_communication_link(0, 1, 1.0);
        net.add_communication_link(1, 2, 2.0);
        assert_eq!(net.agent_count(), 10);
        assert_eq!(net.graph.edge_count(), 2);
    }

    #[test]
    fn test_agent_network_communities() {
        let mut net = AgentNetwork::new();
        for i in 0..20 {
            net.add_agent(Agent {
                id: i,
                name: format!("Agent-{}", i),
                attributes: HashMap::new(),
            });
        }
        // Two groups
        for i in 0..10 {
            for j in (i+1)..10 {
                net.add_communication_link(i, j, 1.0);
            }
        }
        for i in 10..20 {
            for j in (i+1)..20 {
                net.add_communication_link(i, j, 1.0);
            }
        }
        net.add_communication_link(5, 15, 0.5);

        let comms = net.detect_communities();
        assert!(comms.len() >= 2);
    }

    #[test]
    fn test_agent_network_influential() {
        let mut net = AgentNetwork::new();
        for i in 0..10 {
            net.add_agent(Agent {
                id: i,
                name: format!("Agent-{}", i),
                attributes: HashMap::new(),
            });
        }
        // Hub agent 0
        for i in 1..10 {
            net.add_communication_link(0, i, 1.0);
        }
        let top = net.most_influential(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 0); // Agent 0 should be most influential
    }

    #[test]
    fn test_agent_network_summary() {
        let mut net = AgentNetwork::new();
        for i in 0..8 {
            net.add_agent(Agent {
                id: i,
                name: format!("Agent-{}", i),
                attributes: HashMap::new(),
            });
        }
        for i in 0..8 {
            net.add_communication_link(i, (i + 1) % 8, 1.0);
        }
        let summary = net.summary();
        assert_eq!(summary.num_agents, 8);
        assert!(summary.density > 0.0);
    }

    #[test]
    fn test_agent_network_from_log() {
        let log = vec![
            (0, 1, 5.0),
            (1, 2, 3.0),
            (2, 3, 1.0),
        ];
        let agents: Vec<Agent> = (0..4).map(|i| Agent {
            id: i,
            name: format!("Agent-{}", i),
            attributes: HashMap::new(),
        }).collect();
        let net = AgentNetworkAnalyzer::from_communication_log(&log, agents);
        assert_eq!(net.agent_count(), 4);
        assert_eq!(net.graph.edge_count(), 3);
    }

    #[test]
    fn test_agent_network_comparison() {
        let mut net1 = AgentNetwork::new();
        let mut net2 = AgentNetwork::new();
        for i in 0..8 {
            net1.add_agent(Agent { id: i, name: format!("A{}", i), attributes: HashMap::new() });
            net2.add_agent(Agent { id: i, name: format!("B{}", i), attributes: HashMap::new() });
        }
        // Dense ER graph
        for i in 0..8 {
            for j in (i+1)..8 {
                if (i + j) % 3 != 0 { // ~2/3 of edges
                    net1.add_communication_link(i, j, 1.0);
                }
            }
        }
        // Sparse ring
        for i in 0..8 {
            net2.add_communication_link(i, (i+1)%8, 1.0);
        }
        let comp = AgentNetworkAnalyzer::compare(&net1, &net2);
        assert!(comp.density_diff > 0.0);
    }

    #[test]
    fn test_bridge_agents() {
        let mut net = AgentNetwork::new();
        for i in 0..7 {
            net.add_agent(Agent { id: i, name: format!("A{}", i), attributes: HashMap::new() });
        }
        // Two clusters with bridge node 3
        net.add_communication_link(0, 1, 1.0);
        net.add_communication_link(1, 2, 1.0);
        net.add_communication_link(2, 3, 1.0);
        net.add_communication_link(3, 4, 1.0);
        net.add_communication_link(4, 5, 1.0);
        net.add_communication_link(5, 6, 1.0);
        let bridges = net.bridge_agents(3);
        assert!(bridges.len() <= 3);
        // Node 3 should have highest betweenness in a line graph
        assert_eq!(bridges[0].0, 3);
    }

    #[test]
    fn test_random_agent_network() {
        let names: Vec<String> = (0..20).map(|i| format!("Agent-{}", i)).collect();
        let net = AgentNetworkAnalyzer::random_network(20, 0.2, &names);
        assert_eq!(net.agent_count(), 20);
        assert!(net.graph.edge_count() > 10);
    }

    // ==================== Integration ====================

    #[test]
    fn test_full_pipeline() {
        // Generate a BA network, analyze it
        let g = barabasi_albert(50, 3);

        // Centrality
        let dc = degree_centrality(&g);
        assert_eq!(dc.len(), 50);

        let bc = betweenness_centrality(&g);
        assert_eq!(bc.len(), 50);

        let pr = pagerank(&g, 0.85, 100, 1e-6);
        let pr_sum: f64 = pr.values().sum();
        assert_relative_eq!(pr_sum, 1.0, epsilon = 0.01);

        // Community
        let membership = louvain(&g);
        let q = modularity(&g, &membership);
        assert!(q > -1.0 && q < 1.0);

        // Small world
        let cc = clustering_coefficient(&g);
        assert!(cc >= 0.0 && cc <= 1.0);

        // Scale free - small network, just verify function runs
        let _sf = is_scale_free(&g);
    }
}
