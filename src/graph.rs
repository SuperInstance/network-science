use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// An undirected graph using adjacency list representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    /// Adjacency list: node -> set of neighbors
    adj: HashMap<usize, HashSet<usize>>,
    n: usize,
    m: usize,
    directed: bool,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            adj: HashMap::new(),
            n: 0,
            m: 0,
            directed: false,
        }
    }

    /// Create a graph with `n` nodes (0..n) and no edges.
    pub fn with_n_nodes(n: usize) -> Self {
        let mut adj = HashMap::new();
        for i in 0..n {
            adj.insert(i, HashSet::new());
        }
        Graph { adj, n, m: 0, directed: false }
    }

    /// Ensure node exists in the graph.
    pub fn add_node(&mut self, node: usize) {
        if !self.adj.contains_key(&node) {
            self.adj.insert(node, HashSet::new());
            self.n = self.n.max(node + 1);
        }
    }

    /// Add an undirected edge between u and v.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v { return; } // no self-loops by default
        self.add_node(u);
        self.add_node(v);
        if !self.adj[&u].contains(&v) {
            self.adj.get_mut(&u).unwrap().insert(v);
            self.adj.get_mut(&v).unwrap().insert(u);
            self.m += 1;
        }
    }

    /// Remove an undirected edge.
    pub fn remove_edge(&mut self, u: usize, v: usize) {
        if let Some(neighbors) = self.adj.get_mut(&u) {
            if neighbors.remove(&v) {
                self.m -= 1;
            }
        }
        if let Some(neighbors) = self.adj.get_mut(&v) {
            neighbors.remove(&u);
        }
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&mut self, node: usize) {
        if let Some(neighbors) = self.adj.remove(&node) {
            for &nb in &neighbors {
                if let Some(nbs) = self.adj.get_mut(&nb) {
                    nbs.remove(&node);
                    self.m -= 1;
                }
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.adj.len()
    }

    pub fn edge_count(&self) -> usize {
        self.m
    }

    pub fn nodes(&self) -> Vec<usize> {
        self.adj.keys().copied().collect()
    }

    pub fn has_node(&self, node: usize) -> bool {
        self.adj.contains_key(&node)
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adj.get(&u).map_or(false, |s| s.contains(&v))
    }

    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj.get(&node).map_or(vec![], |s| s.iter().copied().collect())
    }

    pub fn degree(&self, node: usize) -> usize {
        self.adj.get(&node).map_or(0, |s| s.len())
    }

    pub fn degrees(&self) -> Vec<(usize, usize)> {
        self.adj.keys().map(|&n| (n, self.degree(n))).collect()
    }

    /// Get all edges as (u, v) pairs with u < v.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        let mut seen = HashSet::new();
        for (&u, nbs) in &self.adj {
            for &v in nbs {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                if seen.insert((a, b)) {
                    edges.push((a, b));
                }
            }
        }
        edges
    }

    /// BFS shortest path lengths from `source` to all reachable nodes.
    pub fn bfs_distances(&self, source: usize) -> HashMap<usize, usize> {
        let mut dist = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        dist.insert(source, 0);
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            let d = dist[&u];
            for &v in &self.adj[&u] {
                if !dist.contains_key(&v) {
                    dist.insert(v, d + 1);
                    queue.push_back(v);
                }
            }
        }
        dist
    }

    /// Check if the graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.adj.is_empty() { return true; }
        let start = *self.adj.keys().next().unwrap();
        let dist = self.bfs_distances(start);
        dist.len() == self.adj.len()
    }

    /// Connected components.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        for &node in self.adj.keys() {
            if visited.contains(&node) { continue; }
            let dist = self.bfs_distances(node);
            let comp: Vec<usize> = dist.keys().copied().collect();
            visited.extend(comp.iter().copied());
            components.push(comp);
        }
        components
    }

    /// Get the adjacency as a HashMap (for external use).
    pub fn adjacency(&self) -> &HashMap<usize, HashSet<usize>> {
        &self.adj
    }

    /// Get the largest connected component as a new Graph.
    pub fn largest_component(&self) -> Graph {
        let components = self.connected_components();
        let largest = components.into_iter().max_by_key(|c| c.len()).unwrap_or_default();
        let node_set: HashSet<usize> = largest.into_iter().collect();
        let mut g = Graph::new();
        for &u in &node_set {
            for &v in &self.adj[&u] {
                if node_set.contains(&v) && u < v {
                    g.add_edge(u, v);
                }
            }
        }
        g
    }
}

/// A directed graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectedGraph {
    /// out_edges[i] = set of nodes i points to
    out_edges: HashMap<usize, HashSet<usize>>,
    /// in_edges[i] = set of nodes pointing to i
    in_edges: HashMap<usize, HashSet<usize>>,
    n: usize,
    m: usize,
}

impl DirectedGraph {
    pub fn new() -> Self {
        DirectedGraph {
            out_edges: HashMap::new(),
            in_edges: HashMap::new(),
            n: 0,
            m: 0,
        }
    }

    pub fn with_n_nodes(n: usize) -> Self {
        let mut out_edges = HashMap::new();
        let mut in_edges = HashMap::new();
        for i in 0..n {
            out_edges.insert(i, HashSet::new());
            in_edges.insert(i, HashSet::new());
        }
        DirectedGraph { out_edges, in_edges, n, m: 0 }
    }

    pub fn add_node(&mut self, node: usize) {
        if !self.out_edges.contains_key(&node) {
            self.out_edges.insert(node, HashSet::new());
            self.in_edges.insert(node, HashSet::new());
            self.n = self.n.max(node + 1);
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        if from == to { return; }
        self.add_node(from);
        self.add_node(to);
        if !self.out_edges[&from].contains(&to) {
            self.out_edges.get_mut(&from).unwrap().insert(to);
            self.in_edges.get_mut(&to).unwrap().insert(from);
            self.m += 1;
        }
    }

    pub fn node_count(&self) -> usize { self.out_edges.len() }
    pub fn edge_count(&self) -> usize { self.m }
    pub fn out_degree(&self, node: usize) -> usize {
        self.out_edges.get(&node).map_or(0, |s| s.len())
    }
    pub fn in_degree(&self, node: usize) -> usize {
        self.in_edges.get(&node).map_or(0, |s| s.len())
    }
    pub fn successors(&self, node: usize) -> Vec<usize> {
        self.out_edges.get(&node).map_or(vec![], |s| s.iter().copied().collect())
    }
    pub fn predecessors(&self, node: usize) -> Vec<usize> {
        self.in_edges.get(&node).map_or(vec![], |s| s.iter().copied().collect())
    }
    pub fn nodes(&self) -> Vec<usize> {
        self.out_edges.keys().copied().collect()
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.out_edges.get(&from).map_or(false, |s| s.contains(&to))
    }
    pub fn out_edges_map(&self) -> &HashMap<usize, HashSet<usize>> {
        &self.out_edges
    }

    /// Convert to undirected Graph.
    pub fn to_undirected(&self) -> Graph {
        let mut g = Graph::new();
        for (&from, tos) in &self.out_edges {
            for &to in tos {
                g.add_edge(from, to);
            }
        }
        g
    }
}
