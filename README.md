# network-science

Network science in Rust. Who's the most important node? What communities exist? How does information spread?

## Features

- **Graph structures** — undirected and directed graphs with adjacency lists
- **Centrality** — degree, betweenness (Brandes 2001), closeness, eigenvector, PageRank
- **Community detection** — Louvain (Blondel et al. 2008), label propagation
- **Epidemic models** — SIR, SIS, epidemic thresholds
- **Degree distributions** — power-law MLE (Clauset–Shalizi–Newman), scale-free detection
- **Network models** — Erdős–Rényi, Barabási–Albert preferential attachment, Watts–Strogatz small world
- **Small-world analysis** — clustering coefficient, average path length, small-world coefficient
- **Resilience** — node/edge percolation, targeted attacks
- **Assortativity** — degree assortativity, mixing matrices
- **Social networks** — agent-based network modeling, community summaries, comparison

## Quick start

```toml
[dependencies]
network-science = "0.1"
```

```rust
use network_science::{Graph, degree_centrality, clustering_coefficient};

let mut g = Graph::new();
g.add_edge(0, 1);
g.add_edge(1, 2);
g.add_edge(2, 0);

let dc = degree_centrality(&g);
let cc = clustering_coefficient(&g);
```

## License

MIT OR Apache-2.0
